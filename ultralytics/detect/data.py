# -*- coding: utf-8 -*-
"""
Last modified: 2025-11-07
Function:
Generate HDF5 (chr1–22 + chrX, including images and positions) and PNG/TXT
(from a balanced .cool file), with specified train/val split and negative-sample filtering.

Key constraints:
- PNG/TXT:
  Train = chr1, chr3–chr16, chr20–chr22
  Val   = chr2, chr17–chr19
  chrX is NOT exported to PNG/TXT
- HDF5:
  Must store ALL windows from chr1–chr22 + chrX (no split),
  and must include a "positions" dataset (genomic interval string per window)
- Negative sample filtering:
  Windows without labels are NOT written to PNG/TXT
  (but ARE still written to HDF5, with continuous indexing)
- Label deduplication:
  Deduplicate (start_px, end_px) within each window
- Boundary-safe windows:
  Use string-based genomic fetch + right-end clamping + internal padding
"""

import os
import math
import h5py
import numpy as np
import pandas as pd
import cooler
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml

# ========================= Path configuration =========================
COOL_PATH    = "/storx/hltao2/TAD/yolov8/process/GM12878/GM12878_10k.cool"
BED_PATH     = "/storx/hltao2/TAD/find_TAD/2014GM12878_10k_output/HTAD.bed"   # <- replace with your BED file
OUT_ROOT     = "/storx/hltao2/yolo--/111"                                   # output root
OUT_H5       = os.path.join(OUT_ROOT, "data.h5")
OUT_IMG_TRAIN= os.path.join(OUT_ROOT, "images", "train")
OUT_IMG_VAL  = os.path.join(OUT_ROOT, "images", "val")
OUT_LAB_TRAIN= os.path.join(OUT_ROOT, "labels", "train")
OUT_LAB_VAL  = os.path.join(OUT_ROOT, "labels", "val")
DATA_YAML    = os.path.join(OUT_ROOT, "data.yaml")        # for YOLO


# ========================= Hyperparameters =========================
RES_BP   = 10_000     # 10 kb (for bookkeeping only; actual resolution follows .cool binsize)
WIN_SIZE = 224        # window size (bins)
STRIDE   = 112        # sliding stride (bins)
MAXPCTL  = 0.998      # quantile (estimated per chromosome)

# ========================= Utility functions =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def window_iter(chrom_size_bp, bin_size, win_bins, stride_bins):
    """Generate window start bins and ensure coverage of the chromosome end."""
    total_bins = int(math.ceil(chrom_size_bp / bin_size))
    if total_bins <= 0:
        return [0], 0
    if total_bins <= win_bins:
        return [0], total_bins
    starts = list(range(0, total_bins - win_bins + 1, stride_bins))
    last_start = total_bins - win_bins
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts, total_bins

def build_chrom_list_for_h5(cool_names):
    """
    HDF5 requirement:
    chr1..chr22 + chrX (or 1..22 + X).
    If .cool uses no 'chr' prefix, fall back to 1..22 + X.
    Only chromosomes that actually exist in .cool are returned.
    """
    names = set(map(str, cool_names))
    with_chr    = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    without_chr = [str(i)   for i in range(1, 23)] + ["X"]

    out = []
    for a, b in zip(with_chr, without_chr):
        if a in names:
            out.append(a)
        elif b in names:
            out.append(b)
    if ("chrX" in names) and ("chrX" not in out):
        out.append("chrX")
    if ("X" in names) and ("X" not in out):
        out.append("X")
    return out

def build_train_val_sets():
    """
    Train/val split ONLY for PNG/TXT output (HDF5 ignores this split).

    Train = chr1, chr3–chr16, chr20–chr22
    Val   = chr2, chr17–chr19

    Also provide non-'chr' versions to support different .cool naming styles.
    """
    train_chr = ["chr1"] + [f"chr{i}" for i in range(3, 17)] + [f"chr{i}" for i in range(20, 23)]
    val_chr   = ["chr2"] + [f"chr{i}" for i in range(17, 20)]
    train_no  = ["1"] + [str(i) for i in range(3, 17)] + [str(i) for i in range(20, 23)]
    val_no    = ["2"] + [str(i) for i in range(17, 20)]
    return set(train_chr), set(val_chr), set(train_no), set(val_no)

def normalize_name_to_used(target, available_names):
    """
    Map between 'chr'-prefixed and non-prefixed chromosome names
    to a name that actually exists in available_names.
    """
    if target in available_names:
        return target
    if target.startswith("chr"):
        alt = target[3:]
    else:
        alt = "chr" + target
    return alt if alt in available_names else None

# ========================= BED loading and label conversion =========================
def read_bed_as_intervals(bed_path, default_score=1.0):
    """
    Read TAD intervals (3-column BED or BED with header) into:
    {chrom: [(start_bp, end_bp, score), ...]}
    """
    try:
        df = pd.read_csv(bed_path, sep='\t', header=None, usecols=[0, 1, 2],
                         names=['chrom', 'startbp', 'endbp'])
    except Exception:
        df = pd.read_csv(bed_path, sep='\t', comment='#')
        cols = {c.lower(): c for c in df.columns}
        df = df.rename(columns={
            cols.get('chrom', 'chrom'): 'chrom',
            cols.get('start', 'startbp'): 'startbp',
            cols.get('end', 'endbp'): 'endbp'
        })
        df = df[['chrom', 'startbp', 'endbp']]

    df['startbp'] = pd.to_numeric(df['startbp'], errors='coerce')
    df['endbp']   = pd.to_numeric(df['endbp'], errors='coerce')
    df = df.dropna()
    df = df[df['endbp'] > df['startbp']]
    df['score'] = default_score

    tad_by_chr = {}
    for chrom, sub in df.groupby('chrom'):
        tad_by_chr[chrom] = list(
            sub[['startbp', 'endbp', 'score']].itertuples(index=False, name=None)
        )
    return tad_by_chr

def intersect_tads_with_window(tads_bp, win_start_bp, win_end_bp):
    """Select TAD intervals (bp) that overlap with the given window."""
    res = []
    for s, e, _score in tads_bp:
        if e <= win_start_bp or s >= win_end_bp:
            continue
        ss = max(s, win_start_bp)
        ee = min(e, win_end_bp)
        if ee > ss:
            res.append((ss, ee))
    return res

def tads_to_yolo_labels_dedup(tads_bp_in_win, win_start_bp, resolution_bp, image_size):
    """
    Convert window-local TADs (bp) to YOLO boxes (square boxes on main diagonal),
    with deduplication.

    Constraint:
    0 <= start_px < image_size and 0 < end_px <= image_size
    """
    lines = []
    seen = set()
    for ss, ee in tads_bp_in_win:
        start_px = int((ss - win_start_bp) / resolution_bp)
        end_px   = int((ee - win_start_bp) / resolution_bp)

        if not (0 <= start_px < image_size and 0 < end_px <= image_size):
            continue
        if end_px <= start_px:
            continue

        key = (start_px, end_px)
        if key in seen:
            continue
        seen.add(key)

        xmin = start_px
        ymin = start_px
        xmax = end_px
        ymax = end_px

        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        w  = (xmax - xmin)
        h  = (ymax - ymin)

        cx_n, cy_n = cx / image_size, cy / image_size
        w_n,  h_n  = w  / image_size, h  / image_size

        if not (0 <= cx_n <= 1 and 0 <= cy_n <= 1 and 0 < w_n <= 1 and 0 < h_n <= 1):
            continue

        lines.append(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
    return lines
