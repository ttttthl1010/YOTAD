# -*- coding: utf-8 -*-
"""
Train directly from data.yaml (224×224, no change to data generation)
Goals:
- Faster convergence
- Preserve accuracy
- Retain partial recall (R) improvement
"""

import yaml
import sys
import os
from pathlib import Path
from ultralytics import YOLO


def train_from_yaml(
    data_yaml_path,
    project_dir,
    model_cfg="",
    # —— Training scale (target convergence within ~50 epochs) —— #
    epochs=60,
    imgsz=224,
    batch=32,
    device="0",
):
    data_yaml_path = os.path.abspath(data_yaml_path)
    project_dir = os.path.abspath(project_dir)
    os.makedirs(project_dir, exist_ok=True)

    print(f"[INFO] Using data config: {data_yaml_path}")
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(f"[INFO] train = {cfg.get('train')}")
    print(f"[INFO] val   = {cfg.get('val')}")
    print(f"[INFO] nc    = {cfg.get('nc')}, names = {cfg.get('names')}")

    model = YOLO(model_cfg)

    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,

        project=project_dir,
        name='train',
        exist_ok=True,

        pretrained=False,

        # —— Optimizer & learning rate (larger early steps, weaker regularization) —— #
        optimizer='AdamW',     # Can be switched to 'SGD' if needed (see notes)
        lr0=0.0012,            # ↑ increased from 8e-4 for faster early convergence
        weight_decay=0.005,    # ↓ reduced from 0.01 to avoid underfitting
        momentum=0.937,        # Used if optimizer is switched to SGD
        warmup_epochs=3,       # Slightly longer warmup for stability in early epochs
        cos_lr=True,
        patience=10,           # More aggressive early stopping to save time

        # —— Single-class task —— #
        single_cls=True,
        overlap_mask=True,

        # —— Validation / logging thresholds (restore precision) —— #
        iou=0.60,              # NMS IoU restored to 0.6
        conf=0.25,             # Confidence restored to 0.25 (0.10 previously hurt precision)

        # —— Data augmentation (moderate but not overly weak) —— #
        augment=True,
        mosaic=0.20,           # 0.10 → 0.20: improve generalization without tearing large TADs
        mixup=0.0,             # Keep disabled
        copy_paste=0.05,
        degrees=2.0,
        translate=0.05,
        scale=0.20,
        fliplr=0.0,            # Keep disabled: avoid diagonal symmetry artifacts
        label_smoothing=0.0,

        # —— Resources & logging —— #
        save_period=10,
        workers=8,
        verbose=True,
        seed=42,

        # —— Loss weights (back to more stable defaults) —— #
        box=7.5,
        cls=0.40,
        dfl=1.5,
    )


if __name__ == "__main__":
    DATA_YAML = "data.yaml"
    OUTPUT_DIR = ""

    train_from_yaml(
        data_yaml_path=DATA_YAML,
        project_dir=OUTPUT_DIR,
        model_cfg="/newbest/ultralytics/cfg/models/FACA/yolo2.yaml",
        epochs=60,
        imgsz=224,
        batch=32,
        device="0"
    )
