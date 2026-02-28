import yaml
import sys
import os
from pathlib import Path

# Safe to import ultralytics-related modules now
from PIL import Image
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pandas as pd

# ===================== hg38 chromosome lengths =====================
OM_LENGTHS = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr10": 133797422,
    "chr11": 135086622,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr19": 58617616,
    "chr20": 64444167,
    "chr21": 46709983,
    "chr22": 50818468,
    "chrX": 156040895,
    "chrY": 57227415,
}

def validate_omosome(om):
    """Normalize chromosome name and validate its existence"""
    if not om.startswith(""):
        om = f"{om}"
    if om not in OM_LENGTHS:
        raise ValueError(f"Invalid chromosome name: {om}")
    return om

def safe_predict(model, img_path, conf_threshold=0.5, iou_threshold=0.6):
    """
    Safe prediction wrapper with exception handling;
    explicitly sets IoU = 0.6 for NMS
    """
    try:
        results = model.predict(
            img_path,
            conf=conf_threshold,
            iou=iou_threshold,   # ★★★ Explicitly set NMS IoU threshold to 0.6
            imgsz=224,
            device='0',
            save=False,
            save_txt=False,
            save_conf=True
        )
        return results
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return None

def yolo_to_bed(yolo_output, om, window_start, window_size, bin_size=10000):
    """
    Convert YOLO predictions to BED coordinates,
    ensuring intervals stay within chromosome boundaries
    """
    bed_entries = []
    if yolo_output is None:
        return bed_entries

    om_length = OM_LENGTHS.get(om, 0)

    for result in yolo_output:
        boxes = result.boxes
        for box in boxes:
            try:
                # Safely retrieve predicted box data
                if len(box.xywhn) == 0:
                    continue

                # Normalized coordinates (center_x, center_y, width, height)
                cx, cy, w, h = box.xywhn[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()

                # Convert to genomic coordinates
                tad_center = window_start + int(cx * window_size * bin_size)
                tad_width = int(w * window_size * bin_size / 2)

                # Compute boundaries and clamp to chromosome range
                tad_start = max(0, tad_center - tad_width)
                tad_end = min(om_length, tad_center + tad_width)

                # Filter invalid intervals
                if tad_end - tad_start < 2 * bin_size:  # At least 2 bins
                    continue

                bed_entries.append(f"{om}\t{tad_start}\t{tad_end}\tTAD\t{conf:.3f}\n")

            except Exception as e:
                print(f"Error while processing prediction box: {str(e)}")
                continue

    return bed_entries

def predict_yolo_to_bed(model_path, h5_path, output_bed_path,
                        conf_threshold=0.5, iou_threshold=0.6):
    """
    Run inference using a trained YOLO model and export predictions in BED format
    """
    # Initialize model
    model = YOLO(model_path)
    WINDOW_SIZE = 224   # Image size
    BIN_SIZE = 10000    # Resolution

    # Create temporary directory
    temp_dir = os.path.join(os.path.dirname(output_bed_path), "temp_pred")
    os.makedirs(temp_dir, exist_ok=True)

    # Statistics
    total_images = 0
    valid_predictions = 0

    with open(output_bed_path, 'w') as bed_file, h5py.File(h5_path, 'r') as h5_file:
        # Write BED header
        bed_file.write("om\tstart\tend\tname\tscore\n")

        for _name in h5_file.keys():
            try:
                _name = validate_omosome(_name)
            except ValueError as e:
                print(str(e))
                continue

            if _name not in h5_file:
                print(f"Chromosome {_name} not found in HDF5 file")
                continue

            images = h5_file[_name]['images']
            positions = h5_file[_name]['positions']
            total_images += len(images)

            print(f"\nProcessing chromosome {_name} ({len(images)} images)...")

            for img_idx in range(len(images)):
                temp_img_path = None
                try:
                    # Load image and genomic position
                    image = np.array(images[img_idx])
                    pos_str = positions[img_idx].decode('utf-8')
                    om, pos_range = pos_str.split(':')
                    start_bp, end_bp = map(int, pos_range.split('-'))

                    # Save image temporarily for YOLO prediction
                    temp_img_path = os.path.join(temp_dir, f'temp_{_name}_{img_idx}.png')
                    plt.imsave(temp_img_path, image, cmap='gray')

                    # YOLO inference (IoU explicitly passed)
                    results = safe_predict(
                        model, temp_img_path, conf_threshold, iou_threshold
                    )
                    if results is None:
                        continue

                    # Convert predictions to BED
                    bed_entries = yolo_to_bed(
                        results, _name, start_bp, WINDOW_SIZE, BIN_SIZE
                    )
                    if not bed_entries:
                        continue

                    # Write results
                    for entry in bed_entries:
                        bed_file.write(entry)
                        valid_predictions += 1

                    # Progress logging
                    if (img_idx + 1) % 100 == 0 or (img_idx + 1) == len(images):
                        print(f"Progress: {img_idx + 1}/{len(images)}")

                except Exception as e:
                    print(f"Error processing {_name}_{img_idx}: {str(e)}")
                    continue
                finally:
                    if temp_img_path and os.path.exists(temp_img_path):
                        os.remove(temp_img_path)

    # Clean up temporary directory
    if os.path.exists(temp_dir):
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

    # Print summary
    print("\n" + "=" * 50)
    print("Prediction completed!")
    print(f"Total images processed: {total_images}")
    print(f"Valid predicted regions: {valid_predictions}")
    print(f"Results saved to: {output_bed_path}")
    print("=" * 50)

def filter_bed_file(input_bed, output_bed,
                    min_size=50000, max_size=2000000):
    """
    Filter BED file by removing intervals that are too small or too large
    """
    valid_count = 0
    with open(input_bed) as fin, open(output_bed, 'w') as fout:
        # Preserve header
        header = next(fin, None)
        if header and header.startswith("om"):
            fout.write(header)

        for line in fin:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            om = parts[0]
            try:
                start = int(parts[1])
                end = int(parts[2])
                size = end - start

                # Validate chromosome and interval size
                if (size >= min_size and
                    size <= max_size and
                    start >= 0 and
                    end <= OM_LENGTHS.get(om, 0)):
                    fout.write(line)
                    valid_count += 1
            except:
                continue

    print(f"\nFiltering result: {valid_count} regions passed size constraints")
    print(f"Filtered file saved to: {output_bed}")

if __name__ == "__main__":
    # Path configuration
    model_path = ''
    h5_path = ''
    output_dir = ''

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate raw predictions (IoU = 0.6)
    raw_bed = os.path.join(output_dir, 'raw_predictions.bed')
    print("Generating raw prediction file...")
    predict_yolo_to_bed(
        model_path=model_path,
        h5_path=h5_path,
        output_bed_path=raw_bed,
        conf_threshold=0.36,
        iou_threshold=0.4
    )

    # Filter predictions
    final_bed = os.path.join(output_dir, 'final_predictions.bed')
    print("\nFiltering prediction results...")
    filter_bed_file(raw_bed, final_bed)
