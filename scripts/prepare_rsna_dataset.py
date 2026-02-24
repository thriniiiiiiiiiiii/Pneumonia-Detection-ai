"""
RSNA Pneumonia Detection Dataset Preprocessor
----------------------------------------------
Converts RSNA DICOM files to JPEG and organizes them into the
folder structure expected by the training pipeline:

    data/raw/chest_xray/
        train/
            normal/      <- images with Target=0
            pneumonia/   <- images with Target=1

Then call create_balanced_dataset.py and train_model.py as normal.

Usage (from the project root directory):
    python scripts/prepare_rsna_dataset.py

Requirements:
    pip install pydicom pillow pandas numpy scikit-learn
"""

import os
import sys
import shutil
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import pydicom for reading DICOM files
# Try to import pydicom for reading DICOM files
try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
except ImportError:
    print("[ERROR] pydicom not installed. Please run: pip install pydicom")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("[ERROR] Pillow not installed. Please run: pip install pillow")
    sys.exit(1)

# --- Configuration ---

# Absolute path to the RSNA dataset folder
RSNA_DIR = Path(r"C:\Users\aroor\Downloads\PROJECTS-2026\chest-xray-pneumonia-detection-ai-main\rsna-pneumonia-detection-challenge")

# Where to write the organised JPEG images (this is what the existing scripts expect)
OUTPUT_DIR = Path(r"C:\Users\aroor\Downloads\PROJECTS-2026\chest-xray-pneumonia-detection-ai-main\data\raw\chest_xray")

# JPEG export quality (95 keeps excellent detail, lower = smaller files)
JPEG_QUALITY = 95

# Number of parallel workers for faster conversion
MAX_WORKERS: int = min(os.cpu_count() or 4, 8)

# ------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def clean_and_enhance_image(img_array: np.ndarray) -> np.ndarray:
    """Detect and remove text watermarks (inpainting) and apply CLAHE enhancement."""
    try:
        # 1. CLAHE Enhancement (Contrast Improvement)
        # Clinical studies show CLAHE significantly helps in detecting lung opacities
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_array)

        # 2. Watermark / Text Detection (Simple Thresholding)
        # Most CXR watermarks are high-intensity artifacts (white text)
        _, mask = cv2.threshold(img_enhanced, 250, 255, cv2.THRESH_BINARY)

        # Dilate mask slightly to cover the edges of anti-aliased text
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # 3. Inpainting
        # Replaces white pixels (text) with surrounding textures
        # This prevents the model from "cheating" based on markers
        img_cleaned = cv2.inpaint(img_enhanced, mask, 3, cv2.INPAINT_TELEA)

        return img_cleaned
    except Exception:
        # Fallback to enhanced image if inpainting fails
        return img_array

def dcm_to_jpeg(dcm_path: Path, jpeg_path: Path, quality: int = 95) -> bool:
    """Convert a single DICOM file to a JPEG image.

    Returns True on success, False on failure.
    """
    try:
        ds = pydicom.dcmread(str(dcm_path))

        # Get pixel array and apply VOI LUT (window/level) if present
        try:
            img_array = apply_voi_lut(ds.pixel_array, ds)
        except Exception:
            img_array = ds.pixel_array

        # Normalise to 0-255 uint8
        img_array = img_array.astype(np.float32)
        img_min, img_max = img_array.min(), img_array.max()
        if img_max > img_min:
            img_array = (img_array - img_min) / (img_max - img_min) * 255.0
        img_array = img_array.astype(np.uint8)

        # --- NEW: Advanced Cleaning & Enhancement ---
        img_array = clean_and_enhance_image(img_array)

        # Convert to PIL RGB (models expect 3-channel images)
        pil_img = Image.fromarray(img_array)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        pil_img.save(str(jpeg_path), "JPEG", quality=quality)
        return True

    except Exception as exc:
        log.warning(f"  [WARN] Failed to convert {dcm_path.name}: {exc}")
        return False


def main():
    print("=" * 60)
    print("RSNA -> JPEG Dataset Preprocessor")
    print("=" * 60)

    # --- 1. Validate input paths ---
    labels_csv = RSNA_DIR / "stage_2_train_labels.csv"
    images_dir = RSNA_DIR / "stage_2_train_images"

    for p, label in [(RSNA_DIR, "RSNA folder"), (labels_csv, "Labels CSV"),
                     (images_dir, "Train images folder")]:
        if not p.exists():
            log.error(f"[ERROR] {label} not found: {p}")
            sys.exit(1)

    log.info(f"[OK] RSNA folder found: {RSNA_DIR}")

    # --- 2. Read labels ---
    log.info("Reading labels CSV...")
    df = pd.read_csv(labels_csv)
    log.info(f"   Rows in CSV: {len(df)}")

    # The CSV may have duplicate patient IDs (one row per bounding box).
    # We only need one label per patient ID.
    if "patientId" not in df.columns or "Target" not in df.columns:
        log.error("[ERROR] Expected columns 'patientId' and 'Target' in CSV")
        log.error(f"   Found: {list(df.columns)}")
        sys.exit(1)

    # Deduplicate: if any box is annotated the patient is Pneumonia (Target=1)
    patient_labels = df.groupby("patientId")["Target"].max().reset_index()
    patient_labels.columns = ["patientId", "label"]

    n_total   = len(patient_labels)
    n_pneumo  = int((patient_labels["label"] == 1).sum())
    n_normal  = int((patient_labels["label"] == 0).sum())

    log.info(f"   Unique patients : {n_total}")
    log.info(f"   Pneumonia (1)   : {n_pneumo}")
    log.info(f"   Normal    (0)   : {n_normal}")

    # --- 3. Create output directory structure ---
    for cls in ("normal", "pneumonia"):
        (OUTPUT_DIR / "train" / cls).mkdir(parents=True, exist_ok=True)

    log.info(f"Output folder ready: {OUTPUT_DIR}")

    # --- 4. Convert and copy DCM -> JPEG ---
    log.info(f"Converting DICOM -> JPEG (workers={MAX_WORKERS})...")

    tasks = []
    for _, row in patient_labels.iterrows():
        pid   = row["patientId"]
        cls   = "pneumonia" if row["label"] == 1 else "normal"
        src   = images_dir / f"{pid}.dcm"
        dst   = OUTPUT_DIR / "train" / cls / f"{pid}.jpg"

        if not src.exists():
            log.warning(f"  [WARN] DCM not found (skipping): {src.name}")
            continue

        # Skip already-converted files so re-runs are fast
        if dst.exists():
            continue

        tasks.append((src, dst))

    log.info(f"   Files to convert: {len(tasks)}  (already done: {n_total - len(tasks)})")

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(dcm_to_jpeg, src, dst, JPEG_QUALITY): (src, dst)
                   for src, dst in tasks}
        for i, fu in enumerate(as_completed(futures), 1):
            if fu.result():
                ok += 1
            else:
                fail += 1
            if i % 500 == 0 or i == len(tasks):
                log.info(f"   Progress: {i}/{len(tasks)}  OK:{ok}  FAIL:{fail}")

    log.info(f"Conversion complete! Successful: {ok}  Failed: {fail}")

    # --- 5. Summary ---
    final_normal   = len(list((OUTPUT_DIR / "train" / "normal").glob("*.jpg")))
    final_pneumo   = len(list((OUTPUT_DIR / "train" / "pneumonia").glob("*.jpg")))

    print("\n" + "=" * 60)
    print("Final dataset summary")
    print("=" * 60)
    print(f"   Normal images    : {final_normal}")
    print(f"   Pneumonia images : {final_pneumo}")
    print(f"   Output location  : {OUTPUT_DIR}")
    print("=" * 60)
    print("\nDONE! Next steps:")
    print("   1. python scripts/create_balanced_dataset.py")
    print("   2. python scripts/train_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
