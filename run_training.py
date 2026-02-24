"""
run_training.py  -  HIGH ACCURACY + HIGH SPEED Training Pipeline
=================================================================
Optimised for the best possible detection results on RSNA data.

Accuracy Features:
  [OK] EfficientNetV2-B0 backbone (Smarter than MobileNetV2)
  [OK] 224x224 images (Industry standard resolution)
  [OK] 6,000 images/class (Total 12,000 samples for better signal)
  [OK] Dense head with 256 units + BatchNormalization
  [OK] Unfreeze top 50 layers (Deep fine-tuning for radiology)
  [OK] Mixed precision + CosineDecay for efficient compute
  [OK] XLA JIT compilation for GPU acceleration
  [OK] Multiprocessing data loading (cpu_count workers)
  [OK] Full metrics: Accuracy, Precision, Recall, AUC, F1, Confusion Matrix

Run from the project root:
    python run_training.py
"""

import os, sys, logging, random, shutil, subprocess, json
from pathlib import Path

# --- Speed: Enable XLA JIT before any TF import ---
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- Paths ---

ROOT          = Path(__file__).resolve().parent
SCRIPTS_DIR   = ROOT / "scripts"
RAW_DATA_DIR  = ROOT / "data" / "raw" / "chest_xray"
PROC_DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- Accuracy Configuration ---

IMG_SIZE        = 224    # Standard for EfficientNet
MAX_PER_CLASS   = 6000   # Good signal for accuracy (total 12k images)
BATCH_SIZE      = 32     # Lower batch for better gradient signal
EPOCHS          = 20     # More epochs for deep learning
UNFREEZE_LAYERS = 50     # Unfreeze deeper for radiology specific details
LEARNING_RATE   = 4e-4   # Slightly lower for more stability

# Speed: number of data-loading workers (use all CPU cores, cap at 8)
NUM_WORKERS     = min(os.cpu_count() or 4, 8)

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def step_banner(n, title):
    print(f"\n{'='*60}\n  STEP {n}: {title}\n{'='*60}")


# --- Step 1: RSNA DICOM -> JPEG ---

def run_rsna_preprocessing():
    step_banner(1, "RSNA DICOM -> JPEG conversion")

    n_normal  = len(list((RAW_DATA_DIR / "train" / "normal").glob("*.jpg")))   if (RAW_DATA_DIR / "train" / "normal").exists()   else 0
    n_pneumo  = len(list((RAW_DATA_DIR / "train" / "pneumonia").glob("*.jpg"))) if (RAW_DATA_DIR / "train" / "pneumonia").exists() else 0

    if n_normal > 100 and n_pneumo > 100:
        log.info(f"[OK] Raw images already present (normal={n_normal}, pneumonia={n_pneumo}) - skipping.")
        return True

    script = SCRIPTS_DIR / "prepare_rsna_dataset.py"
    ret = subprocess.run([sys.executable, str(script)]).returncode
    return ret == 0


# ─── Step 2: Fast balanced split with per-class cap ──────────────────────────

def build_fast_split():
    """
    Build a capped, balanced processed split directly — skips create_balanced_dataset.py
    because we also enforce the MAX_PER_CLASS limit here for speed.
    """
    step_banner(2, f"Fast balanced split (cap={MAX_PER_CLASS}/class)")

    proc_train_n = PROC_DATA_DIR / "train" / "normal"
    proc_train_p = PROC_DATA_DIR / "train" / "pneumonia"

    if (proc_train_n.exists() and len(list(proc_train_n.glob("*"))) > 10 and
            proc_train_p.exists() and len(list(proc_train_p.glob("*"))) > 10):
        log.info("✅ Processed split already exists — skipping.")
        return True

    # Wipe and recreate
    if PROC_DATA_DIR.exists():
        shutil.rmtree(PROC_DATA_DIR)

    for split in ("train", "val", "test"):
        for cls in ("normal", "pneumonia"):
            (PROC_DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    raw_normal   = sorted((RAW_DATA_DIR / "train" / "normal").glob("*.jpg"))
    raw_pneumo   = sorted((RAW_DATA_DIR / "train" / "pneumonia").glob("*.jpg"))

    if not raw_normal or not raw_pneumo:
        log.error("❌ No raw JPEG images found! Run Step 1 first.")
        return False

    random.seed(42)
    random.shuffle(raw_normal)
    random.shuffle(raw_pneumo)

    # Cap and match sizes
    cap = min(MAX_PER_CLASS, len(raw_normal), len(raw_pneumo))
    raw_normal  = raw_normal[:cap]
    raw_pneumo  = raw_pneumo[:cap]

    log.info(f"   Using {cap} images per class  ({cap*2} total)")

    def split_list(lst, train=0.70, val=0.20):
        n = len(lst)
        t = int(n * train)
        v = int(n * val)
        return lst[:t], lst[t:t+v], lst[t+v:]

    def copy_files(files, label, split_name):
        dest = PROC_DATA_DIR / split_name / label
        for f in files:
            shutil.copy2(f, dest / f.name)

    for imgs, label in [(raw_normal, "normal"), (raw_pneumo, "pneumonia")]:
        tr, va, te = split_list(imgs)
        copy_files(tr, label, "train")
        copy_files(va, label, "val")
        copy_files(te, label, "test")
        log.info(f"   {label}: train={len(tr)} val={len(va)} test={len(te)}")

    log.info("[OK] Fast balanced split created.")
    return True


# ─── Step 3: Train (fast + accurate) ─────────────────────────────────────────

def run_training():
    step_banner(3, "Training EfficientNetV2 (High Accuracy + Speed)")

    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetV2B0
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
    from tensorflow.keras.models import Model
    from tensorflow.keras import Input
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.optimizers.schedules import CosineDecay
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import matplotlib.pyplot as plt
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 roc_auc_score, f1_score, precision_score, recall_score)
    import seaborn as sns

    # -- GPU detection + Mixed precision --
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            # Enable XLA JIT for GPU
            tf.config.optimizer.set_jit(True)
            log.info(f"[SPEED] Mixed precision + XLA ENABLED (GPU: {len(gpus)} device(s))")
        else:
            log.info("[CPU] No GPU detected — running on CPU (mixed precision skipped)")
    except Exception:
        pass

    # ── Data generators ───────────────────────────────────────────────────────
    train_datagen = ImageDataGenerator(
        rescale            = 1./255,
        rotation_range     = 10,
        width_shift_range  = 0.1,
        height_shift_range = 0.1,
        zoom_range         = 0.1,
        horizontal_flip    = True,
        fill_mode          = "nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        PROC_DATA_DIR / "train",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="binary", shuffle=True, seed=42,
    )
    val_gen = val_datagen.flow_from_directory(
        PROC_DATA_DIR / "val",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="binary", shuffle=False,
    )
    test_gen = val_datagen.flow_from_directory(
        PROC_DATA_DIR / "test",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="binary", shuffle=False,
    )

    log.info(f"   Train: {train_gen.samples}  |  Val: {val_gen.samples}  |  Test: {test_gen.samples}")
    log.info(f"   Class map: {train_gen.class_indices}")
    log.info(f"   Data workers: {NUM_WORKERS}")

    # ── Build model with partial fine-tuning ──────────────────────────────────
    inp   = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base  = EfficientNetV2B0(weights="imagenet", include_top=False, input_tensor=inp)

    # Freeze all except top layers
    for layer in base.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False
    for layer in base.layers[-UNFREEZE_LAYERS:]:
        layer.trainable = True

    trainable_count = sum(1 for l in base.layers if l.trainable)
    log.info(f"   EfficientNetV2B0 layers trainable: {trainable_count}/{len(base.layers)}")

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation="sigmoid", dtype="float32", name="predictions")(x)

    model = Model(inp, out)

    # ── Cosine Decay LR schedule ──────────────────────────────────────────────
    steps_per_epoch = train_gen.samples // BATCH_SIZE
    total_steps     = steps_per_epoch * EPOCHS
    lr_schedule     = CosineDecay(
        initial_learning_rate = LEARNING_RATE,
        decay_steps           = total_steps,
        alpha                 = 1e-6,
    )

    model.compile(
        optimizer = Adam(learning_rate=lr_schedule),
        loss      = "binary_crossentropy",
        metrics   = [
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    log.info(f"   Total parameters: {model.count_params():,}")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    best_model_path = str(MODELS_DIR / "best_chest_xray_model.h5")

    callbacks = [
        ModelCheckpoint(
            best_model_path,
            monitor="val_accuracy", save_best_only=True, mode="max", verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-8, verbose=1,
        ),
    ]

    # ── Train with multiprocessing data loading ───────────────────────────────
    log.info(f"\n🔥 Training started  (epochs={EPOCHS}  batch={BATCH_SIZE}  img={IMG_SIZE}²  workers={NUM_WORKERS})")
    history = model.fit(
        train_gen,
        epochs              = EPOCHS,
        validation_data     = val_gen,
        callbacks           = callbacks,
        use_multiprocessing = False,   # True on Linux/Mac; False on Windows to avoid pickling issues
        workers             = NUM_WORKERS,
        verbose             = 1,
    )

    # ── Training curves ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Training Results", fontsize=14, fontweight="bold")

    for ax, metric, title in [
        (axes[0], "accuracy",  "Accuracy"),
        (axes[1], "loss",      "Loss"),
        (axes[2], "precision", "Precision & Recall"),
        (axes[3], "auc",       "AUC-ROC"),
    ]:
        ax.plot(history.history[metric],          label="Train")
        ax.plot(history.history[f"val_{metric}"], label="Val")
        if metric == "precision":
            ax.plot(history.history.get("recall", []),     "--", label="Train Recall")
            ax.plot(history.history.get("val_recall", []), "--", label="Val Recall")
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(MODELS_DIR / "training_progress.png"), dpi=150)
    log.info("📊 Training curves saved → models/training_progress.png")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    step_banner(4, "Evaluation on held-out test set")

    model.load_weights(best_model_path)
    test_gen.reset()
    preds  = model.predict(test_gen, verbose=1)
    y_pred = (preds > 0.5).astype(int).flatten()
    y_true = test_gen.classes

    class_names = ["Normal", "Pneumonia"]
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n📋 Classification Report:\n" + "="*50)
    print(report)

    cm          = confusion_matrix(y_true, y_pred)
    accuracy    = (cm[0,0] + cm[1,1]) / cm.sum()
    sensitivity = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0]+cm[1,1]) > 0 else 0
    specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0

    # ── New: AUC-ROC, F1, Precision, Recall ──────────────────────────────────
    auc_roc   = roc_auc_score(y_true, preds.flatten())
    f1        = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)

    print(f"\n📈 Extended Metrics:")
    print(f"   AUC-ROC     : {auc_roc:.4f}")
    print(f"   F1-Score    : {f1:.4f}")
    print(f"   Precision   : {precision:.4f}")
    print(f"   Recall      : {recall:.4f}")
    print(f"   Sensitivity : {sensitivity:.4f}  (Pneumonia recall)")
    print(f"   Specificity : {specificity:.4f}  (Normal recall)")

    # Confusion matrix plot
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Test Confusion Matrix\nAccuracy={accuracy:.1%}  Sensitivity={sensitivity:.1%}  Specificity={specificity:.1%}")
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(str(MODELS_DIR / "confusion_matrix.png"), dpi=150)
    log.info("📊 Confusion matrix saved → models/confusion_matrix.png")

    # ── Save metrics JSON for the Streamlit app ───────────────────────────────
    metrics = {
        "accuracy":    round(float(accuracy),    4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "auc_roc":     round(float(auc_roc),     4),
        "f1_score":    round(float(f1),          4),
        "precision":   round(float(precision),   4),
        "recall":      round(float(recall),      4),
        "confusion_matrix": cm.tolist(),
        "validation_samples": int(test_gen.samples),
    }
    metrics_path = MODELS_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    log.info(f"💾 Evaluation metrics saved → models/evaluation_metrics.json")

    # ── Save final model ──────────────────────────────────────────────────────
    model.save(str(MODELS_DIR / "final_chest_xray_model.h5"))

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"   Accuracy    : {accuracy:.1%}")
    print(f"   AUC-ROC     : {auc_roc:.4f}")
    print(f"   F1-Score    : {f1:.4f}")
    print(f"   Sensitivity : {sensitivity:.1%}  (% pneumonia cases caught)")
    print(f"   Specificity : {specificity:.1%}  (% normal cases correctly cleared)")
    print(f"   Best model  : models/best_chest_xray_model.h5")
    print(f"   Final model : models/final_chest_xray_model.h5")
    print(f"   Metrics JSON: models/evaluation_metrics.json")
    print(f"{'='*60}\n")
    return True


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nChest X-Ray Pneumonia Detection -- High Accuracy + Speed Pipeline")
    print(f"   Config: img={IMG_SIZE}  cap={MAX_PER_CLASS}/class  epochs={EPOCHS}  batch={BATCH_SIZE}  workers={NUM_WORKERS}")
    print("="*60)

    if not run_rsna_preprocessing():
        log.error("[ERROR] Preprocessing failed. Aborting.")
        sys.exit(1)

    if not build_fast_split():
        log.error("[ERROR] Dataset split failed. Aborting.")
        sys.exit(1)

    run_training()
