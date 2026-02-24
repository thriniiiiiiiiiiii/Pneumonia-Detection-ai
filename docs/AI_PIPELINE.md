# AI/ML Pipeline — PneumoDetect AI

> **Document Status:** Production · **Version:** 1.0 · **Model:** MobileNetV2 (Cross-Operator Validated)

---

## 1. Pipeline Overview

The AI pipeline is a **Transfer Learning → Fine-tuning → Cross-Operator Validation** workflow designed to maximize diagnostic sensitivity while maintaining clinical generalizability.

```
Raw Datasets  ─► Preprocessing ─► Training ─► Internal Eval ─► Cross-Op Eval ─► Deployment
    │                                                                    │
    │ Kaggle CXR (train)                                       Kaggle Radiography (independent)
    └─────────────────────────────────────────────────────────────────────────────────────────
```

---

## 2. Dataset Analysis

### 2.1 Training Dataset

| Property | Value |
|---|---|
| **Source** | [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| **Origin** | Guangzhou Women and Children's Medical Center (Cell, 2018) |
| **Total Images** | 5,863 |
| **Pneumonia** | 4,273 (72.9%) |
| **Normal** | 1,583 (27.0%) |
| **Imbalance Ratio** | ≈3.3:1 (Pneumonia:Normal) |
| **License** | CC BY 4.0 |

### 2.2 Cross-Operator Validation Dataset

| Property | Value |
|---|---|
| **Source** | [Kaggle: Pneumonia Radiography Dataset](https://www.kaggle.com/datasets/iamtanmayshukla/pneumonia-radiography-dataset) |
| **Validation Subset Used** | 485 samples |
| **Curated By** | Dr. Pratibha (senior radiologist, independent audit, 2024) |
| **Key Difference** | Different technologists, export batch, and time period (2024 vs 2018) |
| **License** | CC0 (public domain) |

### 2.3 Why Two Datasets?

The critical design choice is using **temporally and operationally separated datasets**:

```
Dataset A (2018)               Dataset B (2024)
───────────────                ─────────────────
Same hospital                  Same hospital
Operator Team A                Operator Team B (cross-operator)
PACS export batch 1            PACS export batch 2
Radiology review team 1        Radiology review team 2 + Dr. Pratibha audit
→ Used for Training            → Used ONLY for validation (no leakage)
```

---

## 3. Data Preprocessing Pipeline

### 3.1 Scripts

| Script | Purpose |
|---|---|
| `scripts/analyze_and_balance.py` | EDA, class distribution, quality checks |
| `scripts/create_balanced_dataset.py` | Dataset balancing, augmentation, train/val/test split |
| `scripts/train_model.py` | Full training execution |
| `scripts/evaluate_model.py` | Internal evaluation (AUC, sensitivity, specificity, CM) |
| `scripts/cross-operator_validation.py` | Cross-op validation + bootstrap statistics |

### 3.2 Preprocessing Steps

```python
# Step 1: Load & decode
img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))

# Step 2: Normalize
arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0

# Step 3: Expand batch dimension
tensor = np.expand_dims(arr, axis=0)  # shape: (1, 224, 224, 3)
```

### 3.3 Class Balancing Strategy

Given the ~3.3:1 imbalance, the pipeline uses:
1. **Random oversampling** of the Normal class to achieve approximate 1:1 balance
2. **Online augmentation** during training (rotation ±15°, horizontal flip, brightness ±0.15, zoom ±0.1)
3. **Class weights** passed to `model.fit()` as a fallback

---

## 4. Model Architecture

### 4.1 Architecture Specification

```
Input Layer: (224, 224, 3)
    │
MobileNetV2 Backbone (154 layers)
├── Pre-trained: ImageNet weights (1.4M images, 1000 classes)
└── Fine-tuning: Top 20 layers unfrozen (phase 2 training)
    │
GlobalAveragePooling2D
├── Reduces spatial dimensions: (7, 7, 1280) → (1280,)
└── Prevents overfitting vs Flatten
    │
Dropout(rate=0.5)
├── Applied during training only
└── Regularizes the 1280-dim feature vector
    │
Dense(128, activation='relu')
├── Intermediate representation
└── ReLU for non-linearity
    │
Dense(1, activation='sigmoid')
└── Output: Pneumonia probability ∈ [0, 1]
```

### 4.2 Model Parameters

| Parameter | Value |
|---|---|
| **Total Parameters** | ~2.4M |
| **Trainable Parameters (Phase 2)** | ~540K |
| **Input Shape** | (224, 224, 3) |
| **Backbone** | MobileNetV2 |
| **Output Activation** | Sigmoid |
| **Decision Threshold** | 0.5 (adjustable) |
| **Saved Format** | `.h5` (Keras legacy) |

---

## 5. Training Configuration

### 5.1 Phase 1: Feature Extraction

```python
optimizer: Adam(learning_rate=1e-4)
loss:      binary_crossentropy
metrics:   [accuracy, AUC]
epochs:    20 (max) with EarlyStopping(patience=5)
batch_size: 32

# Backbone is frozen — only custom head trains
base_model.trainable = False
```

### 5.2 Phase 2: Fine-Tuning

```python
# Unfreeze top 20 layers of MobileNetV2
for layer in base_model.layers[-20:]:
    layer.trainable = True

optimizer: Adam(learning_rate=1e-5)  # Lower LR for fine-tuning
epochs: 10 (max) with EarlyStopping(patience=3)
```

### 5.3 Callbacks

| Callback | Configuration |
|---|---|
| `EarlyStopping` | `monitor='val_auc'`, `patience=5`, `restore_best_weights=True` |
| `ReduceLROnPlateau` | `monitor='val_loss'`, `factor=0.2`, `patience=3` |
| `ModelCheckpoint` | `save_best_only=True`, saved as `best_chest_xray_model.h5` |

---

## 6. Evaluation Framework

### 6.1 Internal Validation Results

| Metric | Value |
|---|---|
| **Accuracy** | 94.8% |
| **Sensitivity (Recall)** | 89.6% |
| **Specificity** | 100.0% |
| **ROC-AUC** | 98.8% |
| **F1-Score** | ~0.94 |

### 6.2 Cross-Operator Validation Results

Tested on 485 **truly independent** samples — different operators, different time period.

| Metric | Value |
|---|---|
| **Accuracy** | 86.0% |
| **Sensitivity (Recall)** | **96.4%** |
| **Specificity** | 74.8% |
| **ROC-AUC** | **96.4%** |
| **True Positives** | 175 |
| **False Positives** | 59 |
| **False Negatives** | 9 |
| **True Negatives** | — |

### 6.3 Statistical Verification

**Primary Test: Bootstrap AUC Comparison**

```python
# From scripts/cross-operator_validation.py
n_bootstraps = 1000
rng = np.random.RandomState(42)

for i in range(n_bootstraps):
    indices = rng.randint(0, len(y_true), len(y_true))
    auc = roc_auc_score(y_true[indices], y_scores[indices])
    bootstrap_aucs.append(auc)

# Result: mean ΔAUC = -0.0001, 95% CI = [-0.0115, 0.0099], p = 0.978
```

**Interpretation:** The 95% CI includes zero, meaning there is **no statistically significant difference** between internal and cross-operator AUC. The model generalizes robustly.

---

## 7. Explainability — GradCAM

The web application generates **GradCAM heatmaps** highlighting the image regions most influential to the model's decision.

```python
# GradCAM implementation summary
last_conv_layer = model.get_layer("Conv_1_bn")  # Last MobileNetV2 conv
grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[last_conv_layer.output, model.output]
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, 0]

grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
heatmap = tf.reduce_mean(conv_outputs[0] * pooled_grads, axis=-1)
heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
```

---

## 8. Model Distribution

| Platform | Purpose | URL |
|---|---|---|
| **HuggingFace Hub** | Production model registry | `ayushirathour/chest-xray-pneumonia-detection` |
| **Zenodo Archive** | Immutable research archive | DOI: 10.5281/zenodo.17520564 |

---

## 9. Reproducibility Checklist

All files required to reproduce the published results:

```
results/reproducibility/
├── bootstrap_auc_results.json    # Bootstrap statistical test
├── internal_metrics.json         # Internal validation metrics
├── crossop_metrics.json          # Cross-operator metrics
├── internal_confusion_matrix.csv
├── crossop_confusion_matrix.csv
└── model_parameters.json         # Full architecture + hyperparameters
```

**Zenodo DOI:** [10.5281/zenodo.17520564](https://zenodo.org/records/17520564)
