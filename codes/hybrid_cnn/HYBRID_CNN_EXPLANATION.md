# Hybrid CNN Project - Phase Explanation

## 📋 Overview: What is This Phase About?

This is the **advanced phase** of your TraceFinder project. Previously, you built:
1. **Baseline Models** (Random Forest, SVM) - using hand-crafted features
2. **Simple CNN Model** - using raw images directly

Now, the **Hybrid CNN** phase combines **both approaches** for better accuracy:
- Uses **residual/noise images** (not raw images)
- Extracts **scanner fingerprints** from special "Flatfield" images
- Combines **CNN features** + **hand-crafted features** in one model

---

## 🗂️ Dataset Structure

### 1. Old Datasets (Already Existed)
```
data/
├── Official/        → 1939 scanned document images (multiple scanners)
├── WikiPedia/       → 2368 scanned document images (multiple scanners)
```
These are **actual scanned documents** used for training and testing.

### 2. NEW: Flatfield Dataset (Just Added!)
```
data/Flatfield/     → 22 images total (11 scanner folders)
├── Canon120-1/     → 150.tif, 300.tif
├── Canon120-2/     → 150.tif, 300.tif
├── Canon220/       → 150.tif, 300.tif
├── Canon9000-1/    → 150.tif, 300.tif
├── Canon9000-2/    → 150.tif, 300.tif
├── EpsonV370-1/    → 150.tif, 300.tif
├── EpsonV370-2/    → 150.tif, 300.tif
├── EpsonV39-1/     → 150.tif, 300.tif
├── EpsonV39-2/     → 150.tif, 300.tif
├── EpsonV550/      → 150.tif, 300.tif
└── HP/             → 150.tif, 300.tif
```

### What are Flatfield Images?
**Flatfield images** are scans of a **blank, uniform surface** (like a white paper or nothing at all). 

**Why are they important?**
- When a scanner scans "nothing", the output should be perfectly uniform
- But it's NOT! Each scanner leaves its **unique noise pattern** (like a fingerprint)
- This noise pattern is called **PRNU (Photo-Response Non-Uniformity)**
- We use these flatfield images to **extract each scanner's unique fingerprint**

---

## 📁 Code Files Explained (In Simple English)

### 1. `utils.py` - The Toolbox 🔧
**What it does:** Contains helper functions used by other files.

| Function | What it does |
|----------|-------------|
| `corr2d()` | Measures similarity between two images (how alike they are) |
| `batch_corr_gpu()` | Same as above, but processes many images at once using GPU (faster) |
| `fft_radial_energy()` | Analyzes frequency patterns in an image (using Fourier Transform) |
| `lbp_hist_safe()` | Extracts texture patterns using LBP (Local Binary Pattern) |
| `extract_enhanced_features()` | Extracts all hand-crafted features: FFT + LBP + texture statistics |
| `process_batch_gpu()` | Processes a batch of images to create "residual" (noise) images |

---

### 2. `processing.py` - Image Preprocessing 🖼️
**What it does:** Converts raw scanned images into "residual" images (noise patterns).

**Process:**
```
Raw Image → Grayscale → Resize (256×256) → Normalize (0-1) → Denoise → Residual
```

- **Residual = Original - Denoised** (this extracts the noise pattern)
- Uses **Wavelet denoising** (Haar wavelet) to remove content, keeping only noise
- Saves processed data as `.pkl` (pickle) files for faster loading later

**Output Files Created:**
- `official_wiki_residuals.pkl` - Residuals from Official + Wikipedia images
- `flatfield_residuals.pkl` - Residuals from Flatfield images (for fingerprints)

---

### 3. `feature_extrac.py` - Feature Extraction 📊
**What it does:** Extracts features from the residual images.

**Two types of features:**
1. **Scanner Fingerprints** (from Flatfield)
   - Average all residuals from same scanner → creates that scanner's "fingerprint"
   - Saved as `scanner_fingerprints.pkl`

2. **PRNU Correlation Features** (for each training image)
   - Compare each image's residual with all scanner fingerprints
   - High correlation = image likely came from that scanner
   - Also extracts **enhanced features** (FFT, LBP, texture)

**Output Files:**
- `scanner_fingerprints.pkl` - 11 scanner fingerprints
- `fp_keys.npy` - Ordered list of scanner names
- `features.pkl` - PRNU correlation features
- `enhanced_features.pkl` - FFT + LBP + texture features

---

### 4. `model.py` - The Hybrid CNN Architecture 🧠
**What it does:** Defines the neural network structure.

```
                    Hybrid CNN
                        │
         ┌──────────────┴──────────────┐
         │                             │
  Image Branch (CNN)           Features Branch (Dense)
         │                             │
  Residual Image              Hand-crafted Features
     (256×256×1)                  (17 values)
         │                             │
  High-Pass Filter                Dense (64)
         │                             │
  Conv2D (32) → MaxPool           Dropout (0.2)
         │                             │
  Conv2D (64) → MaxPool               │
         │                             │
  Conv2D (128) → GAP                  │
         │                             │
         └──────────────┬──────────────┘
                        │
                   Concatenate
                        │
                   Dense (256)
                        │
                   Dropout (0.4)
                        │
                   Softmax Output
                   (11 classes)
```

**Key Features:**
- **Two inputs**: Image + Hand-crafted features
- **High-Pass Filter**: Fixed filter that enhances noise patterns
- **Fusion**: Combines CNN features with traditional features
- **Output**: Probability for each scanner class

---

### 5. `train_hybrid_cnn.py` - Training Script 🏋️
**What it does:** Trains the Hybrid CNN model.

**Process:**
1. Load residual images from `official_wiki_residuals.pkl`
2. Load/compute features (PRNU correlations + enhanced features)
3. Encode scanner labels (text → numbers)
4. Split data: 80% training, 20% testing
5. Normalize features using `StandardScaler`
6. Build and compile the Hybrid CNN model
7. Train for 50 epochs with:
   - Early Stopping (stops if no improvement for 10 epochs)
   - Learning Rate Reduction (reduces if stuck)
   - Model Checkpointing (saves best model)
8. Save model, encoders, scaler, and training plots

**Output Files:**
- `scanner_hybrid.keras` - Best model checkpoint
- `scanner_hybrid_final.keras` - Final trained model
- `hybrid_label_encoder.pkl` - Label encoder (scanner names ↔ numbers)
- `hybrid_feat_scaler.pkl` - Feature scaler
- `hybrid_training_history.pkl` - Training metrics
- `training_plot.png` - Accuracy and loss curves

---

### 6. `eval_hybrid_cnn.py` - Evaluation Script 📈
**What it does:** Tests the trained model and generates performance metrics.

**Process:**
1. Load trained model and preprocessors
2. Reconstruct test set (same split as training)
3. Run predictions on test images
4. Calculate metrics:
   - **Accuracy**: Overall correct predictions
   - **Classification Report**: Precision, Recall, F1 per scanner
   - **Confusion Matrix**: Shows which scanners are confused with each other

**Output:**
- Console output with accuracy and detailed report
- `confusion_matrix.png` - Visual confusion matrix

---

### 7. `test.py` - Inference/Prediction Script 🔍
**What it does:** Predicts scanner for new, unseen images.

**Usage:**
```python
# Predict a single image
result = predict_batch(["path/to/image.tif"])
# Returns: [(image_path, predicted_scanner, confidence%)]

# Predict entire folder
predict_folder("path/to/folder", output_csv="results.csv")
```

**Process:**
1. Process image → Residual
2. Extract features (PRNU + Enhanced)
3. Run through hybrid model
4. Return: Predicted scanner + Confidence percentage

---

## 🔄 Execution Order (How to Run This Phase)

```
Step 1: processing.py
    ↓   Creates: flatfield_residuals.pkl, official_wiki_residuals.pkl
Step 2: feature_extrac.py
    ↓   Creates: scanner_fingerprints.pkl, features.pkl, enhanced_features.pkl
Step 3: train_hybrid_cnn.py
    ↓   Creates: Trained Model & Artifacts
Step 4: eval_hybrid_cnn.py
    ↓   Creates: Confusion Matrix & Report
Step 5: test.py (Optional)
        For predicting new images
```

---

## ⚠️ Issues Found That Need Fixing

### 1. **Path Issues**
The code references `Data_Set` but your actual folder is `data`. This needs to be fixed.

### 2. **Flatfield Structure Mismatch**
The code expects flatfield images directly under scanner folders, but they're in DPI subfolders (150.tif, 300.tif).

### 3. **Missing Results Folder**
The `results/hybrid_cnn` folder doesn't exist and needs to be created.

---

## ✅ What We Need To Do (Step by Step)

1. **Fix Path Variables** - Update `Data_Set` → `../../data` in all scripts
2. **Fix Flatfield Processing** - Handle the DPI subfolder structure
3. **Create Output Directories** - Ensure `results/hybrid_cnn` exists
4. **Run Processing** - Execute `processing.py` first
5. **Run Feature Extraction** - Execute `feature_extrac.py`
6. **Train Model** - Execute `train_hybrid_cnn.py`
7. **Evaluate Model** - Execute `eval_hybrid_cnn.py`
8. **Test (Optional)** - Use `test.py` for new predictions
