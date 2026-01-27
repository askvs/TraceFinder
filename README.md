<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/PyTorch-2.x-red?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-infosys-green?style=for-the-badge" alt="License">
</p>

<h1 align="center">
  🔍 TraceFinder
</h1>

<p align="center">
  <strong>AI-Powered Forensic Scanner Identification System</strong>
</p>

<p align="center">
  <em>Identify the source scanner device from any document using deep learning & signal processing</em>
</p>

<p align="center">
  <a href="#-key-features">Features</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-demo">Demo</a> •
  <a href="#-models">Models</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a>
</p>

---

## 🎯 What is TraceFinder?

**TraceFinder** is a cutting-edge forensic tool that answers one critical question:

> *"Which scanner was used to create this document?"*

Every scanner leaves behind a unique **digital fingerprint** — subtle noise patterns, sensor artifacts, and frequency signatures that are invisible to the naked eye but detectable by AI. TraceFinder exploits these microscopic traces to identify the exact make and model of the source scanner.

<table>
<tr>
<td width="50%">

### 🕵️ Digital Forensics
Determine which scanner was used to forge, duplicate, or originate legal documents.

</td>
<td width="50%">

### ⚖️ Legal Verification
Verify that court-submitted scanned documents came from authorized, known devices.

</td>
</tr>
</table>

---

## ✨ Key Features

<table>
<tr>
<td align="center" width="25%">

### 🧠
**Hybrid CNN**
<br>
<sub>Deep residual learning + handcrafted features</sub>

</td>
<td align="center" width="25%">

### 📉
**PRNU Analysis**
<br>
<sub>Photo-Response Non-Uniformity fingerprinting</sub>

</td>
<td align="center" width="25%">

### 🌊
**FFT + Wavelet**
<br>
<sub>Frequency domain forensics</sub>

</td>
<td align="center" width="25%">

### 🔍
**Grad-CAM**
<br>
<sub>Explainable AI visualization</sub>

</td>
</tr>
</table>

---

## 🚀 How It Works

<p align="center">
  <img src="https://img.shields.io/badge/1-INPUT-2196F3?style=for-the-badge&logo=image&logoColor=white" alt="Step 1">
  <img src="https://img.shields.io/badge/→-gray?style=for-the-badge" alt="arrow">
  <img src="https://img.shields.io/badge/2-PREPROCESS-9C27B0?style=for-the-badge&logo=cog&logoColor=white" alt="Step 2">
  <img src="https://img.shields.io/badge/→-gray?style=for-the-badge" alt="arrow">
  <img src="https://img.shields.io/badge/3-EXTRACT-FF9800?style=for-the-badge&logo=chart-bar&logoColor=white" alt="Step 3">
  <img src="https://img.shields.io/badge/→-gray?style=for-the-badge" alt="arrow">
  <img src="https://img.shields.io/badge/4-CLASSIFY-E91E63?style=for-the-badge&logo=brain&logoColor=white" alt="Step 4">
  <img src="https://img.shields.io/badge/→-gray?style=for-the-badge" alt="arrow">
  <img src="https://img.shields.io/badge/5-OUTPUT-4CAF50?style=for-the-badge&logo=check-circle&logoColor=white" alt="Step 5">
</p>

<table>
<tr>
<td align="center" width="20%">
<img src="https://img.shields.io/badge/-2196F3?style=flat-square" width="100%" height="5">
<br><br>
<b>📄 INPUT</b>
<br><br>
Upload scanned document
<br>
<code>JPG</code> <code>PNG</code> <code>TIFF</code>
</td>
<td align="center" width="20%">
<img src="https://img.shields.io/badge/-9C27B0?style=flat-square" width="100%" height="5">
<br><br>
<b>⚙️ PREPROCESS</b>
<br><br>
Grayscale → Resize → Wavelet Denoise
</td>
<td align="center" width="20%">
<img src="https://img.shields.io/badge/-FF9800?style=flat-square" width="100%" height="5">
<br><br>
<b>📊 EXTRACT</b>
<br><br>
Noise residual + 44 forensic features
</td>
<td align="center" width="20%">
<img src="https://img.shields.io/badge/-E91E63?style=flat-square" width="100%" height="5">
<br><br>
<b>🧠 CLASSIFY</b>
<br><br>
Hybrid CNN / SVM / Random Forest
</td>
<td align="center" width="20%">
<img src="https://img.shields.io/badge/-4CAF50?style=flat-square" width="100%" height="5">
<br><br>
<b>✅ OUTPUT</b>
<br><br>
Scanner ID + Confidence %
</td>
</tr>
</table>

### 🔬 44 Forensic Features Extracted

<table>
<tr>
<td align="center" width="25%">

**🌊 Wavelet Residual**

Haar L1 decomposition extracts scanner-specific noise

</td>
<td align="center" width="25%">

**📈 FFT Spectrum**

3 frequency bands (Low/Mid/High energy)

</td>
<td align="center" width="25%">

**🔲 LBP Texture**

26-bin histogram from 24-neighbor pattern

</td>
<td align="center" width="25%">

**📍 PRNU Correlation**

11 scanner fingerprint similarity scores

</td>
</tr>
</table>

---

## 📊 Model Performance

### 🏆 Hybrid CNN Results

<table>
<tr>
<td width="50%">

**Overall Accuracy: 82.67%**

| Scanner | Precision | Recall |
|---------|-----------|--------|
| Canon120-1 | 0.80 | 0.81 |
| Canon120-2 | 0.66 | 0.74 |
| Canon220 | 0.73 | 0.77 |
| Canon9000-1 | 0.69 | 0.79 |
| Canon9000-2 | 0.61 | 0.81 |
| EpsonV370-1 | 0.87 | 0.95 |
| EpsonV370-2 | 0.87 | 0.91 |
| EpsonV39-1 | 0.61 | 0.74 |
| EpsonV39-2 | 0.75 | 0.70 |
| EpsonV550 | 1.00 | 1.00 |
| HP | 1.00 | 1.00 |

</td>
<td width="50%">

<img src="results/hybrid_cnn/confusion_matrix.png" alt="Confusion Matrix" width="100%">

</td>
</tr>
</table>

### 📈 Training Progress

<p align="center">
<img src="results/hybrid_cnn/training_plot.png" alt="Training Plot" width="80%">
</p>

---

## 🛠️ Models Included

| Model | Architecture | Use Case | Framework |
|-------|--------------|----------|-----------|
| **Hybrid CNN** | ResNet-style + Feature Fusion | High accuracy | TensorFlow/Keras |
| **Standalone CNN** | VGG-style ConvNet | Fast inference | PyTorch |
| **Random Forest** | Ensemble (100 trees) | Baseline | Scikit-learn |
| **SVM** | RBF Kernel | Baseline | Scikit-learn |

### Feature Extraction Pipeline

```
44 Handcrafted Features
├── PRNU Correlations (11)
│   └── Cross-correlation with scanner fingerprints
├── FFT Features (3)
│   └── Low/Mid/High frequency energy
├── LBP Histogram (26)
│   └── Local Binary Pattern texture
└── Gradient Features (4)
    └── Sobel magnitude statistics
```

---

## 🖥️ Live Demo

Try the interactive web application:

```bash
streamlit run landing_page.py
```

### Screenshots

The app includes:
- 🎨 **Modern dark theme** with gradient accents
- 📤 **Drag & drop** image upload
- 🧪 **Sample images** for quick testing
- 📊 **Real-time** probability charts
- 🔍 **Grad-CAM** explainability

---

## 📦 Installation

### Prerequisites

- Python 3.9+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/askvs/TraceFinder.git
cd TraceFinder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Web Application

```bash
# Activate virtual environment first
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Launch the app
streamlit run landing_page.py
```

Then open your browser to `http://localhost:8501`

### Available Models in the App

| Model | Best For |
|-------|----------|
| 🧠 **Hybrid CNN** | Highest accuracy (recommended) |
| 🔬 **Standalone CNN** | Fast deep learning inference |
| 🌲 **Random Forest** | Quick baseline analysis |
| ⚡ **SVM** | Lightweight classification |
| 🔍 **Grad-CAM** | Visual explainability |

---

## 📁 Project Structure

```
TraceFinder/
│
├── 📄 landing_page.py          # Main Streamlit web application
├── 📄 model_inference.py       # Unified inference API for all models
├── 📄 requirements.txt         # Python dependencies
│
├── 📁 codes/
│   ├── 📁 baseline/
│   │   ├── train_baseline.py       # Train SVM & Random Forest
│   │   ├── evaluate_baseline.py    # Evaluate baseline models
│   │   ├── predict_baseline.py     # Prediction utilities
│   │   ├── models.py               # Model definitions
│   │   └── tuning.py               # Hyperparameter tuning
│   │
│   ├── 📁 cnn_model/
│   │   ├── model.py                # PyTorch CNN architecture
│   │   ├── dataset.py              # Dataset loader
│   │   ├── train.py                # Training script
│   │   └── evaluate.py             # Evaluation metrics
│   │
│   ├── 📁 hybrid_cnn/
│   │   ├── train_hybrid_cnn.py     # Train Hybrid CNN model
│   │   ├── eval_hybrid_cnn.py      # Evaluation with metrics
│   │   ├── gradcam.py              # Grad-CAM visualization
│   │   └── processing.py           # Feature preprocessing
│   │
│   ├── 📁 eda/
│   │   ├── eda_official.py         # EDA on official dataset
│   │   └── eda_wikipedia.py        # EDA on Wikipedia dataset
│   │
│   └── preprocess_combined.py      # Combined preprocessing pipeline
│
├── 📁 models/
│   ├── 📁 baseline/
│   │   ├── random_forest.joblib    # Trained Random Forest (~2 MB)
│   │   ├── svm.joblib              # Trained SVM model
│   │   └── scaler.joblib           # Feature scaler
│   │
│   ├── 📁 cnn/
│   │   └── cnn_model.pth           # PyTorch CNN weights (~64 MB)
│   │
│   └── 📁 hybrid_cnn/
│       ├── scanner_hybrid.keras    # Keras Hybrid CNN model
│       ├── hybrid_label_encoder.pkl # Label encoder
│       └── hybrid_feat_scaler.pkl  # Feature scaler
│
├── 📁 results/
│   ├── 📁 baseline/
│   │   ├── Random_Forest_confusion_matrix.png
│   │   └── SVM_confusion_matrix.png
│   │
│   ├── 📁 cnn/
│   │   ├── confusion_matrix.png
│   │   └── training_curves.png
│   │
│   ├── 📁 hybrid_cnn/
│   │   ├── confusion_matrix.png
│   │   ├── training_plot.png
│   │   └── 📁 gradcam/             # Grad-CAM heatmaps
│   │
│   └── 📁 eda/
│       ├── 📁 official/            # Official dataset analysis
│       └── 📁 wikipedia/           # Wikipedia dataset analysis
│
└── 📁 samples/                     # Test images for each scanner
    ├── Canon120-1_sample.jpg
    ├── Canon9000-1_sample.jpg
    ├── Canon9000-2_sample.jpg
    ├── EpsonV370-1_sample.jpg
    ├── EpsonV39-1_sample.jpg
    ├── EpsonV550_sample.jpg
    └── HP_sample.jpg
```

---

## 🔬 Technical Details

### Preprocessing Pipeline

1. **Grayscale conversion** — Remove color information
2. **Resize to 256×256** — Standardize dimensions
3. **Wavelet denoising** — Haar wavelet approximation
4. **Residual extraction** — `Original - Denoised = Noise Pattern`

### Supported Scanners

| # | Scanner Model |
|---|---------------|
| 1 | Canon120-1 |
| 2 | Canon120-2 |
| 3 | Canon220 |
| 4 | Canon9000-1 |
| 5 | Canon9000-2 |
| 6 | EpsonV370-1 |
| 7 | EpsonV370-2 |
| 8 | EpsonV39-1 |
| 9 | EpsonV39-2 |
| 10 | EpsonV550 |
| 11 | HP |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Research inspired by digital forensics and source device identification techniques
- PRNU-based camera identification literature
- TensorFlow and PyTorch communities

---

<p align="center">
  <strong>TraceFinder</strong> — Where every scan leaves a trace 🔍
</p>

<p align="center">
  Made with ❤️ for Digital Forensics
</p>
