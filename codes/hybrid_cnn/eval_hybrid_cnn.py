import os, pickle, numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from utils import corr2d, extract_enhanced_features
from datetime import datetime

# ---- Paths ---- Fixed for correct project structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

BASE_DIR = os.path.join(PROJECT_ROOT, "data")

# Model files from models/hybrid_cnn
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "hybrid_cnn")
# Processed data from processed_data/hybrid_cnn
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_data", "hybrid_cnn")
# Results go to results/hybrid_cnn
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "hybrid_cnn")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Try original model first (82.67% accuracy)
MODEL_PATH = os.path.join(MODEL_DIR, "scanner_hybrid.keras")

ENCODER_PATH = os.path.join(MODEL_DIR, "hybrid_label_encoder.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "hybrid_feat_scaler.pkl")
RES_PATH = os.path.join(PROCESSED_DIR, "official_wiki_residuals.pkl")
FP_PATH  = os.path.join(PROCESSED_DIR, "scanner_fingerprints.pkl")
ORDER_NPY = os.path.join(PROCESSED_DIR, "fp_keys.npy")
FEATURES_PATH = os.path.join(PROCESSED_DIR, "features.pkl")
ENHANCED_PATH = os.path.join(PROCESSED_DIR, "enhanced_features.pkl")

# ---- Reproducibility ----
SEED = 42
np.random.seed(SEED)

# ---- Load artifacts ----
print("="*60)
print("HYBRID CNN EVALUATION")
print("="*60)
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ---- Load Data ----
print("Loading residuals...")
with open(RES_PATH, "rb") as f:
    residuals_dict = pickle.load(f)

# Optional: Load pre-computed features
precomputed = False
if os.path.exists(FEATURES_PATH) and os.path.exists(ENHANCED_PATH):
    print("Loading pre-computed features...")
    with open(FEATURES_PATH, "rb") as f:
        d_feat = pickle.load(f)
        feats_prnu = d_feat["features"]
    with open(ENHANCED_PATH, "rb") as f:
        d_enh = pickle.load(f)
        feats_enh = d_enh["features"]
    precomputed = True
else:
    print("Pre-computed features not found. Computing on the fly...")
    if os.path.exists(FP_PATH):
        with open(FP_PATH, "rb") as f:
            scanner_fps = pickle.load(f)
        fp_keys = np.load(ORDER_NPY, allow_pickle=True).tolist()
    else:
        raise FileNotFoundError("Scanner fingerprints not found.")

# ---- Reconstruct Full Dataset ----
X_img_all, X_feat_all, y_all = [], [], []
idx_counter = 0

for dataset_name in ["Official", "WikiPedia"]:
    if dataset_name not in residuals_dict:
        continue
    
    print(f"Processing {dataset_name}...")
    for scanner, dpi_dict in residuals_dict[dataset_name].items():
        if isinstance(dpi_dict, dict):
            for dpi, res_list in dpi_dict.items():
                for res in res_list:
                    X_img_all.append(np.expand_dims(res, -1))
                    
                    if precomputed:
                        f_p = feats_prnu[idx_counter]
                        f_e = feats_enh[idx_counter]
                        X_feat_all.append(f_p + f_e)
                    else:
                        v_corr = [corr2d(res, scanner_fps[k]) for k in fp_keys]
                        v_enh = extract_enhanced_features(res)
                        X_feat_all.append(v_corr + v_enh)
                        
                    y_all.append(scanner)
                    idx_counter += 1

X_img_all = np.array(X_img_all, dtype=np.float32)
X_feat_all = np.array(X_feat_all, dtype=np.float32)
y_all = np.array(y_all)

# ---- Split to Recover Test Set ----
y_int_all = le.transform(y_all)
num_classes = len(le.classes_)
y_cat_all = to_categorical(y_int_all, num_classes)

print("Splitting data to recover Test Set...")
_, X_img_te, _, X_feat_te, _, y_te = train_test_split(
    X_img_all, X_feat_all, y_cat_all, test_size=0.2, random_state=SEED, stratify=y_int_all
)

y_int_te = np.argmax(y_te, axis=1)

# Scale Features
X_feat_te = scaler.transform(X_feat_te)

print(f"Test Set: {len(X_img_te)} samples")

# ---- Evaluate ----
print("\nRunning prediction...")
y_pred_prob = model.predict([X_img_te, X_feat_te])
y_pred = np.argmax(y_pred_prob, axis=1)

test_acc = accuracy_score(y_int_te, y_pred)
print("\n" + "="*60)
print(f"TEST ACCURACY: {test_acc*100:.2f}%")
print("="*60)

# Classification Report
report_str = classification_report(y_int_te, y_pred, target_names=le.classes_)
print("\nClassification Report:")
print(report_str)

# Per-class Analysis
print("\n" + "="*60)
print("PER-CLASS ANALYSIS")
print("="*60)
report_dict = classification_report(y_int_te, y_pred, target_names=le.classes_, output_dict=True)

# Highlight problematic classes
for scanner in le.classes_:
    f1 = report_dict[scanner]['f1-score']
    status = "✅" if f1 >= 0.85 else "🟡" if f1 >= 0.5 else "❌"
    print(f"{status} {scanner}: F1={f1*100:.1f}%")

# Special attention to Canon9000
print("\n⚠️  CANON9000 FOCUS:")
for scanner in ["Canon9000-1", "Canon9000-2"]:
    if scanner in le.classes_:
        metrics = report_dict[scanner]
        print(f"  {scanner}: Precision={metrics['precision']*100:.1f}%, Recall={metrics['recall']*100:.1f}%, F1={metrics['f1-score']*100:.1f}%")

# ---- Confusion Matrix ----
cm = confusion_matrix(y_int_te, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Hybrid CNN Confusion Matrix (Acc: {test_acc*100:.2f}%)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
print(f"\nSaved confusion matrix to: {cm_path}")

# ---- Save Detailed Report ----
report_path = os.path.join(RESULTS_DIR, "evaluation_report.txt")
with open(report_path, "w") as f:
    f.write("="*60 + "\n")
    f.write("HYBRID CNN EVALUATION REPORT\n")
    f.write("="*60 + "\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Test Samples: {len(X_img_te)}\n")
    f.write(f"\n{'='*60}\n")
    f.write(f"OVERALL ACCURACY: {test_acc*100:.2f}%\n")
    f.write(f"TARGET: >85%\n")
    if test_acc >= 0.85:
        f.write("STATUS: ✅ TARGET ACHIEVED!\n")
    else:
        f.write(f"STATUS: Gap = {(0.85-test_acc)*100:.2f}%\n")
    f.write(f"{'='*60}\n\n")
    f.write("Classification Report:\n")
    f.write(report_str)
    f.write("\n\nPer-Class Summary:\n")
    for scanner in le.classes_:
        f1 = report_dict[scanner]['f1-score']
        status = "PASS" if f1 >= 0.85 else "WARN" if f1 >= 0.5 else "FAIL"
        f.write(f"  [{status}] {scanner}: F1={f1*100:.1f}%\n")

print(f"Saved evaluation report to: {report_path}")

# ---- Summary ----
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Target: >85%")
if test_acc >= 0.85:
    print("🎉 TARGET ACHIEVED!")
else:
    print(f"Gap to target: {(0.85-test_acc)*100:.2f}%")
print("="*60)

