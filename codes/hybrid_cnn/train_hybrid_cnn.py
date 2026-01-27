import os
import pickle
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from collections import Counter

from utils import corr2d, extract_enhanced_features
from model import build_hybrid_model

# ============================================================
# TRAINING CONFIGURATION
# ============================================================
CONFIG = {
    "epochs": 50,              # Same as original
    "batch_size": 32,
    "learning_rate": 1e-3,     # Simple fixed LR (no complex scheduling)
    "patience_early_stop": 10, # Original patience
    "patience_lr_reduce": 5,
    "use_class_weights": True, # KEY CHANGE: Handle Canon9000-1/2 imbalance
}

# ---- Local Paths ---- 
BASE_DIR = "../../data"
PROCESSED_DIR = "../../processed_data/hybrid_cnn"
MODEL_DIR = "../../models/hybrid_cnn"
RESULTS_DIR = "../../results/hybrid_cnn"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Paths for loading processed data
RES_PATH  = os.path.join(PROCESSED_DIR, "official_wiki_residuals.pkl")
FP_PATH   = os.path.join(PROCESSED_DIR, "scanner_fingerprints.pkl")
ORDER_NPY = os.path.join(PROCESSED_DIR, "fp_keys.npy")
FEATURES_PATH = os.path.join(PROCESSED_DIR, "features.pkl")
ENHANCED_PATH = os.path.join(PROCESSED_DIR, "enhanced_features.pkl")

# ---- Reproducibility ----
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---- GPU Setup ----
gpus = tf.config.list_physical_devices('GPU')
device_name = '/GPU:0' if gpus else '/CPU:0'
print("Using device:", device_name)


# Load & Prepare Data
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
        raise FileNotFoundError("Scanner fingerprints not found. Run feature_extrac.py first.")

X_img, X_feat, y = [], [], []
idx_counter = 0

for dataset_name in ["Official", "WikiPedia"]:  
    if dataset_name not in residuals_dict:
        continue
    
    print(f"Processing {dataset_name}...")
    for scanner, dpi_dict in residuals_dict[dataset_name].items():
        if isinstance(dpi_dict, dict):
             for dpi, res_list in dpi_dict.items():
                for res in res_list:
                    X_img.append(np.expand_dims(res, -1))
                    
                    if precomputed:
                        f_p = feats_prnu[idx_counter]
                        f_e = feats_enh[idx_counter]
                        X_feat.append(f_p + f_e)
                    else:
                        v_corr = [corr2d(res, scanner_fps[k]) for k in fp_keys]
                        v_enh = extract_enhanced_features(res)
                        X_feat.append(v_corr + v_enh)
                    
                    y.append(scanner)
                    idx_counter += 1

X_img = np.array(X_img, dtype=np.float32)
X_feat = np.array(X_feat, dtype=np.float32)
y = np.array(y)

print(f"Dataset Shape: Images {X_img.shape}, Features {X_feat.shape}, Labels {y.shape}")

# Encode Labels
le = LabelEncoder()
y_int = le.fit_transform(y)
num_classes = len(le.classes_)
y_cat = to_categorical(y_int, num_classes)

# Print class distribution
print("\n" + "="*50)
print("CLASS DISTRIBUTION")
print("="*50)
class_counts = Counter(y)
for scanner, count in sorted(class_counts.items()):
    print(f"  {scanner}: {count} samples")
print("="*50 + "\n")

# Compute class weights for imbalanced data
if CONFIG["use_class_weights"]:
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_int),
        y=y_int
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights_array)}
    print("Class Weights (to handle imbalance):")
    for i, scanner in enumerate(le.classes_):
        print(f"  {scanner}: {class_weight_dict[i]:.3f}")
    print()
else:
    class_weight_dict = None

# Train/Test Split
X_img_tr, X_img_te, X_feat_tr, X_feat_te, y_tr, y_te = train_test_split(
    X_img, X_feat, y_cat, test_size=0.2, random_state=SEED, stratify=y_int
)

# Normalize Features
scaler = StandardScaler()
X_feat_tr = scaler.fit_transform(X_feat_tr)
X_feat_te = scaler.transform(X_feat_te)

# Save Preprocessors
with open(os.path.join(MODEL_DIR, "hybrid_label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)
with open(os.path.join(MODEL_DIR, "hybrid_feat_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Build & Train Model
print("\n" + "="*50)
print("BUILDING HYBRID CNN MODEL (V3 - Simplified)")
print("="*50)

with tf.device(device_name):
    model = build_hybrid_model(
        img_shape=(256, 256, 1), 
        feat_shape=(X_feat.shape[1],), 
        num_classes=num_classes
    )
    
    # Simple optimizer (no complex scheduling)
    optimizer = keras.optimizers.Adam(learning_rate=CONFIG["learning_rate"])
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy", 
        metrics=["accuracy"]
    )
    model.summary()

    # Data Pipeline
    BATCH = CONFIG["batch_size"]
    train_ds = tf.data.Dataset.from_tensor_slices(((X_img_tr, X_feat_tr), y_tr))\
        .shuffle(len(y_tr)).batch(BATCH).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices(((X_img_te, X_feat_te), y_te))\
        .batch(BATCH).prefetch(tf.data.AUTOTUNE)

    # Callbacks (simple, reliable)
    ckpt_path = os.path.join(MODEL_DIR, "scanner_hybrid_v3.keras")
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=CONFIG["patience_early_stop"], 
            restore_best_weights=True, 
            monitor="val_accuracy",
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, 
            patience=CONFIG["patience_lr_reduce"], 
            min_lr=1e-6, 
            monitor="val_accuracy",
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            ckpt_path, 
            save_best_only=True, 
            monitor="val_accuracy",
            verbose=1
        ),
    ]

    print(f"\nTraining with class_weights: {CONFIG['use_class_weights']}")
    print(f"Epochs: {CONFIG['epochs']}, Batch Size: {BATCH}")
    print(f"Early stopping patience: {CONFIG['patience_early_stop']}\n")

    # Train with class weights
    history = model.fit(
        train_ds, 
        epochs=CONFIG["epochs"], 
        validation_data=val_ds, 
        callbacks=callbacks,
        class_weight=class_weight_dict
    )

    # Save Final Model
    final_model_path = os.path.join(MODEL_DIR, "scanner_hybrid_final.keras")
    model.save(final_model_path)
    print(f"\n✅ Final model saved to: {final_model_path}")
    
    with open(os.path.join(RESULTS_DIR, "hybrid_training_history.pkl"), "wb") as f:
        pickle.dump(history.history, f)
    
    # Print final results
    best_val_acc = max(history.history['val_accuracy'])
    final_train_acc = history.history['accuracy'][-1]
    print(f"\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Final Training Accuracy: {final_train_acc*100:.2f}%")
    print(f"Target: >85%")
    if best_val_acc >= 0.85:
        print("🎉 TARGET ACHIEVED!")
    else:
        print(f"Gap to target: {(0.85 - best_val_acc)*100:.2f}%")
    print("="*50)

# Plot Training Results
def plot_history(hist, save_dir):
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'r*-', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'r*-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(save_dir, "training_plot.png")
    plt.savefig(save_path)
    print(f" Training plots saved to {save_path}")

plot_history(history, RESULTS_DIR)
print(" Training complete!")
