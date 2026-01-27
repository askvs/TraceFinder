import os
import glob
import pickle
import numpy as np
import tensorflow as tf
import cv2
import csv
from utils import process_batch_gpu, batch_corr_gpu, extract_enhanced_features, to_gray, resize_to, normalize_img

# Paths - Fixed for correct project structure
BASE_DIR = "../../data"

# Model files from models/hybrid_cnn
MODEL_DIR = "../../models/hybrid_cnn"
# Processed data from processed_data/hybrid_cnn
PROCESSED_DIR = "../../processed_data/hybrid_cnn"

FP_PATH = os.path.join(PROCESSED_DIR, "scanner_fingerprints.pkl")
ORDER_NPY = os.path.join(PROCESSED_DIR, "fp_keys.npy")
CKPT_PATH = os.path.join(MODEL_DIR, "scanner_hybrid_final.keras")
ENCODER_PATH = os.path.join(MODEL_DIR, "hybrid_label_encoder.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "hybrid_feat_scaler.pkl")

IMG_SIZE = (256, 256)

# Load model & preprocessors
print(f"Loading model from {CKPT_PATH}")
hyb_model = tf.keras.models.load_model(CKPT_PATH, compile=False)

print("Loading artifacts...")
with open(ENCODER_PATH, "rb") as f:
    le_inf = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler_inf = pickle.load(f)

with open(FP_PATH, "rb") as f:
    scanner_fps_inf = pickle.load(f)

fp_keys_inf = np.load(ORDER_NPY, allow_pickle=True).tolist()

def predict_batch(image_paths):
    """
    Predict a batch of images using GPU acceleration.
    """
    # 1. Preprocess Batch (Residuals)
    residuals = process_batch_gpu(image_paths) 
    
    if not residuals:
        return []
        
    residuals = np.array(residuals, dtype=np.float32) # (B, 256, 256)
    
    # 2. Extract Features
    corrs = batch_corr_gpu(residuals, scanner_fps_inf, fp_keys_inf) # (B, K)
    
    enh_feats = []
    for res in residuals:
        enh_feats.append(extract_enhanced_features(res))
    enh_feats = np.array(enh_feats, dtype=np.float32) # (B, F_enh)
    
    # Combine & Scale
    feats_combined = np.hstack([corrs, enh_feats]) # (B, K+F_enh)
    feats_scaled = scaler_inf.transform(feats_combined)
    
    # 3. Model Inference
    X_img = np.expand_dims(residuals, -1)
    
    probs = hyb_model.predict([X_img, feats_scaled], verbose=0) # (B, NumClasses)
    
    results = []
    for i, prob in enumerate(probs):
        idx = int(np.argmax(prob))
        label = le_inf.classes_[idx]
        conf = float(prob[idx] * 100)
        results.append((image_paths[i], label, conf))
        
    return results

def predict_folder(folder_path, output_csv="results.csv", exts=("*.tif","*.png","*.jpg","*.jpeg")):
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))
    print(f"Found {len(image_files)} images in {folder_path}")

    all_results = []
    BATCH_SIZE = 32
    
    # Process in batches
    for i in range(0, len(image_files), BATCH_SIZE):
        batch_files = image_files[i : i+BATCH_SIZE]
        batch_results = predict_batch(batch_files)
        
        for res in batch_results:
            print(f"{res[0]} -> {res[1]} | {res[2]:.2f}%")
            all_results.append(res)

    # Save CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Predicted_Label", "Confidence(%)"])
        writer.writerows(all_results)
    print(f"\n Predictions saved to {output_csv}")
    return all_results

if __name__ == "__main__":
    folder_to_test = os.path.join(BASE_DIR, "Test_Images") 
    if not os.path.exists(folder_to_test):
        print(f"Test folder {folder_to_test} not found. Please set 'folder_to_test' variable or create the folder.")
    else:
        predict_folder(folder_to_test, output_csv="hybrid_folder_results.csv")
