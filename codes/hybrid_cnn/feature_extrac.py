import os
import pickle
import numpy as np
from tqdm import tqdm
from utils import batch_corr_gpu, extract_enhanced_features

# Paths (local) - Fixed for correct project structure
BASE_DIR = "../../data"
# Intermediate files (residuals, fingerprints, features) go to processed_data/hybrid_cnn
PROCESSED_DIR = "../../processed_data/hybrid_cnn"
os.makedirs(PROCESSED_DIR, exist_ok=True)

FLATFIELD_RESIDUALS_PATH = os.path.join(PROCESSED_DIR, "flatfield_residuals.pkl")
FP_OUT_PATH = os.path.join(PROCESSED_DIR, "scanner_fingerprints.pkl")
ORDER_NPY = os.path.join(PROCESSED_DIR, "fp_keys.npy")

RES_PATH = os.path.join(PROCESSED_DIR, "official_wiki_residuals.pkl")
FEATURES_OUT = os.path.join(PROCESSED_DIR, "features.pkl")
ENHANCED_OUT = os.path.join(PROCESSED_DIR, "enhanced_features.pkl")


# Compute scanner fingerprints
if os.path.exists(FLATFIELD_RESIDUALS_PATH):
    with open(FLATFIELD_RESIDUALS_PATH, "rb") as f:
        flatfield_residuals = pickle.load(f)

    scanner_fingerprints = {}
    print("Computing fingerprints from Flatfields...")
    for scanner, residuals in flatfield_residuals.items():
        if not residuals:
            continue
        stack = np.stack(residuals, axis=0)     
        fingerprint = np.mean(stack, axis=0)      
        scanner_fingerprints[scanner] = fingerprint

    # Save fingerprints
    os.makedirs(os.path.dirname(FP_OUT_PATH), exist_ok=True)
    with open(FP_OUT_PATH, "wb") as f:
        pickle.dump(scanner_fingerprints, f)

    # Save deterministic scanner order
    fp_keys = sorted(scanner_fingerprints.keys())
    np.save(ORDER_NPY, np.array(fp_keys))
    print(f"Saved {len(scanner_fingerprints)} fingerprints and fp_keys.npy")

else:
    print(f"Warning: {FLATFIELD_RESIDUALS_PATH} not found. Skipping fingerprint generation.")
    if os.path.exists(FP_OUT_PATH):
        with open(FP_OUT_PATH, "rb") as f:
            scanner_fingerprints = pickle.load(f)
        if os.path.exists(ORDER_NPY):
            fp_keys = np.load(ORDER_NPY, allow_pickle=True).tolist()
        else:
            fp_keys = sorted(scanner_fingerprints.keys())
    else:
        print("No fingerprints found. Cannot proceed with PRNU features.")
        exit(1)

# ---------------------------
# 2) PRNU Features (Batch GPU)
# ---------------------------
if os.path.exists(RES_PATH):
    # Load residuals
    with open(RES_PATH, "rb") as f:
        residuals_dict = pickle.load(f)

    features, labels = [], []
    
    for dataset_name in ["Official", "WikiPedia"]: 
        if dataset_name not in residuals_dict:
            print(f"Dataset {dataset_name} not in residuals.")
            continue
            
        print(f"Computing PRNU features for {dataset_name} (GPU Batch) ...")
        for scanner, dpi_dict in tqdm(residuals_dict[dataset_name].items()):
            if isinstance(dpi_dict, dict):
                for dpi, res_list in dpi_dict.items():
                    if not res_list: continue
                    corrs = batch_corr_gpu(res_list, scanner_fingerprints, fp_keys) # (N, K)
                    features.extend(corrs.tolist())
                    labels.extend([scanner] * len(res_list))

    # Save features
    with open(FEATURES_OUT, "wb") as f:
        pickle.dump({"features": features, "labels": labels}, f)
    print(f"Saved features shape: {len(features)} x {len(features[0])}")

   
    # Enhanced Features (FFT + LBP + Texture)
    enhanced_features, enhanced_labels = [], []
    for dataset_name in ["Official", "WikiPedia"]:  
        if dataset_name not in residuals_dict: 
            continue
            
        print(f"Extracting enhanced features for {dataset_name} ...")
        for scanner, dpi_dict in tqdm(residuals_dict[dataset_name].items()):
            if isinstance(dpi_dict, dict):
                for dpi, res_list in dpi_dict.items():
                    for res in res_list:
                        feat = extract_enhanced_features(res)
                        enhanced_features.append(feat)
                        enhanced_labels.append(scanner)

    # Save enhanced features
    with open(ENHANCED_OUT, "wb") as f:
        pickle.dump({"features": enhanced_features, "labels": enhanced_labels}, f)
    print(f"Enhanced features shape: {len(enhanced_features)} x {len(enhanced_features[0])}")
    print(f"Saved enhanced features to {ENHANCED_OUT}")

else:
    print(f" {RES_PATH} not found. Cannot extract features.")
