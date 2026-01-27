"""
processing.py
Data loading and preprocessing functions.
- Handles `Flatfield`, `official`, `Wikipedia` datasets.
- Produces: residual images (256x256x1) and saves them as pickle files.
"""

import os
import pickle
from tqdm import tqdm
from utils import process_batch_gpu, USE_GPU

# Fallback CPU
import cv2
import numpy as np
import pywt
from scipy.signal import wiener as scipy_wiener
from concurrent.futures import ThreadPoolExecutor, as_completed


# Global Parameters
IMG_SIZE = (256, 256)
DENOISE_METHOD = "wavelet"
BATCH_SIZE = 64
MAX_WORKERS = 8

print(f"Dataset Processing: Use GPU? {USE_GPU}")

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

def resize_to(img, size=IMG_SIZE):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def normalize_img(img):
    return img.astype(np.float32) / 255.0

def denoise_wavelet_img_cpu(img):
    """Wavelet denoising (Haar) on CPU using PyWavelets."""
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    cH[:] = 0; cV[:] = 0; cD[:] = 0
    return pywt.idwt2((cA, (cH, cV, cD)), 'haar')

def preprocess_image_cpu(fpath, method=DENOISE_METHOD):
    """Process single image on CPU."""
    img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img = to_gray(img)
    img = resize_to(img)
    img = normalize_img(img)
    # Denoising
    if method == "wiener":
        den = scipy_wiener(img, mysize=(5,5))
    elif method == "wavelet":
        den = denoise_wavelet_img_cpu(img)
    else:
        raise ValueError(f"Unknown denoise method: {method}")
    
    # Residual
    res = (img - den).astype(np.float32)
    return res

def process_folder(base_dir, use_dpi_subfolders=True):
    """
    Process all images under a folder.
    Returns nested dict: residuals[scanner][dpi] = list_of_residuals
    """
    residuals_dict = {}
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist!")
        return {}

    scanners = sorted(os.listdir(base_dir))

    for scanner in tqdm(scanners, desc="Scanners"):
        scanner_dir = os.path.join(base_dir, scanner)
        if not os.path.isdir(scanner_dir):
            continue

        residuals_dict[scanner] = {}

        if use_dpi_subfolders:
            for dpi in os.listdir(scanner_dir):
                dpi_dir = os.path.join(scanner_dir, dpi)
                if not os.path.isdir(dpi_dir):
                    continue
                files = [os.path.join(dpi_dir, f) for f in os.listdir(dpi_dir)
                         if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))]
                
                # Process files
                residuals_dict[scanner][dpi] = run_processing(files)
        else:
            # Flatfield dataset: no dpi subfolders
            files = [os.path.join(scanner_dir, f) for f in os.listdir(scanner_dir)
                     if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))]
            residuals_dict[scanner] = run_processing(files)

    return residuals_dict

def run_processing(file_list):
    """Dispatches to GPU batch or CPU pool."""
    if not file_list:
        return []
        
    all_residuals = []
    
    if USE_GPU:
        # Batch chunks
        for i in range(0, len(file_list), BATCH_SIZE):
            chunk = file_list[i : i + BATCH_SIZE]
            res_chunk = process_batch_gpu(chunk)
            all_residuals.extend(res_chunk)
    else:
        # Fallback to ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(preprocess_image_cpu, f) for f in file_list]
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    all_residuals.append(res)
                    
    return all_residuals

def save_pickle(obj, path):
    # Ensure dir exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved residuals to {path}")


# Main Execution
if __name__ == "__main__":
    # Fixed path: relative to codes/hybrid_cnn folder
    BASE_DIR = "../../data"
    # Intermediate files go to processed_data/hybrid_cnn
    OUT_DIR = "../../processed_data/hybrid_cnn"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Official + Wikipedia
    datasets = ["Official", "WikiPedia"]  
    official_wiki_residuals = {}

    for dataset in datasets:
        print(f"\nProcessing {dataset} dataset...")
        dataset_dir = os.path.join(BASE_DIR, dataset)
        official_wiki_residuals[dataset] = process_folder(dataset_dir, use_dpi_subfolders=True)

    OUT_PATH = os.path.join(OUT_DIR, "official_wiki_residuals.pkl")
    save_pickle(official_wiki_residuals, OUT_PATH)

    
    print("\nProcessing Flatfield dataset...")
    flatfield_dir = os.path.join(BASE_DIR, "Flatfield")
    flatfield_residuals = process_folder(flatfield_dir, use_dpi_subfolders=False)
    OUT_PATH_FLAT = os.path.join(OUT_DIR, "flatfield_residuals.pkl")
    save_pickle(flatfield_residuals, OUT_PATH_FLAT)

    # Summary
    total_scanners = len(flatfield_residuals)
    total_images = sum([len(v) for v in flatfield_residuals.values() if isinstance(v, list)])
    print(f"\nDone. Flatfield: {total_scanners} scanners, ~{total_images} images processed.")
    
    # Summary for Official + Wikipedia
    for ds_name, ds_data in official_wiki_residuals.items():
        total_sc = len(ds_data)
        total_img = sum([len(res_list) for scanner_data in ds_data.values() 
                        for dpi_key, res_list in (scanner_data.items() if isinstance(scanner_data, dict) else [])])
        print(f"{ds_name}: {total_sc} scanners, ~{total_img} images processed.")
