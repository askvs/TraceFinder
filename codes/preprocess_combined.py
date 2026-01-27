import os
import cv2
import csv
import numpy as np
from skimage import io, img_as_float
from skimage.restoration import denoise_wavelet
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy


# PATH CONFIGURATION

# Get the absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root
project_root = os.path.dirname(script_dir)


DATA_ROOT = os.path.join(project_root, "data")
OFFICIAL_DIR = os.path.join(DATA_ROOT, "Official")
WIKIPEDIA_DIR = os.path.join(DATA_ROOT, "WikiPedia")

OUTPUT_DIR = os.path.join(project_root, "processed_data")
PATCHES_DIR = os.path.join(OUTPUT_DIR, "patches")
CSV_PATH = os.path.join(OUTPUT_DIR, "combined_features.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# IMAGE PROCESSING FUNCTIONS


def load_original(img_path):
    img = io.imread(img_path, as_gray=True)
    return img_as_float(img)

def extract_noise_residual(img):
    """Compute noise residual."""
    denoised = denoise_wavelet(img, channel_axis=None, rescale_sigma=True)
    return img - denoised

def extract_patches(img, patch_size=128, stride=128):
    """Extract fixed-size patches from original image."""
    patches = []
    h, w = img.shape

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patches.append(img[i:i + patch_size, j:j + patch_size])

    return patches


# FEATURE EXTRACTION


def compute_metadata_features(img, file_path, scanner_id, dataset_source, resolution_folder="unknown"):
    """Compute metadata features from original image."""
    
    h, w = img.shape
    aspect_ratio = w / h
    file_size_kb = os.path.getsize(file_path) / 1024

    pixels = img.flatten()

    # Histogram entropy
    hist = np.histogram(pixels, bins=256, range=(0, 1))[0] + 1e-6

    return {
        "file_name": os.path.basename(file_path),
        "dataset_source": dataset_source,
        "main_class": dataset_source.capitalize(),
        "resolution": resolution_folder,
        "class_label": scanner_id,

        # Image geometry
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,

        # File properties
        "file_size_kb": file_size_kb,

        # Pixel statistics
        "mean_intensity": np.mean(pixels),
        "std_intensity": np.std(pixels),
        "skewness": skew(pixels),
        "kurtosis": kurtosis(pixels),
        "entropy": entropy(hist),

        # Edge features
        "edge_density": np.mean(sobel(img) > 0.1)
    }


# PROCESS A DATASET DIRECTORY


def process_directory(source_dir, dataset_source, writer):
    print(f"\n--- Processing dataset: {dataset_source.upper()} ---")

    for root, dirs, files in os.walk(source_dir):

        if root == source_dir:
            continue  # Skip top folder

        # Extract scanner ID + resolution folder name
        rel_path = os.path.relpath(root, source_dir)
        parts = rel_path.split(os.sep)

        if len(parts) < 2:
            continue

        scanner_id = parts[0]
        resolution_folder = parts[-1] if parts[-1].isdigit() else "unknown"

        # Output patch save directory
        save_patch_dir = os.path.join(PATCHES_DIR, dataset_source, scanner_id, resolution_folder)
        os.makedirs(save_patch_dir, exist_ok=True)

        files_processed = 0

        for file in files:

            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                continue

            img_path = os.path.join(root, file)

            try:
                original_img = load_original(img_path)
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
                continue

            # Noise residual for patch extraction
            residual = extract_noise_residual(original_img)
            patches = extract_patches(residual)

            # Save patches
            for idx, patch in enumerate(patches):
                patch_filename = f"{os.path.splitext(file)[0]}_{idx}.npy"
                np.save(os.path.join(save_patch_dir, patch_filename), patch)

            # Metadata (computed from original image)
            features = compute_metadata_features(
                original_img,
                img_path,
                scanner_id,
                dataset_source,
                resolution_folder
            )
            writer.writerow(features)

            files_processed += 1

        if files_processed > 0:
            print(f"Processed {files_processed} files -> {scanner_id}/{resolution_folder}")


# MAIN FUNCTION


def main():
    fieldnames = [
        "file_name", "dataset_source", "main_class", "resolution", "class_label",
        "width", "height", "aspect_ratio", "file_size_kb",
        "mean_intensity", "std_intensity", "skewness", "kurtosis",
        "entropy", "edge_density"
    ]

    with open(CSV_PATH, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        if os.path.exists(OFFICIAL_DIR):
            process_directory(OFFICIAL_DIR, "official", writer)
        else:
            print("Warning: Official dataset folder missing!")

        if os.path.exists(WIKIPEDIA_DIR):
            process_directory(WIKIPEDIA_DIR, "wikipedia", writer)
        else:
            print("Warning: Wikipedia dataset folder missing!")

    print(f"\n Feature extraction complete.\n Metadata saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()
