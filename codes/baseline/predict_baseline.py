
import cv2
import os
import numpy as np
import joblib
from skimage.restoration import denoise_wavelet
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy
import pandas as pd
from skimage.feature import local_binary_pattern


def load_and_preprocess(img_path, size=(512, 512)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f" Could not load image: {img_path}")
    img = img.astype(np.float32) / 255.0
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def extract_noise_residual(img):
    denoised = denoise_wavelet(img, channel_axis=None, rescale_sigma=True)
    return img - denoised

def compute_metadata_features(img, file_path, resolution):
    h, w = img.shape
    aspect_ratio = w / h
    file_size_kb = os.path.getsize(file_path) / 1024

    residual = extract_noise_residual(img)
    pixels = residual.flatten()
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    skewness = skew(pixels)
    kurt = kurtosis(pixels)
    ent = entropy(np.histogram(pixels, bins=256)[0] + 1e-6)

    edges = sobel(img)
    edge_density = np.mean(edges > 0.1)

    features = {
        "resolution": resolution,
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "skewness": skewness,
        "kurtosis": kurt,
        "entropy": ent,
        "edge_density": edge_density
    }
        
    return features


def predict_scanner(img_path, model_choice="rf"):
    
    # Get the absolute path to the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    scaler_path = os.path.join(project_root, "models", "scaler.pkl")
    scaler = joblib.load(scaler_path)

    if model_choice == "rf":
        model_path = os.path.join(project_root, "models", "random_forest.pkl")
    else:
        model_path = os.path.join(project_root, "models", "svm.pkl")
    model = joblib.load(model_path)

    
    img = load_and_preprocess(img_path)
    
    # Extract resolution from path
    try:
        resolution = int(os.path.basename(os.path.dirname(img_path)))
    except (IndexError, ValueError):
        resolution = 0 # Or some default value

    features = compute_metadata_features(img, img_path, resolution)

    
    df = pd.DataFrame([features])

    # Reorder columns to match training order
    feature_order = [
        'resolution', 'width', 'height', 'aspect_ratio', 'file_size_kb',
        'mean_intensity', 'std_intensity', 'skewness', 'kurtosis', 'entropy',
        'edge_density'
    ]

    df = df[feature_order]
    
    X_scaled = scaler.transform(df)

    
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]

    return pred, prob


if __name__ == "__main__":
    test_image = "data/Official/EpsonV39-1/EpsonV39-1/300/s8_1.tif"  
    pred, prob = predict_scanner(test_image, model_choice="rf")
    print("Predicted Scanner:", pred)
    print("Class Probabilities:", prob)
    print(" Prediction completed successfully!")