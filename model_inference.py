"""
Model Inference Module for Trace Finder
========================================
Unified interface for all scanner identification models:
- Hybrid CNN (TensorFlow/Keras)
- Standalone CNN (PyTorch)
- Baseline models (SVM, Random Forest)
"""
# This is the code is for the managing the all the models to the landing page of this project. where it loads the models and do the inference.
import os
import pickle
import numpy as np
import cv2
from PIL import Image
import io

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Hybrid CNN paths
HYBRID_MODEL_PATH = os.path.join(MODELS_DIR, "hybrid_cnn", "scanner_hybrid.keras")
HYBRID_ENCODER_PATH = os.path.join(MODELS_DIR, "hybrid_cnn", "hybrid_label_encoder.pkl")
HYBRID_SCALER_PATH = os.path.join(MODELS_DIR, "hybrid_cnn", "hybrid_feat_scaler.pkl")
FINGERPRINTS_PATH = os.path.join(BASE_DIR, "processed_data", "hybrid_cnn", "scanner_fingerprints.pkl")
FP_KEYS_PATH = os.path.join(BASE_DIR, "processed_data", "hybrid_cnn", "fp_keys.npy")

# Standalone CNN paths
CNN_MODEL_PATH = os.path.join(MODELS_DIR, "cnn", "cnn_model.pth")

# Baseline paths
RF_MODEL_PATH = os.path.join(MODELS_DIR, "baseline", "random_forest.joblib")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "baseline", "svm.joblib")
BASELINE_SCALER_PATH = os.path.join(MODELS_DIR, "baseline", "scaler.joblib")


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================
def corr2d(a, b):
    """Compute 2D correlation coefficient between two images/residuals."""
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a @ b) / denom) if denom != 0 else 0.0


def preprocess_for_hybrid(image_bytes):
    """
    Preprocess image for Hybrid CNN model.
    Returns: (residual_256x256x1, handcrafted_features)
    
    Features = PRNU correlations (11) + Enhanced features (33) = 44 total
    """
    import pywt
    
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError("Could not decode image")
    
    # Convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to 256x256
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # Compute residual using wavelet approximation (Haar L1)
    coeffs = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs
    cH[:] = 0
    cV[:] = 0
    cD[:] = 0
    denoised = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    
    # Handle size mismatch from wavelet
    if denoised.shape != img.shape:
        denoised = cv2.resize(denoised, (256, 256))
    
    residual = (img - denoised).astype(np.float32)
    
    # Load fingerprints for PRNU correlations
    fingerprints = None
    fp_keys = None
    if os.path.exists(FINGERPRINTS_PATH):
        with open(FINGERPRINTS_PATH, 'rb') as f:
            fingerprints = pickle.load(f)
        if os.path.exists(FP_KEYS_PATH):
            fp_keys = np.load(FP_KEYS_PATH, allow_pickle=True).tolist()
    
    # Compute PRNU correlations (11 features)
    prnu_features = []
    if fingerprints and fp_keys:
        for k in fp_keys:
            corr = corr2d(residual, fingerprints[k])
            prnu_features.append(corr)
    else:
        # Fallback: zeros if no fingerprints
        prnu_features = [0.0] * 11
    
    # Extract enhanced features (33 features)
    enhanced_features = extract_enhanced_features(residual)
    
    # Combine: 11 PRNU + 33 enhanced = 44 total features
    all_features = prnu_features + enhanced_features.tolist()
    
    return residual, np.array(all_features, dtype=np.float32)


def preprocess_for_cnn(image_bytes):
    """
    Preprocess image for standalone CNN model.
    Returns: tensor ready for PyTorch (1, 3, 128, 128)
    """
    import torch
    from torchvision import transforms
    
    # Decode image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Transform pipeline (must match training)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return tensor


def preprocess_for_baseline(image_bytes, file_size_bytes):
    """
    Preprocess image for baseline models (SVM/RF).
    Returns: feature vector matching training format
    """
    from skimage.restoration import denoise_wavelet
    from skimage.filters import sobel
    from scipy.stats import skew, kurtosis, entropy
    
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Could not decode image")
    
    # Resize to 512x512 (baseline training size)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    
    h, w = img.shape
    
    # Extract noise residual
    denoised = denoise_wavelet(img, channel_axis=None, rescale_sigma=True)
    residual = img - denoised
    pixels = residual.flatten()
    
    # Compute features (must match training: width, height, aspect_ratio, file_size_kb,
    # mean_intensity, std_intensity, skewness, kurtosis, entropy, edge_density)
    features = {
        'width': w,
        'height': h,
        'aspect_ratio': w / h,
        'file_size_kb': file_size_bytes / 1024,
        'mean_intensity': float(np.mean(pixels)),
        'std_intensity': float(np.std(pixels)),
        'skewness': float(skew(pixels)),
        'kurtosis': float(kurtosis(pixels)),
        'entropy': float(entropy(np.histogram(pixels, bins=256)[0] + 1e-6)),
        'edge_density': float(np.mean(sobel(img) > 0.1))
    }
    
    return features


def extract_enhanced_features(residual):
    """
    Extract FFT + LBP + texture features for Hybrid CNN.
    """
    from scipy.fft import fft2, fftshift
    from scipy import ndimage
    from skimage.feature import local_binary_pattern as sk_lbp
    
    # FFT Features
    f = fftshift(fft2(residual))
    mag = np.abs(f)
    h, w = mag.shape
    center_h, center_w = h // 2, w // 2
    
    # 3 frequency bands
    low_freq = np.mean(mag[max(0, center_h-20):center_h+20, 
                          max(0, center_w-20):center_w+20])
    mid_region = mag[max(0, center_h-60):center_h+60, 
                    max(0, center_w-60):center_w+60]
    mid_freq = np.mean(mid_region) - low_freq
    high_freq = np.mean(mag) - np.mean(mid_region)
    
    # LBP histogram
    lbp = sk_lbp(residual, P=24, R=3, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=26, range=(0, 26), density=True)
    
    # Gradient / texture
    grad_x = ndimage.sobel(residual, axis=1)
    grad_y = ndimage.sobel(residual, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    texture_features = [
        np.std(residual),
        np.mean(np.abs(residual)),
        np.std(grad_mag),
        np.mean(grad_mag)
    ]
    
    # Combine: 3 FFT + 26 LBP + 4 texture = 33 features
    all_features = ([float(low_freq), float(mid_freq), float(high_freq)] + 
                   lbp_hist.tolist() + texture_features)
    
    return np.array(all_features, dtype=np.float32)


# ============================================================================
# MODEL LOADING (Cached)
# ============================================================================
_cached_models = {}

def get_hybrid_model():
    """Load Hybrid CNN model and encoders."""
    if 'hybrid' not in _cached_models:
        import tensorflow as tf
        
        model = tf.keras.models.load_model(HYBRID_MODEL_PATH)
        
        with open(HYBRID_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open(HYBRID_SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load fingerprints for PRNU correlation
        fingerprints = None
        fp_keys = None
        if os.path.exists(FINGERPRINTS_PATH):
            with open(FINGERPRINTS_PATH, 'rb') as f:
                fingerprints = pickle.load(f)
            if os.path.exists(FP_KEYS_PATH):
                fp_keys = np.load(FP_KEYS_PATH, allow_pickle=True).tolist()
        
        _cached_models['hybrid'] = {
            'model': model,
            'encoder': label_encoder,
            'scaler': scaler,
            'fingerprints': fingerprints,
            'fp_keys': fp_keys
        }
    
    return _cached_models['hybrid']


def get_cnn_model():
    """Load standalone PyTorch CNN model."""
    if 'cnn' not in _cached_models:
        import torch
        import sys
        cnn_model_path = os.path.join(BASE_DIR, 'codes', 'cnn_model')
        if cnn_model_path not in sys.path:
            sys.path.insert(0, cnn_model_path)
        try:
            from model import SimpleCNN
        except ImportError:
            # Fallback: try to load from alternative location
            from .codes.cnn_model.model import SimpleCNN
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleCNN(num_classes=11)
        model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        
        # Scanner class names (from training)
        class_names = [
            'Canon9000-1', 'Canon9000-2', 'Epson10000XL', 'Epson12000XL',
            'EpsonV39-1', 'EpsonV39-2', 'HP_8300', 'Mustek-1200Cu',
            'Plustek-OpticPro', 'Ricoh-MP2014', 'Samsung-SCX4521F'
        ]
        
        _cached_models['cnn'] = {
            'model': model,
            'device': device,
            'class_names': class_names
        }
    
    return _cached_models['cnn']


def get_baseline_model(model_type='rf'):
    """Load baseline SVM or Random Forest model."""
    import joblib
    
    key = f'baseline_{model_type}'
    if key not in _cached_models:
        if model_type == 'rf':
            model = joblib.load(RF_MODEL_PATH)
        else:
            model = joblib.load(SVM_MODEL_PATH)
        
        scaler = joblib.load(BASELINE_SCALER_PATH)
        
        _cached_models[key] = {
            'model': model,
            'scaler': scaler
        }
    
    return _cached_models[key]


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================
def predict_hybrid_cnn(image_bytes):
    """
    Run Hybrid CNN inference.
    Returns: (scanner_name, confidence, all_probabilities)
    """
    import tensorflow as tf
    
    # Preprocess
    residual, features = preprocess_for_hybrid(image_bytes)
    
    # Load model
    hybrid = get_hybrid_model()
    model = hybrid['model']
    encoder = hybrid['encoder']
    scaler = hybrid['scaler']
    
    # Prepare inputs
    residual_input = residual.reshape(1, 256, 256, 1)
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Predict
    predictions = model.predict([residual_input, features_scaled], verbose=0)
    
    pred_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][pred_idx]) * 100
    scanner_name = encoder.inverse_transform([pred_idx])[0]
    
    # All class probabilities
    all_probs = {encoder.inverse_transform([i])[0]: float(p) * 100 
                 for i, p in enumerate(predictions[0])}
    
    return scanner_name, confidence, all_probs


def predict_cnn(image_bytes):
    """
    Run standalone CNN inference.
    Returns: (scanner_name, confidence, all_probabilities)
    """
    import torch
    import torch.nn.functional as F
    
    # Preprocess
    input_tensor = preprocess_for_cnn(image_bytes)
    
    # Load model
    cnn = get_cnn_model()
    model = cnn['model']
    device = cnn['device']
    class_names = cnn['class_names']
    
    # Predict
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
    
    probs = probabilities.cpu().numpy()[0]
    pred_idx = np.argmax(probs)
    confidence = float(probs[pred_idx]) * 100
    scanner_name = class_names[pred_idx]
    
    # All class probabilities
    all_probs = {class_names[i]: float(p) * 100 for i, p in enumerate(probs)}
    
    return scanner_name, confidence, all_probs


def predict_baseline(image_bytes, file_size_bytes, model_type='rf'):
    """
    Run baseline model inference (SVM or Random Forest).
    Returns: (scanner_name, confidence, all_probabilities)
    """
    import pandas as pd
    
    # Preprocess
    features = preprocess_for_baseline(image_bytes, file_size_bytes)
    
    # Load model
    baseline = get_baseline_model(model_type)
    model = baseline['model']
    scaler = baseline['scaler']
    
    # Prepare feature vector (must match training order exactly)
    feature_order = [
        'width', 'height', 'aspect_ratio', 'file_size_kb',
        'mean_intensity', 'std_intensity', 'skewness', 'kurtosis', 
        'entropy', 'edge_density'
    ]
    
    df = pd.DataFrame([features])
    df = df[feature_order]
    X_scaled = scaler.transform(df)
    
    # Predict
    pred = model.predict(X_scaled)[0]
    
    # Get probabilities (handle SVM without probability=True)
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_scaled)[0]
    elif hasattr(model, 'decision_function'):
        # Use decision function and apply softmax for pseudo-probabilities
        decision = model.decision_function(X_scaled)[0]
        # Handle binary vs multiclass
        if decision.ndim == 0 or (hasattr(decision, '__len__') and len(model.classes_) == 2):
            # Binary classification
            decision = np.array([decision]) if decision.ndim == 0 else decision
            exp_d = np.exp(np.clip(decision, -100, 100))
            probs = np.array([1 / (1 + exp_d[0]), exp_d[0] / (1 + exp_d[0])])
        else:
            # Multiclass: apply softmax
            exp_d = np.exp(decision - np.max(decision))  # Numerical stability
            probs = exp_d / exp_d.sum()
    else:
        # Fallback: 100% confidence on predicted class
        probs = np.zeros(len(model.classes_))
        pred_idx = np.where(model.classes_ == pred)[0][0]
        probs[pred_idx] = 1.0
    
    confidence = float(np.max(probs)) * 100
    
    # All class probabilities
    all_probs = {model.classes_[i]: float(p) * 100 for i, p in enumerate(probs)}
    
    return pred, confidence, all_probs


# ============================================================================
# GRAD-CAM VISUALIZATION
# ============================================================================
def generate_gradcam(image_bytes, target_class=None):
    """
    Generate Grad-CAM heatmap for Hybrid CNN.
    Returns: (heatmap_image, prediction, confidence)
    """
    import tensorflow as tf
    
    # Preprocess
    residual, features = preprocess_for_hybrid(image_bytes)
    
    # Load model
    hybrid = get_hybrid_model()
    model = hybrid['model']
    encoder = hybrid['encoder']
    scaler = hybrid['scaler']
    
    # Prepare inputs
    residual_input = tf.convert_to_tensor(residual.reshape(1, 256, 256, 1))
    features_scaled = scaler.transform(features.reshape(1, -1))
    features_input = tf.convert_to_tensor(features_scaled.astype(np.float32))
    
    # Find last conv layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            last_conv_layer = layer
            break
    
    if last_conv_layer is None:
        return None, None, None
    
    # Create gradient model
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([residual_input, features_input])
        if target_class is None:
            target_class = tf.argmax(predictions[0])
        class_channel = predictions[:, target_class]
    
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    
    # Resize to original size
    heatmap = cv2.resize(heatmap, (256, 256))
    
    # Create colored heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Get prediction info
    pred_idx = int(tf.argmax(predictions[0]))
    confidence = float(predictions[0][pred_idx]) * 100
    scanner_name = encoder.inverse_transform([pred_idx])[0]
    
    # Create overlay
    original = (residual * 255).astype(np.uint8)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(original_rgb, 0.6, heatmap_colored, 0.4, 0)
    
    return overlay, scanner_name, confidence


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def check_models_available():
    """Check which models are available."""
    available = {
        'hybrid_cnn': os.path.exists(HYBRID_MODEL_PATH),
        'cnn': os.path.exists(CNN_MODEL_PATH),
        'random_forest': os.path.exists(RF_MODEL_PATH),
        'svm': os.path.exists(SVM_MODEL_PATH)
    }
    return available


if __name__ == "__main__":
    # Quick test
    print("Checking available models...")
    available = check_models_available()
    for model, status in available.items():
        print(f"  {model}: {'✓ Available' if status else '✗ Not found'}")
