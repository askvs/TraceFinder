import numpy as np
import tensorflow as tf
import cv2
from skimage.feature import local_binary_pattern as sk_lbp
from scipy.fft import fft2, fftshift
from scipy import ndimage

# Check GPU availability once
gpus = tf.config.list_physical_devices('GPU')
USE_GPU = len(gpus) > 0
print(f"Hybrid utils: Use GPU? {USE_GPU}")

def corr2d(a, b):
    """
    Compute 2D correlation coefficient between two images/residuals (CPU version).
    """
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float((a @ b) / denom) if denom != 0 else 0.0

def batch_corr_gpu(residuals, fingerprints, fp_keys):
    """
    Compute normalised cross-correlation for a batch of residuals against all fingerprints using GPU.
    
    Args:
        residuals: list or array of shape (N, H, W) or (N, H, W, 1)
        fingerprints: dict of scanner -> fingerprint (H, W) or (H, W, 1)
        fp_keys: list of keys to ensure order
        
    Returns:
        corrs: shape (N, K) where K is number of fingerprints
    """
    if not residuals:
        return []
        
    # Prepare Fingerprints Matrix (K, D) where D = H*W
    fp_matrix = []
    for k in fp_keys:
        fp = fingerprints[k].astype(np.float32).ravel()
        # Normalize (zero mean)
        fp -= fp.mean()
        # Unit norm (so dot product == correlation)
        norm = np.linalg.norm(fp)
        if norm > 0:
            fp /= norm
        fp_matrix.append(fp)
    
    fp_matrix = np.array(fp_matrix).T # (D, K)
    
    # Prepare Residuals Matrix (N, D)
    res_matrix = []
    for r in residuals:
        r_flat = r.astype(np.float32).ravel()
        r_flat -= r_flat.mean()
        norm = np.linalg.norm(r_flat)
        if norm > 0:
            r_flat /= norm
        res_matrix.append(r_flat)
    
    res_matrix = np.array(res_matrix)
    
    # GPU Matmul
    with tf.device('/GPU:0' if USE_GPU else '/CPU:0'):
        # A (N, D) . B (D, K) -> (N, K)
        A = tf.convert_to_tensor(res_matrix, dtype=tf.float32)
        B = tf.convert_to_tensor(fp_matrix, dtype=tf.float32)
        C = tf.matmul(A, B)
        
    return C.numpy()

def fft_radial_energy(img, K=6):
    """
    Compute radial energy spectrum from FFT.
    """
    f = fftshift(fft2(img))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h//2, w//2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    rmax = r.max() + 1e-6
    bins = np.linspace(0, rmax, K+1)
    feats = [float(np.mean(mag[(r >= bins[i]) & (r < bins[i+1])])) for i in range(K)]
    return feats

def lbp_hist_safe(img, P=8, R=1.0):
    """
    Compute LBP histogram.
    """
    rng = float(np.ptp(img))
    g = np.zeros_like(img, dtype=np.float32) if rng < 1e-12 else (img - img.min()) / (rng + 1e-8)
    g8 = (g*255).astype(np.uint8)
    codes = sk_lbp(g8, P=P, R=R, method="uniform")
    n_bins = P + 2 
    hist, _ = np.histogram(codes, bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32).tolist()

def extract_enhanced_features(residual):
    """
    FFT + LBP + statistical texture
    """
    # FFT Features
    f = fftshift(fft2(residual))
    mag = np.abs(f)
    h, w = mag.shape
    center_h, center_w = h//2, w//2
    
    # 3 freq bands
    low_freq = np.mean(mag[max(0,center_h-20):center_h+20, max(0,center_w-20):center_w+20])
    mid_region = mag[max(0,center_h-60):center_h+60, max(0,center_w-60):center_w+60]
    mid_freq = np.mean(mid_region) - low_freq 
    high_freq = np.mean(mag) - np.mean(mid_region)

    # LBP
    lbp = sk_lbp(residual, P=24, R=3, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=26, range=(0,26), density=True)

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

    return [float(low_freq), float(mid_freq), float(high_freq)] + lbp_hist.tolist() + texture_features

# ---- GPU Preprocessing Helpers ----

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

def resize_to(img, size=(256, 256)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def normalize_img(img):
    return img.astype(np.float32) / 255.0

def process_batch_gpu(file_paths):
    """
    Process a batch of file paths on GPU using TensorFlow.
    Approximates Haar Wavelet Denoising (Level 1).
    """
    imgs = []
    valid_indices = []
    
    for idx, fpath in enumerate(file_paths):
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if img is None: 
            continue
        img = to_gray(img)
        img = resize_to(img)
        img = normalize_img(img)
        imgs.append(img)
        valid_indices.append(idx)
        
    if not imgs:
        return []

    batch_np = np.array(imgs, dtype=np.float32)
    batch_np = np.expand_dims(batch_np, -1)
    
    with tf.device('/GPU:0' if USE_GPU else '/CPU:0'):
        x = tf.convert_to_tensor(batch_np)
        
        # Haar Approximation (L1)
        pooled = tf.nn.avg_pool2d(x, ksize=2, strides=2, padding='VALID')
        denoised = tf.image.resize(pooled, (256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        residual = x - denoised
        
    res_np = residual.numpy()
    
    results = []
    for i in range(len(res_np)):
        results.append(res_np[i].squeeze())
        
    return results
