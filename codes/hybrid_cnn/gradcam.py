"""
Grad-CAM Visualization for Hybrid CNN Model
============================================
This script generates Grad-CAM heatmaps to show which image regions
the CNN focuses on when identifying scanner traces.

Grad-CAM (Gradient-weighted Class Activation Mapping):
- Highlights important regions in the input image
- Shows what the model "looks at" to make predictions
- Helps validate that the model is learning scanner-specific patterns

Usage:
    python gradcam.py                    # Visualize sample images
    python gradcam.py --image path.tif   # Visualize specific image
"""

import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2

# ---- Paths ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "hybrid_cnn")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_data", "hybrid_cnn")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "hybrid_cnn")
GRADCAM_DIR = os.path.join(RESULTS_DIR, "gradcam")
os.makedirs(GRADCAM_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "scanner_hybrid.keras")
ENCODER_PATH = os.path.join(MODEL_DIR, "hybrid_label_encoder.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "hybrid_feat_scaler.pkl")
RES_PATH = os.path.join(PROCESSED_DIR, "official_wiki_residuals.pkl")
FEATURES_PATH = os.path.join(PROCESSED_DIR, "features.pkl")
ENHANCED_PATH = os.path.join(PROCESSED_DIR, "enhanced_features.pkl")

# ---- Load Model and Encoders ----
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)


def get_gradcam_heatmap(model, img_array, feat_array, target_class=None, layer_name=None):
    """
    Generate Grad-CAM heatmap for the Hybrid CNN model.
    
    Args:
        model: Trained Keras model
        img_array: Input image array (1, 256, 256, 1)
        feat_array: Feature array (1, num_features)
        target_class: Class index to visualize (None = predicted class)
        layer_name: Name of conv layer to use (None = last conv layer)
    
    Returns:
        heatmap: Grad-CAM heatmap (256, 256)
        pred_class: Predicted class index
        confidence: Prediction confidence
    """
    
    # Find the last convolutional layer if not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            if 'conv2d' in layer.name.lower() and 'hp_filter' not in layer.name:
                layer_name = layer.name
                break
    
    # Create a model that outputs the conv layer output and final predictions
    conv_layer = model.get_layer(layer_name)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model([img_array, feat_array])
        
        if target_class is None:
            target_class = tf.argmax(predictions[0])
        
        class_output = predictions[:, target_class]
    
    # Get gradients of the target class with respect to conv layer output
    grads = tape.gradient(class_output, conv_output)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the conv output by the pooled gradients
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    
    # Resize to input size
    heatmap = cv2.resize(heatmap, (256, 256))
    
    # Get prediction info
    pred_class = int(tf.argmax(predictions[0]))
    confidence = float(predictions[0][pred_class])
    
    return heatmap, pred_class, confidence


def overlay_heatmap(img, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on the original image.
    
    Args:
        img: Original image (256, 256)
        heatmap: Grad-CAM heatmap (256, 256)
        alpha: Transparency of heatmap
    
    Returns:
        overlaid: Combined image with heatmap
    """
    # Normalize image to 0-255
    img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    
    # Convert to RGB
    if len(img_norm.shape) == 2:
        img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_norm
    
    # Apply colormap to heatmap
    heatmap_colored = cm.jet(heatmap)[:, :, :3]  # Remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Overlay
    overlaid = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlaid


def visualize_gradcam(img, feat, true_label, save_path=None):
    """
    Create full Grad-CAM visualization with original image, heatmap, and overlay.
    """
    # Prepare inputs
    img_input = np.expand_dims(np.expand_dims(img, -1), 0).astype(np.float32)
    feat_input = scaler.transform(feat.reshape(1, -1))
    
    # Get Grad-CAM
    heatmap, pred_class, confidence = get_gradcam_heatmap(model, img_input, feat_input)
    
    # Create overlay
    overlay = overlay_heatmap(img, heatmap)
    
    # Get class names
    true_name = true_label if isinstance(true_label, str) else le.classes_[true_label]
    pred_name = le.classes_[pred_class]
    
    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original residual image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'Residual Image\nTrue: {true_name}')
    axes[0].axis('off')
    
    # Grad-CAM heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay\nPred: {pred_name} ({confidence*100:.1f}%)')
    axes[2].axis('off')
    
    # Colorbar reference
    im = axes[3].imshow(heatmap, cmap='jet')
    axes[3].set_title('Attention Intensity')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.suptitle('Grad-CAM: What the Model Focuses On', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    
    return pred_name, confidence


def generate_sample_visualizations(num_samples=3):
    """
    Generate Grad-CAM visualizations for sample images from each scanner class.
    """
    print("\n" + "="*60)
    print("GRAD-CAM VISUALIZATION")
    print("="*60)
    
    # Load data
    print("Loading data...")
    with open(RES_PATH, "rb") as f:
        residuals_dict = pickle.load(f)
    
    with open(FEATURES_PATH, "rb") as f:
        d_feat = pickle.load(f)
        feats_prnu = d_feat["features"]
    with open(ENHANCED_PATH, "rb") as f:
        d_enh = pickle.load(f)
        feats_enh = d_enh["features"]
    
    # Collect samples per scanner
    samples_by_scanner = {}
    idx = 0
    
    for dataset_name in ["Official", "WikiPedia"]:
        if dataset_name not in residuals_dict:
            continue
        for scanner, dpi_dict in residuals_dict[dataset_name].items():
            if scanner not in samples_by_scanner:
                samples_by_scanner[scanner] = []
            if isinstance(dpi_dict, dict):
                for dpi, res_list in dpi_dict.items():
                    for res in res_list:
                        feat = np.array(feats_prnu[idx] + feats_enh[idx])
                        samples_by_scanner[scanner].append((res, feat))
                        idx += 1
    
    # Generate visualizations for each scanner
    print(f"\nGenerating Grad-CAM for {len(samples_by_scanner)} scanner classes...")
    
    for scanner, samples in samples_by_scanner.items():
        if len(samples) == 0:
            continue
            
        # Take first N samples
        for i, (img, feat) in enumerate(samples[:num_samples]):
            save_path = os.path.join(GRADCAM_DIR, f"gradcam_{scanner}_sample{i+1}.png")
            pred_name, conf = visualize_gradcam(img, feat, scanner, save_path)
            
            status = "✅" if pred_name == scanner else "❌"
            print(f"  {status} {scanner} sample {i+1}: Pred={pred_name} ({conf*100:.1f}%)")
    
    # Create summary grid
    create_summary_grid(samples_by_scanner)
    
    print(f"\n✅ Grad-CAM visualizations saved to: {GRADCAM_DIR}")


def create_summary_grid(samples_by_scanner):
    """
    Create a summary grid showing one Grad-CAM example per scanner.
    """
    scanners = sorted(samples_by_scanner.keys())
    n_scanners = len(scanners)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, scanner in enumerate(scanners):
        if idx >= 11:  # Max 11 scanners
            break
            
        ax = axes[idx]
        samples = samples_by_scanner[scanner]
        if len(samples) == 0:
            continue
            
        img, feat = samples[0]
        
        # Get Grad-CAM
        img_input = np.expand_dims(np.expand_dims(img, -1), 0).astype(np.float32)
        feat_input = scaler.transform(feat.reshape(1, -1))
        heatmap, pred_class, conf = get_gradcam_heatmap(model, img_input, feat_input)
        overlay = overlay_heatmap(img, heatmap)
        
        ax.imshow(overlay)
        pred_name = le.classes_[pred_class]
        status = "✓" if pred_name == scanner else "✗"
        ax.set_title(f'{scanner}\n{status} {conf*100:.0f}%', fontsize=9)
        ax.axis('off')
    
    # Hide unused axes
    for idx in range(n_scanners, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Grad-CAM Summary: Scanner Pattern Focus Areas', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    summary_path = os.path.join(GRADCAM_DIR, "gradcam_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Summary grid saved: {summary_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    parser.add_argument("--samples", type=int, default=2, help="Samples per scanner")
    args = parser.parse_args()
    
    generate_sample_visualizations(num_samples=args.samples)
