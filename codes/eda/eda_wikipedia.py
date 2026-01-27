"""
EDA (Exploratory Data Analysis) for Wikipedia Dataset
======================================================
This script analyzes the Wikipedia scanner dataset to understand:
1. Class distribution (how many images per scanner)
2. Sample images from each class
3. Image statistics (mean, std of pixel values)
"""

import os
import sys

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..", "..")
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

# Configuration
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "WikiPedia")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "eda", "wikipedia")
BATCH_SIZE = 32

# Transform: Resize images to uniform size and convert to tensor
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def main():
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("EDA - Wikipedia Dataset Analysis")
    print("=" * 60)
    
    # Load dataset
    dataset = ImageFolder(root=DATA_ROOT, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    classes = dataset.classes
    print(f"\n📂 Dataset Location: {DATA_ROOT}")
    print(f"📊 Number of Classes (Scanners): {len(classes)}")
    print(f"🖼️  Total Images: {len(dataset)}")
    print(f"\n📋 Scanner Classes:")
    for i, cls in enumerate(classes, 1):
        print(f"   {i}. {cls}")
    
    # =========================================================================
    # 1. CLASS DISTRIBUTION ANALYSIS
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. Analyzing Class Distribution...")
    print("=" * 60)
    
    class_counts = {cls: 0 for cls in classes}
    for imgs, labels in dataloader:
        for lab in labels:
            class_counts[classes[int(lab)]] += 1
    
    # Print class counts
    print("\n📊 Images per Scanner:")
    for cls, count in class_counts.items():
        print(f"   {cls}: {count} images")
    
    # Check for class imbalance
    counts = list(class_counts.values())
    max_count, min_count = max(counts), min(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\n⚖️  Class Balance Analysis:")
    print(f"   Max images in a class: {max_count}")
    print(f"   Min images in a class: {min_count}")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}x")
    
    if imbalance_ratio > 2:
        print("   ⚠️  WARNING: Dataset is imbalanced! Consider data augmentation.")
    else:
        print("   ✅ Dataset is relatively balanced.")
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("husl", len(classes))
    bars = plt.bar(list(class_counts.keys()), list(class_counts.values()), color=colors)
    
    # Add value labels on bars
    for bar, count in zip(bars, class_counts.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                 str(count), ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.ylabel("Number of Images", fontsize=12)
    plt.xlabel("Scanner Class", fontsize=12)
    plt.title("Wikipedia Dataset - Class Distribution", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, "class_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n💾 Saved: {save_path}")
    
    # =========================================================================
    # 2. SAMPLE IMAGES VISUALIZATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. Creating Sample Images Grid...")
    print("=" * 60)
    
    # Get a batch of images
    imgs, labels = next(iter(dataloader))
    n_show = min(16, imgs.size(0))
    rows, cols = 4, 4
    
    plt.figure(figsize=(10, 10))
    for i in range(n_show):
        plt.subplot(rows, cols, i + 1)
        img = imgs[i].permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
        plt.imshow(img)
        plt.axis("off")
        plt.title(classes[int(labels[i])], fontsize=9)
    
    plt.suptitle("Sample Images from Wikipedia Dataset", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, "sample_images.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: {save_path}")
    
    # =========================================================================
    # 3. ONE SAMPLE PER CLASS
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. Creating One Sample Per Class Grid...")
    print("=" * 60)
    
    # Get one sample from each class
    class_samples = {}
    for img, label in dataset:
        cls_name = classes[label]
        if cls_name not in class_samples:
            class_samples[cls_name] = img
        if len(class_samples) == len(classes):
            break
    
    # Plot one sample per class
    n_classes = len(classes)
    cols = 4
    rows = (n_classes + cols - 1) // cols
    
    plt.figure(figsize=(12, rows * 3))
    for i, (cls_name, img) in enumerate(class_samples.items()):
        plt.subplot(rows, cols, i + 1)
        img_np = img.permute(1, 2, 0).numpy()
        plt.imshow(img_np)
        plt.axis("off")
        plt.title(cls_name, fontsize=10, fontweight='bold')
    
    plt.suptitle("One Sample Per Scanner Class (Wikipedia)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, "samples_per_class.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: {save_path}")
    
    # =========================================================================
    # 4. IMAGE STATISTICS
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. Computing Image Statistics...")
    print("=" * 60)
    
    # Calculate mean and std across all images
    all_means = []
    all_stds = []
    
    for imgs, _ in dataloader:
        # Calculate per-channel mean and std for each batch
        batch_mean = imgs.mean(dim=[0, 2, 3])  # Mean across batch, height, width
        batch_std = imgs.std(dim=[0, 2, 3])
        all_means.append(batch_mean.numpy())
        all_stds.append(batch_std.numpy())
    
    overall_mean = np.mean(all_means, axis=0)
    overall_std = np.mean(all_stds, axis=0)
    
    print(f"\n📈 Image Statistics (RGB channels):")
    print(f"   Mean: R={overall_mean[0]:.4f}, G={overall_mean[1]:.4f}, B={overall_mean[2]:.4f}")
    print(f"   Std:  R={overall_std[0]:.4f}, G={overall_std[1]:.4f}, B={overall_std[2]:.4f}")
    
    # =========================================================================
    # 5. SAVE SUMMARY REPORT
    # =========================================================================
    print("\n" + "=" * 60)
    print("5. Saving EDA Summary Report...")
    print("=" * 60)
    
    report_path = os.path.join(RESULTS_DIR, "eda_summary.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EDA SUMMARY - Wikipedia Dataset\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Dataset Location: {DATA_ROOT}\n")
        f.write(f"Number of Classes: {len(classes)}\n")
        f.write(f"Total Images: {len(dataset)}\n\n")
        
        f.write("Class Distribution:\n")
        f.write("-" * 40 + "\n")
        for cls, count in class_counts.items():
            f.write(f"  {cls}: {count} images\n")
        
        f.write(f"\nImbalance Ratio: {imbalance_ratio:.2f}x\n")
        
        f.write(f"\nImage Statistics (RGB):\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Mean: [{overall_mean[0]:.4f}, {overall_mean[1]:.4f}, {overall_mean[2]:.4f}]\n")
        f.write(f"  Std:  [{overall_std[0]:.4f}, {overall_std[1]:.4f}, {overall_std[2]:.4f}]\n")
    
    print(f"💾 Saved: {report_path}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("✅ EDA COMPLETE!")
    print("=" * 60)
    print(f"\n📁 All results saved in: {RESULTS_DIR}")
    print("\nGenerated files:")
    print("  1. class_distribution.png - Bar chart of images per class")
    print("  2. sample_images.png - Grid of 16 random samples")
    print("  3. samples_per_class.png - One sample from each scanner")
    print("  4. eda_summary.txt - Text summary of analysis")


if __name__ == "__main__":
    main()
