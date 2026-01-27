from torchvision import datasets
import os

# JUST FOR THE CHECKING THE no of Images in each sets

# Auto-resolve paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.normpath(os.path.join(script_dir, "../../data"))

# Test Wikipedia dataset
print("=== WIKIPEDIA DATASET ===")
wiki_path = os.path.join(data_dir, "WikiPedia")
print(f"Loading from: {wiki_path}")
ds_wiki = datasets.ImageFolder(wiki_path)
print(f"Total images: {len(ds_wiki)}")
print(f"Classes: {ds_wiki.classes}")
print(f"\nSample image paths:")
for i in range(min(10, len(ds_wiki))):
    path, label = ds_wiki.imgs[i]
    print(f"  {ds_wiki.classes[label]}: {path}")
print(f"\nTotal images in Wikipedia dataset: {len(ds_wiki)}")

print("\n=== OFFICIAL DATASET ===")
official_path = os.path.join(data_dir, "Official")
print(f"Loading from: {official_path}")
ds_official = datasets.ImageFolder(official_path)
print(f"Total images: {len(ds_official)}")
print(f"Classes: {ds_official.classes}")
print(f"\nSample image paths:")
for i in range(min(10, len(ds_official))):
    path, label = ds_official.imgs[i]
    print(f"  {ds_official.classes[label]}: {path}")



# Count images per class
print("\n=== IMAGES PER CLASS ===")
print("\nWikipedia:")
for cls_name in ds_wiki.classes:
    count = sum(1 for _, label in ds_wiki.imgs if ds_wiki.classes[label] == cls_name)
    print(f"  {cls_name}: {count}")

print("\nOfficial:")
for cls_name in ds_official.classes:
    count = sum(1 for _, label in ds_official.imgs if ds_official.classes[label] == cls_name)
    print(f"  {cls_name}: {count}")
