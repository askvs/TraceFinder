import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split

def get_dataloaders(data_root=None, batch_size=32, img_size=128, val_split=0.1, test_split=0.1):
    # Auto-resolve data path relative to this script's location
    if data_root is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_root = os.path.normpath(os.path.join(script_dir, "../../data"))
    """
    Creates DataLoaders for train, validation, and test sets.
    Combines 'official' and 'Wikipedia' datasets.
    
    VERSION 4: NO AUGMENTATION (like V1 that worked)
    - V1: No augmentation → 60% accuracy ✅
    - V2: Heavy augmentation → 11% accuracy ❌
    - V3: Mild augmentation → 10% accuracy ❌
    - V4: No augmentation + BatchNorm → Target: 65-75% ✅
    """
    
    # =====================================================
    # NO AUGMENTATION (BACK TO BASICS)
    # =====================================================
    # V1 with no augmentation worked (60%)
    # V2/V3 with augmentation failed (.% 10%)
    # Let's use V1 approach + BatchNorm in model
    # =====================================================
    
    # SIMPLE transforms - NO AUGMENTATION
    # Same for training, validation, and test
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize to 128x128
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    
    print("✅ Using NO augmentation (V1 approach + BatchNorm model)")
    
    datasets_list = []
    
    # 1. Official Dataset
    official_dir = os.path.join(data_root, "Official")
    if os.path.exists(official_dir):
        print(f"Found official dataset at {official_dir}")
        ds_official = datasets.ImageFolder(root=official_dir, transform=transform)
        datasets_list.append(ds_official)
    else:
        print(f"Warning: Official dataset not found at {official_dir}")

    # 2. Wikipedia Dataset
    wiki_dir = os.path.join(data_root, "WikiPedia")
    if os.path.exists(wiki_dir):
        print(f"Found Wikipedia dataset at {wiki_dir}")
        ds_wiki = datasets.ImageFolder(root=wiki_dir, transform=transform)
        datasets_list.append(ds_wiki)
    else:
        print(f"Warning: Wikipedia dataset not found at {wiki_dir}")

    if not datasets_list:
        raise ValueError("No datasets found! Please check Data_Set structure.")
    
    # Combine datasets
    full_dataset = ConcatDataset(datasets_list)
    
    # Split into train/val/test
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size
    
    # Create consistent splits using the same seed
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Total images: {total_size}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create DataLoaders
    # num_workers=0 for Windows compatibility (avoids multiprocessing issues)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    classes = datasets_list[0].classes
    
    return train_loader, val_loader, test_loader, classes
