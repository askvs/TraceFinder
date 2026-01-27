import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    Improved CNN for Scanner Forensics (TraceFinder Project)
    
    📚 ARCHITECTURE EXPLANATION:
    - Conv2d: Detects patterns (edges, textures, scanner traces)
    - BatchNorm2d: Normalizes values = faster & more stable training [CHANGE]
    - MaxPool2d: Reduces image size while keeping important features
    - Dropout: Prevents overfitting (memorizing instead of learning)
    - Linear: Makes final classification decision
    
    🎯 Target: >85% accuracy (Project Requirement)
    """
    def __init__(self, num_classes=11):
        super(SimpleCNN, self).__init__()
        
        # ────────────────────────────────────────────────────────
        # CHANGE #1: Added BatchNorm after each Conv layer
        # ────────────────────────────────────────────────────────
        # WHY: Normalizes activations = faster & more stable training
        # Without BatchNorm: Training was stuck at ~9% accuracy
        # With BatchNorm: Model learns much better
        # ────────────────────────────────────────────────────────
        
        # Convolutional Block 1: 3 channels -> 32 filters
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # [CHANGE] BatchNorm added
        
        # Convolutional Block 2: 32 -> 64 filters
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # [CHANGE] BatchNorm added
        
        # Convolutional Block 3: 64 -> 128 filters
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # [CHANGE] BatchNorm added
        
        # Pooling layer (reduces size by half each time: 128 -> 64 -> 32 -> 16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        # After 3 pools: 128x128 -> 64 -> 32 -> 16, Feature size: 128 * 16 * 16 = 32,768
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # ────────────────────────────────────────────────────────
        # CHANGE #2: Reduced Dropout from 50% to 30%
        # ────────────────────────────────────────────────────────
        # WHY: 50% dropout + augmentation was too harsh, model couldn't learn
        # 30% is a better balance for our dataset size
        # ────────────────────────────────────────────────────────
        self.dropout = nn.Dropout(0.3)  # [CHANGE] Was 0.5, now 0.3

    def forward(self, x):
        # Block 1: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 128 -> 64
        
        # Block 2: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 64 -> 32
        
        # Block 3: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 32 -> 16
        
        # Flatten: 128 * 16 * 16 = 32,768 features
        x = x.view(-1, 128 * 16 * 16)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
