"""
Configuration for Vision Transformer (ViT)
Example: Using transformers for image classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Try to import timm (PyTorch Image Models)
try:
    import timm
    MODEL_CLASS = lambda **kwargs: timm.create_model('vit_tiny_patch16_224', **kwargs)
    HAS_TIMM = True
except ImportError:
    print("⚠️  timm not installed. Install with: pip install timm")
    print("Using simple model as fallback...")
    
    # Fallback to simple model
    import sys
    sys.path.append('..')
    from model import Net
    MODEL_CLASS = Net
    HAS_TIMM = False

# ============================================
# MODEL CONFIGURATION
# ============================================
if HAS_TIMM:
    MODEL_KWARGS = {
        'pretrained': False,
        'num_classes': 10
    }
else:
    MODEL_KWARGS = {
        'input_size': 28*28,
        'hidden_size': 128,
        'num_classes': 10
    }

# ============================================
# TRAINING CONFIGURATION
# ============================================
OPTIMIZER_CLASS = optim.AdamW
OPTIMIZER_KWARGS = {
    'lr': 0.0001,
    'weight_decay': 0.05
}
CRITERION = nn.CrossEntropyLoss()
EPOCHS_PER_ROUND = 2  # Transformers need less epochs
BATCH_SIZE = 32

# ============================================
# DATA LOADING
# ============================================
def get_data_loaders(client_id=0):
    """Load data for transformer model"""
    
    if HAS_TIMM:
        # ViT expects 224x224 images
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])
        ])
        
        trainset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        testset = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    else:
        # Fallback to MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.MNIST("./data", train=True, download=True, transform=transform)
        testset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    
    # Split data
    if client_id == 0:
        train_indices = list(range(0, len(trainset) // 2))
        test_indices = list(range(0, len(testset) // 2))
    else:
        train_indices = list(range(len(trainset) // 2, len(trainset)))
        test_indices = list(range(len(testset) // 2, len(testset)))
    
    train_subset = Subset(trainset, train_indices)
    test_subset = Subset(testset, test_indices)
    
    trainloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    return trainloader, testloader


