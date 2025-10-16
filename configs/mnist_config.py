"""
Configuration for MNIST with simple Net model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Import your model (adjust path if needed)
import sys
sys.path.append('..')
from model import Net

# ============================================
# MODEL CONFIGURATION
# ============================================
MODEL_CLASS = Net
MODEL_KWARGS = {
    'input_size': 28*28,
    'hidden_size': 128,
    'num_classes': 10
}

# ============================================
# TRAINING CONFIGURATION
# ============================================
OPTIMIZER_CLASS = optim.SGD
OPTIMIZER_KWARGS = {
    'lr': 0.01,
    'momentum': 0.9
}
CRITERION = nn.CrossEntropyLoss()
EPOCHS_PER_ROUND = 3
BATCH_SIZE = 32

# ============================================
# DATA LOADING
# ============================================
def get_data_loaders(client_id=0):
    """
    Load MNIST data for a specific client
    
    Args:
        client_id: 0 or 1 (splits data in half)
    """
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


