"""
Configuration for CIFAR-10 with ResNet18
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18

# ============================================
# MODEL CONFIGURATION
# ============================================
MODEL_CLASS = resnet18
MODEL_KWARGS = {
    'num_classes': 10,  # CIFAR-10 has 10 classes
    'pretrained': False
}

# ============================================
# TRAINING CONFIGURATION
# ============================================
OPTIMIZER_CLASS = optim.Adam
OPTIMIZER_KWARGS = {
    'lr': 0.001,
    'weight_decay': 1e-4
}
CRITERION = nn.CrossEntropyLoss()
EPOCHS_PER_ROUND = 5
BATCH_SIZE = 64

# ============================================
# DATA LOADING
# ============================================
def get_data_loaders(client_id=0):
    """
    Load CIFAR-10 data for a specific client
    
    Args:
        client_id: 0 or 1 (splits data in half)
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = datasets.CIFAR10("./data", train=True, download=True, 
                                transform=transform_train)
    testset = datasets.CIFAR10("./data", train=False, download=True, 
                               transform=transform_test)
    
    # Split data between clients
    if client_id == 0:
        train_indices = list(range(0, len(trainset) // 2))
        test_indices = list(range(0, len(testset) // 2))
    else:
        train_indices = list(range(len(trainset) // 2, len(trainset)))
        test_indices = list(range(len(testset) // 2, len(testset)))
    
    train_subset = Subset(trainset, train_indices)
    test_subset = Subset(testset, test_indices)
    
    trainloader = DataLoader(train_subset, batch_size=BATCH_SIZE, 
                           shuffle=True, num_workers=2)
    testloader = DataLoader(test_subset, batch_size=BATCH_SIZE, 
                          shuffle=False, num_workers=2)
    
    return trainloader, testloader


