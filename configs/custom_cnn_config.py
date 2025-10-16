"""
Configuration for Custom CNN model
Example: Image classification with your own architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ============================================
# CUSTOM MODEL DEFINITION
# ============================================
class CustomCNN(nn.Module):
    """Custom CNN for image classification"""
    def __init__(self, num_classes=10, input_channels=3):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Adjust this based on input size (32x32 → 4x4 after 3 pools)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ============================================
# MODEL CONFIGURATION
# ============================================
MODEL_CLASS = CustomCNN
MODEL_KWARGS = {
    'num_classes': 10,
    'input_channels': 3
}

# ============================================
# TRAINING CONFIGURATION
# ============================================
OPTIMIZER_CLASS = optim.Adam
OPTIMIZER_KWARGS = {
    'lr': 0.001,
    'betas': (0.9, 0.999)
}
CRITERION = nn.CrossEntropyLoss()
EPOCHS_PER_ROUND = 3
BATCH_SIZE = 64

# ============================================
# DATA LOADING
# ============================================
def get_data_loaders(client_id=0):
    """
    Load CIFAR-10 for custom CNN
    You can replace this with your own dataset!
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    
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


