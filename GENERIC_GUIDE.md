# 🌐 Generic Federated Learning System

## Overview

This is a **completely generic** federated learning system that works with:
- ✅ **ANY PyTorch model** (CNN, ResNet, Transformer, custom architectures)
- ✅ **ANY dataset** (MNIST, CIFAR, ImageNet, medical images, custom data)
- ✅ **ANY optimizer** (SGD, Adam, AdamW, custom optimizers)
- ✅ **Flexible blending** (server + client collaboration)

**No more hardcoded models or datasets!** 🎉

---

## 📁 File Structure

```
configs/
├── mnist_config.py          # MNIST with simple Net
├── cifar10_config.py        # CIFAR-10 with ResNet18
├── custom_cnn_config.py     # Custom CNN architecture
└── transformer_config.py    # Vision Transformer (ViT)

server_generic.py            # Generic server (works with any model)
client_generic.py            # Generic client (works with any model)
```

---

## 🚀 Quick Start

### Example 1: MNIST with Simple Net

**Terminal 1 - Server:**
```bash
python server_generic.py \
    --config configs/mnist_config.py \
    --num-rounds 5 \
    --alpha 0.5 \
    --min-clients 1
```

**Terminal 2 - Client:**
```bash
python client_generic.py \
    --config configs/mnist_config.py \
    --server-address 127.0.0.1:8080 \
    --client-id 0
```

### Example 2: CIFAR-10 with ResNet18

**Server:**
```bash
python server_generic.py \
    --config configs/cifar10_config.py \
    --num-rounds 10 \
    --alpha 0.7
```

**Client:**
```bash
python client_generic.py \
    --config configs/cifar10_config.py \
    --server-address 127.0.0.1:8080
```

### Example 3: Custom CNN

**Server:**
```bash
python server_generic.py \
    --config configs/custom_cnn_config.py \
    --num-rounds 8 \
    --alpha 0.5
```

**Client:**
```bash
python client_generic.py \
    --config configs/custom_cnn_config.py \
    --server-address 127.0.0.1:8080
```

---

## 🔧 Creating Your Own Configuration

### Step 1: Create Config File

Create `configs/my_model_config.py`:

```python
"""
My Custom Model Configuration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ============================================
# 1. DEFINE YOUR MODEL
# ============================================
class MyCustomModel(nn.Module):
    def __init__(self, input_dim=784, num_classes=10):
        super(MyCustomModel, self).__init__()
        # Your architecture here
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# ============================================
# 2. MODEL CONFIGURATION
# ============================================
MODEL_CLASS = MyCustomModel
MODEL_KWARGS = {
    'input_dim': 784,
    'num_classes': 10
}

# ============================================
# 3. TRAINING CONFIGURATION
# ============================================
OPTIMIZER_CLASS = optim.Adam
OPTIMIZER_KWARGS = {
    'lr': 0.001,
    'weight_decay': 1e-5
}
CRITERION = nn.CrossEntropyLoss()
EPOCHS_PER_ROUND = 5
BATCH_SIZE = 64

# ============================================
# 4. DATA LOADING FUNCTION
# ============================================
def get_data_loaders(client_id=0):
    """
    Load data for your specific use case
    
    Args:
        client_id: Identifier for this client (0, 1, 2, ...)
    
    Returns:
        trainloader, testloader
    """
    # Your data loading logic
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load your dataset
    trainset = datasets.MNIST("./data", train=True, 
                             download=True, transform=transform)
    testset = datasets.MNIST("./data", train=False, 
                            download=True, transform=transform)
    
    # Split data between clients (customize this!)
    total_clients = 2
    samples_per_client = len(trainset) // total_clients
    start_idx = client_id * samples_per_client
    end_idx = start_idx + samples_per_client
    
    train_indices = list(range(start_idx, end_idx))
    test_indices = list(range(start_idx // 5, end_idx // 5))
    
    train_subset = Subset(trainset, train_indices)
    test_subset = Subset(testset, test_indices)
    
    trainloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    return trainloader, testloader
```

### Step 2: Run with Your Config

```bash
# Server
python server_generic.py --config configs/my_model_config.py --num-rounds 10

# Client
python client_generic.py --config configs/my_model_config.py
```

**That's it!** No need to modify `server_generic.py` or `client_generic.py`! 🎉

---

## 🎨 Real-World Examples

### Medical Imaging (X-ray Classification)

```python
# configs/xray_config.py

class XRayClassifier(nn.Module):
    def __init__(self, num_diseases=5):
        super().__init__()
        # Use pretrained ResNet50
        from torchvision.models import resnet50
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(2048, num_diseases)
    
    def forward(self, x):
        return self.backbone(x)

MODEL_CLASS = XRayClassifier
MODEL_KWARGS = {'num_diseases': 5}

def get_data_loaders(client_id):
    # Load hospital's private X-ray data
    # Each hospital (client) has different patients
    from torchvision.datasets import ImageFolder
    
    hospital_data_path = f"/hospital_{client_id}/xrays"
    dataset = ImageFolder(hospital_data_path, transform=your_transform)
    
    # Split train/test
    # ... return trainloader, testloader
```

### Text Classification (Sentiment Analysis)

```python
# configs/sentiment_config.py

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

MODEL_CLASS = SentimentClassifier
MODEL_KWARGS = {
    'vocab_size': 10000,
    'embedding_dim': 128,
    'num_classes': 2  # Positive/Negative
}

def get_data_loaders(client_id):
    # Load company's private customer reviews
    # Each company has different review data
    # ... your text loading logic
```

---

## 🔍 How It Works Internally

### Server Side (`server_generic.py`)

```python
# 1. Load model class from config
model_class, model_kwargs = load_model_from_config("configs/my_config.py")

# 2. Create strategy with ANY model
strategy = GenericBlendStrategy(
    model_class=model_class,      # ← Works with ANY nn.Module
    model_kwargs=model_kwargs,    # ← Custom parameters
    alpha=0.5
)

# 3. During aggregation
def aggregate_fit(self, ...):
    # Blend ANY model's weights
    for w_server, w_client in zip(server_weights, client_weights):
        blended = (1-α) * w_server + α * w_client
    
    # Save ANY model
    model = self.model_class(**self.model_kwargs)  # ← Generic!
    model.load_state_dict(blended_weights)
    torch.save(model.state_dict(), 'model.pth')
```

### Client Side (`client_generic.py`)

```python
# 1. Load config
config = load_client_config("configs/my_config.py")

# 2. Create client with ANY model
client = GenericClient(
    model_class=config.MODEL_CLASS,     # ← ANY model
    model_kwargs=config.MODEL_KWARGS,
    train_loader=your_data,
    optimizer_class=config.OPTIMIZER_CLASS  # ← ANY optimizer
)

# 3. Training works with ANY model
def fit(self, parameters):
    self.set_parameters(parameters)  # ← Works with any architecture
    
    for data, target in self.trainloader:
        output = self.model(data)  # ← Your custom forward pass
        loss = self.criterion(output, target)
        loss.backward()
```

**Key insight:** Everything is passed as **parameters**, not hardcoded! 🎯

---

## 📊 Supported Model Types

| Model Type | Config File | Example |
|------------|-------------|---------|
| **Simple NN** | `mnist_config.py` | 3-layer fully connected |
| **CNN** | `custom_cnn_config.py` | Custom convolutional network |
| **ResNet** | `cifar10_config.py` | ResNet18/34/50 |
| **Transformer** | `transformer_config.py` | Vision Transformer (ViT) |
| **LSTM/RNN** | Create your own | Text/sequence models |
| **GAN** | Create your own | Generative models |
| **Autoencoder** | Create your own | Unsupervised learning |

**Any `torch.nn.Module` works!** ✅

---

## 🌐 Deployment

### Deploy Server (Cloud)

```bash
# On Railway/AWS/GCP
python server_generic.py \
    --config configs/your_config.py \
    --server-address 0.0.0.0:8080 \
    --num-rounds 20 \
    --alpha 0.5
```

### Run Client (Local)

```bash
# On client's machine
python client_generic.py \
    --config configs/your_config.py \
    --server-address your-server.railway.app:8080 \
    --client-id 0
```

**Config file must be available on both sides!**

---

## 💡 Advanced Usage

### Use Pretrained Baseline

```bash
# Server with pretrained model
python server_generic.py \
    --config configs/cifar10_config.py \
    --baseline-path pretrained_resnet18.pth \
    --alpha 0.3  # Trust baseline more
```

### Multiple Clients

```bash
# Terminal 1 - Server (waits for 2 clients)
python server_generic.py \
    --config configs/mnist_config.py \
    --min-clients 2

# Terminal 2 - Client 0
python client_generic.py \
    --config configs/mnist_config.py \
    --client-id 0

# Terminal 3 - Client 1
python client_generic.py \
    --config configs/mnist_config.py \
    --client-id 1
```

### Different Optimizers

In your config:
```python
# SGD with momentum
OPTIMIZER_CLASS = optim.SGD
OPTIMIZER_KWARGS = {'lr': 0.01, 'momentum': 0.9, 'nesterov': True}

# Adam
OPTIMIZER_CLASS = optim.Adam
OPTIMIZER_KWARGS = {'lr': 0.001, 'betas': (0.9, 0.999)}

# AdamW (better for transformers)
OPTIMIZER_CLASS = optim.AdamW
OPTIMIZER_KWARGS = {'lr': 0.0001, 'weight_decay': 0.05}
```

---

## 🎯 Summary

### Old System (Hardcoded)
```python
# server_blend.py
from model import Net  # ← HARDCODED
model = Net()          # ← HARDCODED
```
❌ Only works with Net model  
❌ Only works with MNIST  
❌ Must edit code for new models

### New System (Generic)
```python
# server_generic.py
model_class, kwargs = load_from_config(args.config)
model = model_class(**kwargs)  # ← ANY MODEL!
```
✅ Works with ANY PyTorch model  
✅ Works with ANY dataset  
✅ Just create a new config file  
✅ No code changes needed!

---

## 🚀 Get Started

1. **Choose a config** from `configs/` or create your own
2. **Start server:** `python server_generic.py --config configs/yourconfig.py`
3. **Start client:** `python client_generic.py --config configs/yourconfig.py`
4. **Watch it train!** Models saved to `./models/`

**That's it! Federated learning with ANY model!** 🎉


