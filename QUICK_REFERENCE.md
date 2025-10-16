# 🚀 Quick Reference: Generic Federated Learning

## One-Line Commands

### MNIST (Simple Net)
```bash
python run_generic_example.py --config configs/mnist_config.py
```

### CIFAR-10 (ResNet18)
```bash
python run_generic_example.py --config configs/cifar10_config.py --num-rounds 10
```

### Custom CNN
```bash
python run_generic_example.py --config configs/custom_cnn_config.py --alpha 0.7
```

### Multiple Clients
```bash
python run_generic_example.py --config configs/mnist_config.py --num-clients 2
```

---

## Manual Commands

### Start Server
```bash
python server_generic.py --config configs/YOURCONFIG.py --num-rounds 5 --alpha 0.5
```

### Start Client
```bash
python client_generic.py --config configs/YOURCONFIG.py --server-address 127.0.0.1:8080
```

---

## Create New Config in 3 Steps

### 1. Create File
```bash
touch configs/my_model.py
```

### 2. Add This Template
```python
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Your model
class MyModel(nn.Module):
    def __init__(self, ...):
        # Your architecture

MODEL_CLASS = MyModel
MODEL_KWARGS = {...}
OPTIMIZER_CLASS = optim.Adam
OPTIMIZER_KWARGS = {'lr': 0.001}

def get_data_loaders(client_id):
    # Your data loading
    return trainloader, testloader
```

### 3. Run It
```bash
python run_generic_example.py --config configs/my_model.py
```

---

## Common Parameters

| Parameter | What It Does | Example |
|-----------|--------------|---------|
| `--config` | Config file path | `configs/mnist_config.py` |
| `--num-rounds` | Training rounds | `10` |
| `--alpha` | Blending weight | `0.5` (50/50 blend) |
| `--num-clients` | Number of clients | `2` |
| `--min-clients` | Min clients to start | `1` |
| `--server-address` | Server URL | `127.0.0.1:8080` |
| `--baseline-path` | Pretrained model | `baseline.pth` |

---

## Alpha (Blending Weight) Guide

```
α = 0.1  →  90% server, 10% client  (conservative)
α = 0.3  →  70% server, 30% client  (cautious)
α = 0.5  →  50% server, 50% client  (balanced) ← DEFAULT
α = 0.7  →  30% server, 70% client  (aggressive)
α = 1.0  →   0% server, 100% client (pure FedAvg)
```

---

## File Outputs

After running, you'll get:
```
models/
├── YourModel_round_1.pth
├── YourModel_round_2.pth
├── YourModel_round_3.pth
└── YourModel_final.pth       ← Use this one!
```

---

## Supported Models

✅ Any `torch.nn.Module`  
✅ torchvision models (ResNet, VGG, etc.)  
✅ Hugging Face models  
✅ timm models (Vision Transformers)  
✅ Your custom architectures

---

## Supported Datasets

✅ torchvision datasets (MNIST, CIFAR, ImageNet)  
✅ Custom ImageFolder  
✅ CSV/JSON datasets  
✅ Medical imaging (DICOM, NIfTI)  
✅ Text datasets  
✅ Audio datasets  
✅ Your custom data

---

## Troubleshooting

### "Module not found"
Make sure config file imports work:
```python
# Add parent directory to path if needed
import sys
sys.path.append('..')
from your_module import YourModel
```

### "Connection refused"
Start server first, wait 10 seconds, then start client.

### "Shape mismatch"
Check that `MODEL_KWARGS` match your model's `__init__` parameters.

### "CUDA out of memory"
Reduce `BATCH_SIZE` in your config file.

---

## Migration from Old System

### Old (Hardcoded)
```bash
python server_blend.py  # Only works with Net model
python client_single.py
```

### New (Generic)
```bash
python server_generic.py --config configs/mnist_config.py
python client_generic.py --config configs/mnist_config.py
```

**Same functionality, but now works with ANY model!**

---

## Examples by Use Case

### Hospital Data (Medical Imaging)
```python
# configs/xray_config.py
MODEL_CLASS = YourMedicalCNN
def get_data_loaders(client_id):
    # Load hospital_id's private patient data
    # Data never leaves hospital!
```

### Mobile App (Personalization)
```python
# configs/keyboard_config.py
MODEL_CLASS = TextPredictionModel
def get_data_loaders(client_id):
    # Load user_id's typing patterns
    # Data stays on phone!
```

### IoT Devices (Sensor Data)
```python
# configs/sensor_config.py
MODEL_CLASS = AnomalyDetectionModel
def get_data_loaders(client_id):
    # Load sensor_id's readings
    # Data stays on device!
```

---

## Next Steps

1. ✅ Try example: `python run_generic_example.py`
2. ✅ Create your config based on templates in `configs/`
3. ✅ Test locally with `--num-clients 1`
4. ✅ Deploy server to cloud
5. ✅ Run clients on edge devices

**You now have a production-ready generic FL system!** 🎉


