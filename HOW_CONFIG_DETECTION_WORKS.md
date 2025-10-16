# 🔍 How Config Detection Works

## 📋 **Short Answer**

The system reads the config file path from the command line (`--config`), then **dynamically imports** it as a Python module and reads attributes like `MODEL_CLASS`, `MODEL_KWARGS`, etc.

**It's like `import your_config` but done at runtime!**

---

## 🎬 **Step-by-Step Flow**

### **Step 1: You Run the Command**

```bash
python server_generic.py --config configs/resnet_config.py --num-rounds 5
```

### **Step 2: Argparse Reads the Path**

```python
# In server_generic.py, line ~175
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

# args.config = "configs/resnet_config.py"
```

### **Step 3: Dynamic Import of Config File**

```python
# In server_generic.py, line ~188
model_class, model_kwargs = load_model_from_config(args.config)
```

**What `load_model_from_config` does:**

```python
def load_model_from_config(config_path):
    # config_path = "configs/resnet_config.py"
    
    import sys
    from pathlib import Path
    import importlib
    
    config_path = Path(config_path)  # Convert to Path object
    # config_path = Path("configs/resnet_config.py")
    
    # Add "configs/" directory to Python's search path
    sys.path.insert(0, str(config_path.parent))
    # Now Python can find modules in "configs/" directory
    
    # Get the module name (filename without .py)
    module_name = config_path.stem  # "resnet_config"
    
    # Dynamically import the module (like "import resnet_config")
    config = importlib.import_module(module_name)
    # This executes the config file and creates a module object
    
    # Read attributes from the module
    model_class = getattr(config, 'MODEL_CLASS')
    # If config has: MODEL_CLASS = ResNet50
    # Then model_class = ResNet50
    
    model_kwargs = getattr(config, 'MODEL_KWARGS', {})
    # If config has: MODEL_KWARGS = {'num_classes': 10}
    # Then model_kwargs = {'num_classes': 10}
    
    return model_class, model_kwargs
```

### **Step 4: Use the Loaded Config**

```python
# Now the server has:
model_class = ResNet50  # The actual class object
model_kwargs = {'num_classes': 10}

# Create strategy
strategy = GenericBlendStrategy(
    model_class=model_class,      # Pass ResNet50 class
    model_kwargs=model_kwargs,    # Pass {'num_classes': 10}
    alpha=0.5
)

# Later, when saving model:
model = model_class(**model_kwargs)
# Same as: model = ResNet50(num_classes=10)
```

---

## 🎨 **Visual Diagram**

```
USER COMMAND
═══════════
python server_generic.py --config configs/resnet_config.py
                                   ↓
                          ┌────────────────────┐
                          │ Argparse extracts  │
                          │ "configs/resnet_   │
                          │  config.py"        │
                          └─────────┬──────────┘
                                    ↓
                          ┌─────────────────────┐
                          │ load_model_from_    │
                          │ config() function   │
                          └─────────┬───────────┘
                                    ↓
                    ┌───────────────────────────────┐
                    │ Dynamic Import Process        │
                    ├───────────────────────────────┤
                    │ 1. Read file path             │
                    │ 2. Add directory to sys.path  │
                    │ 3. import resnet_config       │
                    │ 4. Execute the .py file       │
                    └─────────┬─────────────────────┘
                              ↓
              ┌───────────────────────────────────┐
              │ Config Module in Memory           │
              ├───────────────────────────────────┤
              │ MODEL_CLASS = ResNet50            │
              │ MODEL_KWARGS = {'num_classes':10} │
              │ OPTIMIZER_CLASS = Adam            │
              │ get_data_loaders = <function>     │
              └─────────┬─────────────────────────┘
                        ↓
         ┌──────────────────────────────────────┐
         │ Extract Attributes using getattr()   │
         ├──────────────────────────────────────┤
         │ model_class = config.MODEL_CLASS     │
         │ model_kwargs = config.MODEL_KWARGS   │
         └─────────┬────────────────────────────┘
                   ↓
    ┌──────────────────────────────────────────┐
    │ Use in Strategy/Client                   │
    ├──────────────────────────────────────────┤
    │ strategy = GenericBlendStrategy(         │
    │     model_class=ResNet50,                │
    │     model_kwargs={'num_classes': 10}     │
    │ )                                        │
    └──────────────────────────────────────────┘
```

---

## 🔬 **Code Trace Example**

Let's trace **exactly** what happens with `configs/mnist_config.py`:

### **1. Config File Content**

```python
# configs/mnist_config.py

from model import Net
import torch.optim as optim

MODEL_CLASS = Net
MODEL_KWARGS = {
    'input_size': 784,
    'hidden_size': 128,
    'num_classes': 10
}

OPTIMIZER_CLASS = optim.SGD
OPTIMIZER_KWARGS = {'lr': 0.01, 'momentum': 0.9}

def get_data_loaders(client_id):
    # ... data loading code ...
    return trainloader, testloader
```

### **2. Command Execution**

```bash
python client_generic.py --config configs/mnist_config.py --client-id 0
```

### **3. Inside client_generic.py**

```python
# Line 163: Parse arguments
args = parser.parse_args()
# args.config = "configs/mnist_config.py"

# Line 172-173: Load config
print(f"📋 Loading config from: {args.config}")
config = load_client_config(args.config)
# config is now the imported module!

# Line 176-177: Extract model info
model_class = config.MODEL_CLASS
# model_class = Net (the actual class)

model_kwargs = config.MODEL_KWARGS
# model_kwargs = {'input_size': 784, 'hidden_size': 128, 'num_classes': 10}

# Line 181: Call function from config
train_loader, test_loader = config.get_data_loaders(args.client_id)
# Calls the get_data_loaders function defined in mnist_config.py

# Line 184-187: Get optional attributes (with defaults)
optimizer_class = getattr(config, 'OPTIMIZER_CLASS', optim.SGD)
# optimizer_class = optim.SGD (from config)

optimizer_kwargs = getattr(config, 'OPTIMIZER_KWARGS', {'lr': 0.01})
# optimizer_kwargs = {'lr': 0.01, 'momentum': 0.9}

criterion = getattr(config, 'CRITERION', nn.CrossEntropyLoss())
# If CRITERION exists in config, use it; otherwise use CrossEntropyLoss()

# Line 195-204: Create client with all config values
client = GenericClient(
    model_class=Net,                              # From config
    model_kwargs={'input_size': 784, ...},       # From config
    train_loader=trainloader,                     # From config function
    test_loader=testloader,                       # From config function
    optimizer_class=optim.SGD,                    # From config
    optimizer_kwargs={'lr': 0.01, 'momentum': 0.9}, # From config
    criterion=nn.CrossEntropyLoss(),             # Default or from config
    epochs_per_round=3                            # Default or from config
)
```

---

## 🎯 **How It Knows Which Framework**

### **Option 1: Implicit Detection (Current System)**

The system **doesn't care** what framework you use! It just:

1. Takes `MODEL_CLASS` from config
2. Creates instance: `model = MODEL_CLASS(**MODEL_KWARGS)`
3. Calls `model.state_dict()` to get parameters

**If it's PyTorch:**
```python
MODEL_CLASS = ResNet50  # PyTorch model
model = ResNet50(num_classes=10)
params = model.state_dict()  # ✅ Works! Returns OrderedDict
```

**If it's TensorFlow (with adapter):**
```python
MODEL_CLASS = create_keras_model  # TensorFlow function
FRAMEWORK = 'tensorflow'  # ← Explicit declaration

model = create_keras_model(num_classes=10)
# System would check: if FRAMEWORK == 'tensorflow', use TensorFlowAdapter
```

### **Option 2: Explicit Framework Declaration**

For non-PyTorch frameworks, config can declare:

```python
# configs/tensorflow_config.py

FRAMEWORK = 'tensorflow'  # ← Tell system explicitly
MODEL_CLASS = create_keras_model
# ...
```

Then in client/server:

```python
# Check if FRAMEWORK attribute exists
framework = getattr(config, 'FRAMEWORK', 'pytorch')

if framework == 'tensorflow':
    adapter = TensorFlowAdapter()
else:
    adapter = PyTorchAdapter()
```

---

## 🔍 **How getattr() Works**

```python
# Basic usage
config = import('mnist_config')  # Hypothetical

# If config has: MODEL_CLASS = Net
model_class = getattr(config, 'MODEL_CLASS')
# Returns: Net

# If config DOESN'T have MODEL_CLASS
model_class = getattr(config, 'MODEL_CLASS', DefaultModel)
# Returns: DefaultModel (the default value)

# This is like:
try:
    model_class = config.MODEL_CLASS
except AttributeError:
    model_class = DefaultModel
```

**In the code:**

```python
optimizer_class = getattr(config, 'OPTIMIZER_CLASS', optim.SGD)
```

Means:
- If config file has `OPTIMIZER_CLASS = optim.Adam`, use Adam
- If config file doesn't have `OPTIMIZER_CLASS`, use default SGD

---

## 🧪 **Try It Yourself**

Create a test to see config loading:

```python
# test_config_loading.py

import importlib
import sys
from pathlib import Path

def load_config(config_path):
    """Same as in the real system"""
    config_path = Path(config_path)
    sys.path.insert(0, str(config_path.parent))
    module_name = config_path.stem
    config = importlib.import_module(module_name)
    return config

# Load the config
config = load_config('configs/mnist_config.py')

# Check what's in it
print("MODEL_CLASS:", config.MODEL_CLASS)
print("MODEL_KWARGS:", config.MODEL_KWARGS)
print("OPTIMIZER_CLASS:", config.OPTIMIZER_CLASS)
print("Has FRAMEWORK?:", hasattr(config, 'FRAMEWORK'))

# Try to create the model
model = config.MODEL_CLASS(**config.MODEL_KWARGS)
print("Model created:", model)
```

Run it:
```bash
python test_config_loading.py

# Output:
# MODEL_CLASS: <class 'model.Net'>
# MODEL_KWARGS: {'input_size': 784, 'hidden_size': 128, 'num_classes': 10}
# OPTIMIZER_CLASS: <class 'torch.optim.sgd.SGD'>
# Has FRAMEWORK?: False
# Model created: Net(...)
```

---

## 💡 **Key Insights**

### **1. Config is Just a Python Module**

```python
# Writing this in a config file:
MODEL_CLASS = ResNet50

# Is the same as:
import resnet_config
model_class = resnet_config.MODEL_CLASS
```

### **2. Dynamic Import is Powerful**

```python
# Instead of hardcoding:
from model import Net
model = Net()

# We do:
config_path = args.config  # User specifies
config = import_module(config_path)
model = config.MODEL_CLASS(**config.MODEL_KWARGS)
```

### **3. Framework Detection is Attribute-Based**

```python
# System checks for attributes:
if hasattr(config, 'FRAMEWORK'):
    framework = config.FRAMEWORK
else:
    framework = 'pytorch'  # Default
```

---

## 🎯 **Summary**

### **How System Knows Config:**

```
1. User specifies: --config configs/resnet_config.py
2. System reads: args.config
3. System imports: importlib.import_module('resnet_config')
4. System extracts: config.MODEL_CLASS, config.MODEL_KWARGS
5. System uses: model = MODEL_CLASS(**MODEL_KWARGS)
```

### **How System Knows Framework:**

**Currently:**
- Assumes PyTorch (works for all PyTorch models)
- Doesn't check framework explicitly

**With Adapters:**
- Check `config.FRAMEWORK` attribute
- If 'tensorflow', use TensorFlowAdapter
- If 'sklearn', use SklearnAdapter
- Default to PyTorchAdapter

### **Why This is Flexible:**

- ✅ No hardcoded models
- ✅ No hardcoded datasets
- ✅ User controls everything via config
- ✅ System is framework-agnostic
- ✅ Just swap config file to change model!

**That's the power of dynamic configuration!** 🚀


