#!/usr/bin/env python3
"""
Simple Example: How Config Detection Works

Run this to see exactly how the system loads and uses config files!
"""

import importlib
import sys
from pathlib import Path

def load_config(config_path):
    """
    This is EXACTLY how server_generic.py and client_generic.py load configs
    """
    print(f"\n{'='*60}")
    print(f"LOADING CONFIG: {config_path}")
    print('='*60)
    
    # Convert to Path object
    config_path = Path(config_path)
    print(f"1️⃣  Converted to Path: {config_path}")
    
    # Add parent directory to Python's import path
    parent_dir = str(config_path.parent)
    sys.path.insert(0, parent_dir)
    print(f"2️⃣  Added to sys.path: {parent_dir}")
    
    # Get module name (filename without .py)
    module_name = config_path.stem  # "mnist_config" from "mnist_config.py"
    print(f"3️⃣  Module name: {module_name}")
    
    # Dynamically import the module (like "import mnist_config")
    print(f"4️⃣  Importing module...")
    config = importlib.import_module(module_name)
    print(f"    ✅ Module imported successfully!")
    
    return config


def inspect_config(config):
    """Show what's inside the config"""
    print(f"\n{'='*60}")
    print("CONFIG ATTRIBUTES")
    print('='*60)
    
    # Check for MODEL_CLASS
    if hasattr(config, 'MODEL_CLASS'):
        model_class = config.MODEL_CLASS
        print(f"✅ MODEL_CLASS found: {model_class}")
        print(f"   Type: {type(model_class)}")
        print(f"   Name: {model_class.__name__}")
    else:
        print("❌ MODEL_CLASS not found")
    
    # Check for MODEL_KWARGS
    if hasattr(config, 'MODEL_KWARGS'):
        model_kwargs = config.MODEL_KWARGS
        print(f"✅ MODEL_KWARGS found: {model_kwargs}")
    else:
        print("❌ MODEL_KWARGS not found")
    
    # Check for OPTIMIZER_CLASS
    if hasattr(config, 'OPTIMIZER_CLASS'):
        optimizer_class = config.OPTIMIZER_CLASS
        print(f"✅ OPTIMIZER_CLASS found: {optimizer_class.__name__}")
    else:
        print("⚠️  OPTIMIZER_CLASS not found (will use default)")
    
    # Check for FRAMEWORK (optional)
    if hasattr(config, 'FRAMEWORK'):
        framework = config.FRAMEWORK
        print(f"✅ FRAMEWORK found: {framework}")
    else:
        print(f"⚠️  FRAMEWORK not found (will assume PyTorch)")
    
    # Check for get_data_loaders function
    if hasattr(config, 'get_data_loaders'):
        print(f"✅ get_data_loaders function found")
        print(f"   Type: {type(config.get_data_loaders)}")
    else:
        print("❌ get_data_loaders not found")


def test_model_creation(config):
    """Test creating a model from config"""
    print(f"\n{'='*60}")
    print("TESTING MODEL CREATION")
    print('='*60)
    
    try:
        model_class = config.MODEL_CLASS
        model_kwargs = config.MODEL_KWARGS
        
        print(f"Creating: {model_class.__name__}(**{model_kwargs})")
        model = model_class(**model_kwargs)
        print(f"✅ Model created successfully!")
        print(f"   Model: {model}")
        
        # Show model parameters
        if hasattr(model, 'state_dict'):
            state_dict = model.state_dict()
            print(f"   Parameters: {len(state_dict)} layers")
            for i, (name, param) in enumerate(list(state_dict.items())[:3]):
                print(f"      {i+1}. {name}: shape {param.shape}")
            if len(state_dict) > 3:
                print(f"      ... and {len(state_dict)-3} more")
        
        return model
        
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return None


def main():
    """Run the demonstration"""
    print("\n" + "="*60)
    print("CONFIG DETECTION DEMONSTRATION")
    print("="*60)
    print("This shows EXACTLY how server/client load configs!\n")
    
    # Example 1: MNIST config
    print("\n" + "🔹"*30)
    print("EXAMPLE 1: MNIST Config")
    print("🔹"*30)
    
    config_path = "configs/mnist_config.py"
    config = load_config(config_path)
    inspect_config(config)
    model = test_model_creation(config)
    
    # Example 2: CIFAR-10 config
    print("\n" + "🔹"*30)
    print("EXAMPLE 2: CIFAR-10 Config")
    print("🔹"*30)
    
    config_path = "configs/cifar10_config.py"
    config = load_config(config_path)
    inspect_config(config)
    
    # Example 3: Custom CNN config
    print("\n" + "🔹"*30)
    print("EXAMPLE 3: Custom CNN Config")
    print("🔹"*30)
    
    config_path = "configs/custom_cnn_config.py"
    config = load_config(config_path)
    inspect_config(config)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("✅ Config files are just Python modules")
    print("✅ System imports them dynamically using importlib")
    print("✅ Extracts MODEL_CLASS, MODEL_KWARGS, etc. using getattr()")
    print("✅ Creates model: MODEL_CLASS(**MODEL_KWARGS)")
    print("✅ No hardcoded models or frameworks!")
    print("\n" + "="*60)
    print("This is how your FL system is framework-agnostic! 🚀")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()


