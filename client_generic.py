#!/usr/bin/env python3
"""
Generic Federated Learning Client
Works with ANY PyTorch model and dataset!
"""

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import importlib

class GenericClient(fl.client.NumPyClient):
    """
    Generic client that works with any PyTorch model and dataset
    
    Usage:
        client = GenericClient(
            model_class=YourModel,
            model_kwargs={'num_classes': 100},
            train_loader=your_train_loader,
            test_loader=your_test_loader,
            optimizer_class=optim.Adam,
            optimizer_kwargs={'lr': 0.001}
        )
    """
    
    def __init__(self, model_class, model_kwargs, train_loader, test_loader,
                 optimizer_class=optim.SGD, optimizer_kwargs=None,
                 criterion=None, epochs_per_round=3, device=None):
        
        # Create model
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.model = model_class(**model_kwargs)
        
        # Setup device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Data loaders
        self.trainloader = train_loader
        self.testloader = test_loader
        
        # Training configuration
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 0.01}
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.epochs_per_round = epochs_per_round
        
        print(f"✅ Client initialized:")
        print(f"   Model: {model_class.__name__}")
        print(f"   Training samples: {len(train_loader.dataset)}")
        print(f"   Test samples: {len(test_loader.dataset)}")
        print(f"   Device: {self.device}")
        print(f"   Optimizer: {optimizer_class.__name__}")
        print(f"   Epochs per round: {epochs_per_round}")
    
    def get_parameters(self, config):
        """Get model parameters as NumPy arrays"""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters):
        """Set model parameters from NumPy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train the model"""
        print("\n" + "="*60)
        print("🏋️  Starting local training...")
        print("="*60)
        
        self.set_parameters(parameters)
        
        # Create optimizer
        optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs_per_round):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(self.trainloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            avg_loss = running_loss / len(self.trainloader)
            accuracy = 100 * correct / total
            print(f"  Epoch {epoch+1}/{self.epochs_per_round} - "
                  f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        print("✅ Training completed!")
        print("="*60)
        
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        """Evaluate the model"""
        self.set_parameters(parameters)
        
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        avg_loss = test_loss / len(self.testloader)
        
        print(f"\n📊 Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return avg_loss, len(self.testloader.dataset), {"accuracy": accuracy}


def load_client_config(config_path):
    """Load client configuration from Python file"""
    import sys
    from pathlib import Path
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    sys.path.insert(0, str(config_path.parent))
    module_name = config_path.stem
    config = importlib.import_module(module_name)
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Generic FL Client")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to client config file")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--client-id", type=int, default=0)
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 GENERIC FEDERATED LEARNING CLIENT")
    print("=" * 70)
    
    # Load configuration
    print(f"\n📋 Loading config from: {args.config}")
    config = load_client_config(args.config)
    
    # Get model class and kwargs
    model_class = config.MODEL_CLASS
    model_kwargs = config.MODEL_KWARGS
    
    # Get data loaders
    print(f"📊 Loading data for client {args.client_id}...")
    train_loader, test_loader = config.get_data_loaders(args.client_id)
    
    # Get training configuration
    optimizer_class = getattr(config, 'OPTIMIZER_CLASS', optim.SGD)
    optimizer_kwargs = getattr(config, 'OPTIMIZER_KWARGS', {'lr': 0.01})
    criterion = getattr(config, 'CRITERION', nn.CrossEntropyLoss())
    epochs_per_round = getattr(config, 'EPOCHS_PER_ROUND', 3)
    
    print(f"✅ Model: {model_class.__name__}")
    print(f"✅ Optimizer: {optimizer_class.__name__}")
    print(f"📡 Connecting to: {args.server_address}")
    print("=" * 70)
    
    # Create client
    client = GenericClient(
        model_class=model_class,
        model_kwargs=model_kwargs,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        criterion=criterion,
        epochs_per_round=epochs_per_round
    )
    
    # Connect to server
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client
    )
    
    print("\n🎉 Client finished!")


if __name__ == "__main__":
    main()


