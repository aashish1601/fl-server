#!/usr/bin/env python3
"""
Single Client for Federated Learning with Server Blending
Uses the second half of MNIST (images 30,000-59,999)
"""

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model import Net
import argparse
import os

torch.manual_seed(42)

def load_client_data():
    """Load client's private data (second half of MNIST)"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST
    trainset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    testset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    
    # Client gets second half (30,000-59,999) - different from baseline!
    train_indices = list(range(30000, len(trainset)))
    test_indices = list(range(5000, len(testset)))
    
    train_subset = Subset(trainset, train_indices)
    test_subset = Subset(testset, test_indices)
    
    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    return trainloader, testloader


class SingleClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = Net()
        self.trainloader, self.testloader = load_client_data()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"📊 Client loaded {len(self.trainloader.dataset)} training samples")
        print(f"📊 Client loaded {len(self.testloader.dataset)} test samples")
    
    def get_parameters(self, config):
        """Get model parameters as NumPy arrays"""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters):
        """Set model parameters from NumPy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train the model on client's private data"""
        print("\n" + "="*50)
        print("🏋️  Client starting local training...")
        print("="*50)
        
        self.set_parameters(parameters)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        # Train for 3 epochs per round
        self.model.train()
        for epoch in range(3):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            avg_loss = running_loss / len(self.trainloader)
            accuracy = 100 * correct / total
            print(f"  Epoch {epoch+1}/3 - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        print("✅ Local training completed!")
        print("="*50)
        
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        """Evaluate the model on client's test data"""
        self.set_parameters(parameters)
        
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        avg_loss = test_loss / len(self.testloader)
        
        print(f"\n📊 Client Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return avg_loss, len(self.testloader.dataset), {"accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(description="Single Flower Client")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080", help="Server address")
    parser.add_argument("--cloud-mode", action="store_true", help="Enable cloud deployment mode")
    args = parser.parse_args()
    
    # Get server address from environment if in cloud mode
    if args.cloud_mode or os.getenv('RAILWAY_ENVIRONMENT'):
        server_address = os.getenv('SERVER_URL', os.getenv('SERVER_ADDRESS', args.server_address))
        print(f"☁️  Running in cloud mode")
    else:
        server_address = args.server_address
    
    print("=" * 60)
    print("🚀 Starting Single Flower Client")
    print("=" * 60)
    print(f"📡 Connecting to server: {server_address}")
    print("=" * 60)
    
    # Create and start client
    client = SingleClient()
    
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )
    
    print("\n🎉 Client finished all rounds!")


if __name__ == "__main__":
    main()


