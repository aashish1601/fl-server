import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model import Net
import numpy as np
import argparse
import os

# Set random seed for reproducibility
torch.manual_seed(42)

def load_data(client_id):
    """Load data for a specific client (split MNIST dataset)"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    # Load MNIST dataset
    trainset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    testset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    
    # Split data between clients (client_id can be 0 or 1)
    if client_id == 0:
        # Client 0 gets first half of training data
        train_indices = list(range(0, len(trainset) // 2))
        test_indices = list(range(0, len(testset) // 2))
    else:
        # Client 1 gets second half of training data
        train_indices = list(range(len(trainset) // 2, len(trainset)))
        test_indices = list(range(len(testset) // 2, len(testset)))
    
    train_subset = Subset(trainset, train_indices)
    test_subset = Subset(testset, test_indices)
    
    trainloader = DataLoader(train_subset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    return trainloader, testloader

class MnistClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = Net()
        self.trainloader, self.testloader = load_data(client_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"Client {client_id}: Loaded {len(self.trainloader.dataset)} training samples")
        print(f"Client {client_id}: Loaded {len(self.testloader.dataset)} test samples")
    
    def get_parameters(self, config):
        """Get model parameters as a list of NumPy arrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train the model using the provided parameters"""
        self.set_parameters(parameters)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        # Train for a few epochs
        self.model.train()
        for epoch in range(3):  # Train for 3 epochs per round
            running_loss = 0.0
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            avg_loss = running_loss / len(self.trainloader)
            print(f"Client {self.client_id}, Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        """Evaluate the model using the provided parameters"""
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
        
        print(f"Client {self.client_id} - Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return avg_loss, len(self.testloader.dataset), {"accuracy": accuracy}

def main():
    parser = argparse.ArgumentParser(description="Flower MNIST Client")
    parser.add_argument("--client-id", type=int, default=0, help="Client ID (0 or 1)")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080", help="Server address")
    parser.add_argument("--cloud-mode", action="store_true", help="Enable cloud deployment mode")
    args = parser.parse_args()
    
    # Get server address from environment if in cloud mode
    if args.cloud_mode or os.getenv('RAILWAY_ENVIRONMENT'):
        server_address = os.getenv('SERVER_URL', os.getenv('SERVER_ADDRESS', args.server_address))
        print(f"☁️ Running in cloud mode")
        print(f"🌐 Server URL from environment: {server_address}")
    else:
        server_address = args.server_address
    
    print(f"🚀 Starting Flower Client {args.client_id}")
    print(f"📡 Connecting to server: {server_address}")
    
    # Create and start client
    client = MnistClient(args.client_id)
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )

if __name__ == "__main__":
    main()
