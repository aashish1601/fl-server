#!/usr/bin/env python3
"""
Server with Baseline Model Blending
Blends server's existing model with single client updates each round
"""

import flwr as fl
import torch
import numpy as np
from model import Net
import argparse
import os

class BlendServerClient(fl.server.strategy.FedAvg):
    """
    Custom strategy that blends server's baseline model with client updates
    
    Formula: W_new = (1-α)·W_server + α·W_client
    
    Args:
        alpha: Blending weight (0 < α ≤ 1)
               - α=1.0: Use only client's weights (standard FedAvg)
               - α=0.5: Equal blend (default)
               - α=0.1: Trust server more, small client nudges
        initial_parameters: Server's baseline model weights
        model_class: Model architecture for saving
        num_rounds: Total training rounds
    """
    
    def __init__(self, alpha: float, initial_parameters, model_class=Net, 
                 num_rounds=5, model_save_path='./models', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self._current_parameters = initial_parameters
        self.model_class = model_class
        self.num_rounds = num_rounds
        self.model_save_path = model_save_path
        os.makedirs(self.model_save_path, exist_ok=True)
        
        print(f"🔀 BlendServerClient initialized:")
        print(f"   α (alpha) = {self.alpha}")
        print(f"   Server contribution = {1-self.alpha:.1%}")
        print(f"   Client contribution = {self.alpha:.1%}")
    
    def initialize_parameters(self, client_manager):
        """Send baseline model to clients at the start"""
        print("📤 Sending baseline model to clients...")
        return fl.common.ndarrays_to_parameters(self._current_parameters)
    
    def aggregate_fit(self, server_round, results, failures):
        """Blend server's current model with client's update"""
        
        # Handle failures
        if len(failures) > 0:
            print(f"⚠️  Round {server_round}: {len(failures)} client(s) failed")
            if len(results) == 0:
                print(f"❌ No successful results, keeping current model")
                return fl.common.ndarrays_to_parameters(self._current_parameters), {}
        
        # Get client's updated weights using parent FedAvg
        aggregated = super().aggregate_fit(server_round, results, failures)
        
        if aggregated is None:
            print(f"⚠️  No aggregation for round {server_round}")
            return fl.common.ndarrays_to_parameters(self._current_parameters), {}
        
        # Extract parameters (Flower returns tuple of (Parameters, metrics))
        aggregated_parameters = aggregated[0] if isinstance(aggregated, tuple) else aggregated
        
        # Convert parameters to list of numpy arrays using Flower's utility
        aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
        
        # Blend: W_new = (1-α)·W_server + α·W_client
        print(f"\n🔀 Round {server_round} - Blending models:")
        print(f"   Server weight: {1-self.alpha:.1%}")
        print(f"   Client weight: {self.alpha:.1%}")
        
        blended_weights = []
        for w_server, w_client in zip(self._current_parameters, aggregated_ndarrays):
            # Ensure both are numpy arrays
            w_server_np = np.array(w_server, dtype=np.float32)
            w_client_np = np.array(w_client, dtype=np.float32)
            
            # Blend
            w_blended = (1 - self.alpha) * w_server_np + self.alpha * w_client_np
            blended_weights.append(w_blended.astype(np.float32))
        
        # Update server's current parameters
        self._current_parameters = blended_weights
        
        # Save the blended model
        self._save_model(server_round)
        
        # Return tuple of (Parameters, metrics) as expected by Flower
        return fl.common.ndarrays_to_parameters(blended_weights), {}
    
    def _save_model(self, server_round):
        """Save the blended model to disk"""
        try:
            model = self.model_class()
            
            # Create state dict
            state_dict = {}
            model_keys = list(model.state_dict().keys())
            
            for i, param_array in enumerate(self._current_parameters):
                if i < len(model_keys):
                    state_dict[model_keys[i]] = torch.tensor(param_array)
            
            model.load_state_dict(state_dict, strict=True)
            
            # Save round model
            model_path = os.path.join(self.model_save_path, f'blended_model_round_{server_round}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"💾 Saved blended model to '{model_path}'")
            
            # Save final model
            if server_round == self.num_rounds:
                final_path = os.path.join(self.model_save_path, 'final_blended_model.pth')
                torch.save(model.state_dict(), final_path)
                print(f"🏆 Final blended model saved to '{final_path}'")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not save model for round {server_round}: {e}")


def create_baseline_model(train_epochs=2, save_path='baseline_model.pth'):
    """
    Create a baseline model by training on a small portion of MNIST
    This simulates the server having some initial knowledge
    """
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    import torch.nn as nn
    import torch.optim as optim
    
    print("🏗️  Creating baseline model for server...")
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    
    # Use only first 10,000 samples for baseline (server's initial data)
    baseline_indices = list(range(0, 10000))
    baseline_subset = Subset(trainset, baseline_indices)
    baseline_loader = DataLoader(baseline_subset, batch_size=64, shuffle=True)
    
    # Train baseline model
    model = Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    model.train()
    for epoch in range(train_epochs):
        running_loss = 0.0
        for images, labels in baseline_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(baseline_loader)
        print(f"  Baseline Epoch {epoch+1}/{train_epochs}, Loss: {avg_loss:.4f}")
    
    # Save baseline
    torch.save(model.state_dict(), save_path)
    print(f"✅ Baseline model saved to '{save_path}'\n")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Flower Server with Model Blending")
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080", help="Server address")
    parser.add_argument("--num-rounds", type=int, default=5, help="Number of federated rounds")
    parser.add_argument("--alpha", type=float, default=0.5, help="Blending weight (0 < α ≤ 1)")
    parser.add_argument("--baseline-path", type=str, default="baseline_model.pth", help="Path to baseline model")
    parser.add_argument("--create-baseline", action="store_true", help="Create new baseline model")
    args = parser.parse_args()
    
    # Get port from environment (for Railway)
    port = os.getenv('PORT', '8080')
    server_address = f"0.0.0.0:{port}"
    
    print("=" * 60)
    print("🏠 Starting Flower Server with Model Blending")
    print("=" * 60)
    print(f"🌐 Server address: {server_address}")
    print(f"🔄 Number of rounds: {args.num_rounds}")
    print(f"🔀 Alpha (blending weight): {args.alpha}")
    print(f"📁 Models will be saved to: ./models")
    print("=" * 60)
    
    # Create or load baseline model
    if args.create_baseline or not os.path.exists(args.baseline_path):
        baseline_model = create_baseline_model(train_epochs=2, save_path=args.baseline_path)
    else:
        print(f"📂 Loading baseline model from '{args.baseline_path}'...")
        baseline_model = Net()
        baseline_model.load_state_dict(torch.load(args.baseline_path))
        print("✅ Baseline model loaded\n")
    
    # Convert to numpy arrays for Flower
    initial_parameters = [val.cpu().numpy() for val in baseline_model.state_dict().values()]
    
    # Create blending strategy
    strategy = BlendServerClient(
        alpha=args.alpha,
        initial_parameters=initial_parameters,
        model_class=Net,
        num_rounds=args.num_rounds,
        model_save_path='./models',
        min_fit_clients=1,              # Only need 1 client!
        min_evaluate_clients=1,
        min_available_clients=1,
    )
    
    # Start server
    print("\n⏳ Waiting for client to connect...")
    fl.server.start_server(
        server_address=server_address,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds)
    )
    
    print("\n" + "=" * 60)
    print("🎉 Federated learning with blending completed!")
    print("=" * 60)
    print(f"📊 Server model improved over {args.num_rounds} rounds")
    print(f"📁 Check './models' for saved models")
    print("=" * 60)


if __name__ == "__main__":
    main()


