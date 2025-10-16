#!/usr/bin/env python3
"""
Generic Federated Learning Server with Model Blending
Works with ANY PyTorch model and dataset!
"""

import flwr as fl
import torch
import numpy as np
import argparse
import os
import importlib

class GenericBlendStrategy(fl.server.strategy.FedAvg):
    """
    Generic blending strategy that works with ANY PyTorch model
    
    Usage:
        strategy = GenericBlendStrategy(
            model_class=YourModel,
            model_kwargs={'num_classes': 100, 'input_dim': 512},
            alpha=0.5,
            initial_parameters=baseline_weights
        )
    """
    
    def __init__(self, model_class, model_kwargs=None, alpha=0.5, 
                 initial_parameters=None, num_rounds=5, 
                 model_save_path='./models', *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model_class = model_class  # Can be ANY PyTorch nn.Module
        self.model_kwargs = model_kwargs or {}
        self.alpha = alpha
        self.num_rounds = num_rounds
        self.model_save_path = model_save_path
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # Create initial model to get parameters if not provided
        if initial_parameters is None:
            print("📝 No baseline provided, creating random initialization...")
            model = self.model_class(**self.model_kwargs)
            self._current_parameters = [p.detach().cpu().numpy() for p in model.state_dict().values()]
        else:
            self._current_parameters = initial_parameters
        
        print(f"🔀 GenericBlendStrategy initialized:")
        print(f"   Model: {self.model_class.__name__}")
        print(f"   Model kwargs: {self.model_kwargs}")
        print(f"   α (alpha) = {self.alpha}")
        print(f"   Server contribution = {1-self.alpha:.1%}")
        print(f"   Client contribution = {self.alpha:.1%}")
        print(f"   Parameters: {len(self._current_parameters)} layers")
    
    def initialize_parameters(self, client_manager):
        """Send initial model to clients"""
        print("📤 Sending initial model to clients...")
        return fl.common.ndarrays_to_parameters(self._current_parameters)
    
    def aggregate_fit(self, server_round, results, failures):
        """Blend server's current model with client updates"""
        
        # Show how many clients participated
        print(f"\n{'='*70}")
        print(f"📊 Round {server_round} - Received {len(results)} client(s), {len(failures)} failure(s)")
        for i, (params, num_samples) in enumerate(results):
            print(f"   Client {i}: {num_samples} training samples")
        print('='*70)
        
        if len(failures) > 0:
            print(f"⚠️  Round {server_round}: {len(failures)} client(s) failed")
            if len(results) == 0:
                print(f"❌ No successful results, keeping current model")
                return fl.common.ndarrays_to_parameters(self._current_parameters), {}
        
        # Get client's updated weights (already averaged by FedAvg)
        aggregated = super().aggregate_fit(server_round, results, failures)
        
        if aggregated is None:
            print(f"⚠️  No aggregation for round {server_round}")
            return fl.common.ndarrays_to_parameters(self._current_parameters), {}
        
        # Extract parameters
        aggregated_parameters = aggregated[0] if isinstance(aggregated, tuple) else aggregated
        
        # Convert parameters to list of numpy arrays
        aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
        
        # Blend: W_new = (1-α)·W_server + α·W_client_average
        print(f"\n🔀 Round {server_round} - Blending models:")
        print(f"   Server weight: {1-self.alpha:.1%}")
        print(f"   Clients (averaged) weight: {self.alpha:.1%}")
        print(f"   📌 NOTE: Client weights were first averaged from {len(results)} client(s)")
        
        # Blend server weights with aggregated client weights
        blended_weights = []
        for w_server, w_client in zip(self._current_parameters, aggregated_ndarrays):
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
        """Save model - works with ANY architecture"""
        try:
            # Create model instance
            model = self.model_class(**self.model_kwargs)
            
            # Load weights into model
            state_dict = {}
            model_keys = list(model.state_dict().keys())
            
            for i, param_array in enumerate(self._current_parameters):
                if i < len(model_keys):
                    state_dict[model_keys[i]] = torch.tensor(param_array)
            
            model.load_state_dict(state_dict, strict=True)
            
            # Save
            model_path = os.path.join(
                self.model_save_path, 
                f'{self.model_class.__name__}_round_{server_round}.pth'
            )
            torch.save(model.state_dict(), model_path)
            print(f"💾 Saved model to '{model_path}'")
            
            if server_round == self.num_rounds:
                final_path = os.path.join(
                    self.model_save_path, 
                    f'{self.model_class.__name__}_final.pth'
                )
                torch.save(model.state_dict(), final_path)
                print(f"🏆 Final model saved to '{final_path}'")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not save model for round {server_round}: {e}")


def load_model_from_config(config_path):
    """
    Load model class from a Python config file
    
    Config file should have:
        MODEL_CLASS = YourModel
        MODEL_KWARGS = {'param1': value1, ...}
    """
    import sys
    from pathlib import Path
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Add config directory to path
    sys.path.insert(0, str(config_path.parent))
    
    # Import the config module
    module_name = config_path.stem
    config = importlib.import_module(module_name)
    
    model_class = getattr(config, 'MODEL_CLASS')
    model_kwargs = getattr(config, 'MODEL_KWARGS', {})
    
    return model_class, model_kwargs


def main():
    parser = argparse.ArgumentParser(description="Generic FL Server")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to model config file (e.g., configs/mnist_config.py)")
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--baseline-path", type=str, default=None,
                       help="Path to pretrained baseline model (optional)")
    parser.add_argument("--min-clients", type=int, default=2)
    args = parser.parse_args()
    
    print("=" * 70)
    print("🌐 GENERIC FEDERATED LEARNING SERVER")
    print("=" * 70)
    
    # Load model configuration
    print(f"\n📋 Loading model config from: {args.config}")
    model_class, model_kwargs = load_model_from_config(args.config)
    print(f"✅ Model: {model_class.__name__}")
    print(f"✅ Parameters: {model_kwargs}")
    
    # Load or create baseline
    initial_parameters = None
    if args.baseline_path and os.path.exists(args.baseline_path):
        print(f"\n📂 Loading baseline from: {args.baseline_path}")
        model = model_class(**model_kwargs)
        model.load_state_dict(torch.load(args.baseline_path))
        initial_parameters = [p.detach().cpu().numpy() for p in model.state_dict().values()]
        print("✅ Baseline loaded")
    else:
        print("\n📝 No baseline provided, using random initialization")
    
    # Get port
    port = os.getenv('PORT', args.server_address.split(':')[-1])
    server_address = f"0.0.0.0:{port}"
    
    print(f"\n🌐 Server address: {server_address}")
    print(f"🔄 Number of rounds: {args.num_rounds}")
    print(f"🔀 Alpha: {args.alpha}")
    print(f"👥 Min clients: {args.min_clients}")
    print("=" * 70)
    
    # Create strategy
    strategy = GenericBlendStrategy(
        model_class=model_class,
        model_kwargs=model_kwargs,
        alpha=args.alpha,
        initial_parameters=initial_parameters,
        num_rounds=args.num_rounds,
        model_save_path='./models',
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
    )
    
    # Start server
    print("\n⏳ Waiting for clients to connect...\n")
    fl.server.start_server(
        server_address=server_address,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds)
    )
    
    print("\n" + "=" * 70)
    print("🎉 Federated learning completed!")
    print(f"📁 Models saved to: ./models")
    print("=" * 70)


if __name__ == "__main__":
    main()


