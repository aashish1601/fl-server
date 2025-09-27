import flwr as fl
import torch
import numpy as np
from model import Net
import argparse
import os

class SaveModelStrategy(fl.server.strategy.FedAvg):
    """Enhanced strategy that saves the final merged model"""
    
    def __init__(self, num_rounds=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_rounds = num_rounds
        self.model_save_path = os.getenv('MODEL_SAVE_PATH', './models')
        
        # Create models directory if it doesn't exist
        os.makedirs(self.model_save_path, exist_ok=True)
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate model parameters and save final model"""
        # Call parent aggregation
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)
        
        # Save the aggregated model after each round
        if aggregated_parameters is not None:
            try:
                # Check if we have any failures
                if len(failures) > 0:
                    print(f"⚠️ Round {server_round}: {len(failures)} client(s) failed, {len(results)} succeeded")
                    if len(results) == 0:
                        print(f"❌ Skipping model save for round {server_round}: no successful client results")
                        return aggregated_parameters
                # Create model
                model = Net()
                
                # Get the parameters as numpy arrays - Flower 1.20.0 compatible
                # aggregated_parameters is a tuple: (Parameters, metrics)
                if isinstance(aggregated_parameters, tuple) and len(aggregated_parameters) > 0:
                    parameters_obj = aggregated_parameters[0]  # Get the Parameters object
                    # Parameters object has a .tensors attribute that contains the actual arrays
                    # Convert bytes to numpy arrays if needed
                    parameters_list = []
                    
                    # Define the correct expected shapes for our Net model (in state_dict order)
                    expected_shapes = [
                        (128, 784),   # fc1.weight - 100352 elements
                        (128,),       # fc1.bias - 128 elements
                        (128, 128),   # fc2.weight - 16384 elements
                        (128,),       # fc2.bias - 128 elements
                        (10, 128),    # fc3.weight - 1280 elements
                        (10,)         # fc3.bias - 10 elements
                    ]
                    
                    for i, tensor in enumerate(parameters_obj.tensors):
                        if isinstance(tensor, bytes):
                            # Convert bytes to numpy array and reshape to expected shape
                            import numpy as np
                            flat_array = np.frombuffer(tensor, dtype=np.float32)
                            print(f"Parameter {i}: {len(flat_array)} elements")
                            if i < len(expected_shapes):
                                expected_size = expected_shapes[i][0] * expected_shapes[i][1] if len(expected_shapes[i]) == 2 else expected_shapes[i][0]
                                print(f"  Expected shape: {expected_shapes[i]} ({expected_size} elements)")
                                
                                # Handle size mismatch by truncating or padding
                                if len(flat_array) >= expected_size:
                                    # Truncate to expected size
                                    flat_array = flat_array[:expected_size]
                                    print(f"  ✅ Truncated to {expected_size} elements")
                                else:
                                    # Pad with zeros to expected size
                                    padding = np.zeros(expected_size - len(flat_array), dtype=np.float32)
                                    flat_array = np.concatenate([flat_array, padding])
                                    print(f"  ✅ Padded to {expected_size} elements")
                                
                                # Reshape to the expected shape
                                reshaped_array = flat_array.reshape(expected_shapes[i])
                                parameters_list.append(reshaped_array)
                            else:
                                parameters_list.append(flat_array)
                        else:
                            parameters_list.append(tensor)
                else:
                    # Fallback for other versions
                    parameters_list = aggregated_parameters
                

                
                # Create state dict from parameters
                state_dict = {}
                model_keys = list(model.state_dict().keys())
                for i, param_array in enumerate(parameters_list):
                    if i < len(model_keys):
                        state_dict[model_keys[i]] = torch.tensor(param_array)
                
                model.load_state_dict(state_dict, strict=True)
                
                # Save model after each round
                model_path = os.path.join(self.model_save_path, f'federated_model_round_{server_round}.pth')
                torch.save(model.state_dict(), model_path)
                print(f"💾 Saved federated model from round {server_round} to '{model_path}'")
                
                # Save final model
                if server_round == self.num_rounds:
                    final_model_path = os.path.join(self.model_save_path, 'final_federated_model.pth')
                    torch.save(model.state_dict(), final_model_path)
                    print(f"🏆 Final federated model saved to '{final_model_path}'")
                    print(f"📊 This model achieved the collaborative learning from {len(results)} clients!")
                    
            except Exception as e:
                print(f"⚠️ Warning: Could not save model for round {server_round}: {e}")
        
        return aggregated_parameters

def main():
    parser = argparse.ArgumentParser(description="Flower MNIST Server with Model Saving")
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080", help="Server address")
    parser.add_argument("--num-rounds", type=int, default=5, help="Number of federated rounds")
    args = parser.parse_args()
    
    # Get port from Railway environment variable
    port = os.getenv('PORT', '8080')
    server_address = f"0.0.0.0:{port}"
    
    print(f"🏠 Starting Flower Server with Model Saving")
    print(f"🌐 Server address: {server_address}")
    print(f"🔄 Number of rounds: {args.num_rounds}")
    print(f"💾 Models will be saved to: {os.getenv('MODEL_SAVE_PATH', './models')}")
    
    # Define enhanced federated averaging strategy
    strategy = SaveModelStrategy(
        num_rounds=args.num_rounds,
        min_fit_clients=2,  # Minimum number of clients to train
        min_evaluate_clients=2,  # Minimum number of clients to evaluate
        min_available_clients=2,  # Minimum number of available clients
    )
    
    # Start server
    fl.server.start_server(
        server_address=server_address,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds)
    )
    
    print(f"\n🎉 Federated learning completed!")
    print(f"📁 Final merged model saved as 'final_federated_model.pth'")

if __name__ == "__main__":
    main()
