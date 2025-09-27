import flwr as fl
import torch
from model import Net
import argparse

def main():
    parser = argparse.ArgumentParser(description="Flower MNIST Server")
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080", help="Server address")
    parser.add_argument("--num-rounds", type=int, default=5, help="Number of federated rounds")
    args = parser.parse_args()
    
    print(f"🏠 Starting Flower Server")
    print(f"🌐 Server address: {args.server_address}")
    print(f"🔄 Number of rounds: {args.num_rounds}")
    
    # Define federated averaging strategy
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2,  # Minimum number of clients to train
        min_evaluate_clients=2,  # Minimum number of clients to evaluate
        min_available_clients=2,  # Minimum number of available clients
    )
    
    # Start server
    fl.server.start_server(
        server_address=args.server_address,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds)
    )

if __name__ == "__main__":
    main()
