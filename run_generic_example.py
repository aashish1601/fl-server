#!/usr/bin/env python3
"""
Example: Run generic federated learning with any config
"""

import subprocess
import sys
import time
import threading
import argparse

def run_server(config_path, alpha=0.5, num_rounds=5, min_clients=1):
    """Run generic server"""
    print(f"🏠 Starting server with config: {config_path}")
    try:
        subprocess.run([
            sys.executable, "server_generic.py",
            "--config", config_path,
            "--server-address", "0.0.0.0:8080",
            "--num-rounds", str(num_rounds),
            "--alpha", str(alpha),
            "--min-clients", str(min_clients)
        ], check=True)
    except Exception as e:
        print(f"❌ Server error: {e}")

def run_client(config_path, client_id=0):
    """Run generic client"""
    print(f"👤 Starting client {client_id} with config: {config_path}")
    try:
        subprocess.run([
            sys.executable, "client_generic.py",
            "--config", config_path,
            "--server-address", "127.0.0.1:8080",
            "--client-id", str(client_id)
        ], check=True)
    except Exception as e:
        print(f"❌ Client {client_id} error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run Generic FL Example")
    parser.add_argument("--config", type=str, default="configs/mnist_config.py",
                       help="Path to config file")
    parser.add_argument("--num-clients", type=int, default=2,
                       help="Number of clients")
    parser.add_argument("--num-rounds", type=int, default=5,
                       help="Number of training rounds")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Blending weight")
    args = parser.parse_args()
    
    print("=" * 70)
    print("🌐 GENERIC FEDERATED LEARNING")
    print("=" * 70)
    print(f"📋 Config: {args.config}")
    print(f"👥 Clients: {args.num_clients}")
    print(f"🔄 Rounds: {args.num_rounds}")
    print(f"🔀 Alpha: {args.alpha}")
    print("=" * 70)
    
    # Start server
    server_thread = threading.Thread(
        target=run_server,
        args=(args.config, args.alpha, args.num_rounds, args.num_clients),
        daemon=True
    )
    server_thread.start()
    
    # Wait for server
    print("\n⏳ Waiting for server to start...")
    time.sleep(10)
    
    # Start clients
    client_threads = []
    for client_id in range(args.num_clients):
        thread = threading.Thread(
            target=run_client,
            args=(args.config, client_id),
            daemon=True
        )
        thread.start()
        client_threads.append(thread)
        time.sleep(2)
    
    # Wait for completion
    print("\n🔄 Training in progress...\n")
    server_thread.join()
    for thread in client_threads:
        thread.join()
    
    print("\n" + "=" * 70)
    print("🎉 FEDERATED LEARNING COMPLETED!")
    print("=" * 70)
    print("📁 Check ./models/ for saved models")
    print("=" * 70)

if __name__ == "__main__":
    main()


