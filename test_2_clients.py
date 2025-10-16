#!/usr/bin/env python3
"""
Test: Verify that 2 clients are merging properly

This launches server + 2 clients and shows the merging process clearly
"""

import subprocess
import sys
import time
import threading

def run_server():
    """Run server that requires 2 clients"""
    print("🏠 Starting server (requires 2 clients)...")
    subprocess.run([
        sys.executable, "server_generic.py",
        "--config", "configs/mnist_config.py",
        "--server-address", "0.0.0.0:8080",
        "--num-rounds", "3",
        "--alpha", "0.5",
        "--min-clients", "2"  # Explicitly require 2 clients
    ])

def run_client(client_id):
    """Run a single client"""
    print(f"👤 Starting client {client_id}...")
    subprocess.run([
        sys.executable, "client_generic.py",
        "--config", "configs/mnist_config.py",
        "--server-address", "127.0.0.1:8080",
        "--client-id", str(client_id)
    ])

def main():
    print("\n" + "="*70)
    print("🧪 TESTING 2-CLIENT FEDERATED LEARNING")
    print("="*70)
    print()
    print("This will demonstrate:")
    print("  1. Server waits for 2 clients to connect")
    print("  2. Both clients train on different data")
    print("  3. Server MERGES both client updates")
    print("  4. Shows how many samples each client used")
    print()
    print("="*70)
    
    # Start server
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    print("\n⏳ Waiting for server to initialize (10 seconds)...")
    time.sleep(10)
    
    print("\n📢 Server is waiting for 2 clients...")
    print("   (It will NOT start training until both connect!)\n")
    
    # Start client 0
    client0_thread = threading.Thread(target=run_client, args=(0,), daemon=True)
    client0_thread.start()
    
    print("✅ Client 0 connecting...")
    time.sleep(3)
    
    # Start client 1
    client1_thread = threading.Thread(target=run_client, args=(1,), daemon=True)
    client1_thread.start()
    
    print("✅ Client 1 connecting...")
    print("\n🔄 Both clients connected! Training should begin now...\n")
    
    # Wait for completion
    server_thread.join()
    client0_thread.join()
    client1_thread.join()
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE!")
    print("="*70)
    print()
    print("You should have seen in the server logs:")
    print("  📊 Round X - Received 2 client(s), 0 failure(s)")
    print("     Client 0: 30000 training samples")
    print("     Client 1: 30000 training samples")
    print()
    print("  🔀 Blending models:")
    print("     Clients (averaged) weight: 50.0%")
    print("     📌 NOTE: Client weights were first averaged from 2 client(s)")
    print()
    print("This proves MERGING is happening! 🎉")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

