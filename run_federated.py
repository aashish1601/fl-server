#!/usr/bin/env python3
"""
Railway deployment script for federated learning
This script runs the complete federated learning process
"""

import subprocess
import time
import os
import sys
import threading

def run_server():
    """Run the server in a separate thread"""
    print("🏠 Starting server...")
    try:
        subprocess.run([
            sys.executable, "server_with_save.py", 
            "--server-address", "0.0.0.0:8080",
            "--num-rounds", "3"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed: {e}")
    except Exception as e:
        print(f"❌ Server error: {e}")

def run_client(client_id):
    """Run a client in a separate thread"""
    print(f"👥 Starting client {client_id}...")
    try:
        subprocess.run([
            sys.executable, "client.py", 
            "--client-id", str(client_id),
            "--server-address", "127.0.0.1:8080",
            "--cloud-mode"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Client {client_id} failed: {e}")
    except Exception as e:
        print(f"❌ Client {client_id} error: {e}")

def main():
    """Main function to run federated learning"""
    print("🚂 Starting Federated Learning on Railway")
    print("=" * 50)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Start server in background
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(10)
    
    # Start clients
    client_threads = []
    for client_id in [0, 1]:
        client_thread = threading.Thread(target=run_client, args=(client_id,), daemon=True)
        client_thread.start()
        client_threads.append(client_thread)
        time.sleep(2)  # Stagger client starts
    
    # Wait for all threads to complete
    print("🔄 Federated learning in progress...")
    server_thread.join()
    for client_thread in client_threads:
        client_thread.join()
    
    print("🎉 Federated learning completed!")
    print("📁 Check the 'models' directory for saved models")

if __name__ == "__main__":
    main()
