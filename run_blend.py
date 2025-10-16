#!/usr/bin/env python3
"""
Run Federated Learning with Server-Client Blending
Server has baseline model, one client improves it each round
"""

import subprocess
import time
import sys
import threading

def run_server(alpha=0.5, num_rounds=5):
    """Run the blending server"""
    print("🏠 Starting server with baseline model...")
    try:
        subprocess.run([
            sys.executable, "server_blend.py",
            "--server-address", "0.0.0.0:8080",
            "--num-rounds", str(num_rounds),
            "--alpha", str(alpha),
            "--create-baseline"  # Creates baseline model first time
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed: {e}")
    except Exception as e:
        print(f"❌ Server error: {e}")

def run_client():
    """Run the single client"""
    print("👤 Starting client...")
    try:
        subprocess.run([
            sys.executable, "client_single.py",
            "--server-address", "127.0.0.1:8080"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Client failed: {e}")
    except Exception as e:
        print(f"❌ Client error: {e}")

def main():
    """Main orchestrator"""
    print("=" * 70)
    print("🔀 FEDERATED LEARNING WITH SERVER-CLIENT BLENDING")
    print("=" * 70)
    print()
    print("📋 How it works:")
    print("   1. Server creates baseline model (trains on 10k MNIST images)")
    print("   2. Client has different data (30k MNIST images)")
    print("   3. Each round:")
    print("      • Server sends current model → Client")
    print("      • Client trains on private data")
    print("      • Server blends: W_new = (1-α)·W_server + α·W_client")
    print("   4. Server model improves iteratively!")
    print()
    print("=" * 70)
    
    # Configuration
    ALPHA = 0.5  # 50% server, 50% client
    NUM_ROUNDS = 5
    
    print(f"\n⚙️  Configuration:")
    print(f"   Alpha (α) = {ALPHA}")
    print(f"   Rounds = {NUM_ROUNDS}")
    print(f"   Min clients = 1 (single client mode)")
    print()
    
    # Start server in background
    server_thread = threading.Thread(
        target=run_server, 
        args=(ALPHA, NUM_ROUNDS),
        daemon=True
    )
    server_thread.start()
    
    # Wait for server to start and create baseline
    print("⏳ Waiting for server to initialize (15 seconds)...")
    time.sleep(15)
    
    # Start client
    client_thread = threading.Thread(target=run_client, daemon=True)
    client_thread.start()
    
    # Wait for completion
    print("\n🔄 Federated learning in progress...\n")
    server_thread.join()
    client_thread.join()
    
    print("\n" + "=" * 70)
    print("🎉 FEDERATED LEARNING COMPLETED!")
    print("=" * 70)
    print("📁 Check these files:")
    print("   • baseline_model.pth - Server's initial model")
    print("   • models/blended_model_round_*.pth - Each round's blended model")
    print("   • models/final_blended_model.pth - Final improved model")
    print()
    print("💡 Try different alpha values:")
    print("   • α = 0.1 → Server trusts itself more (slow learning)")
    print("   • α = 0.5 → Equal blend (balanced)")
    print("   • α = 0.9 → Trust client more (fast learning)")
    print("=" * 70)

if __name__ == "__main__":
    main()


