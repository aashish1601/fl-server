#!/bin/bash
# start.sh - Railway deployment script

echo "🚂 Starting Federated Learning on Railway"

# Create models directory
mkdir -p /app/models

# Start server
echo "🏠 Starting server..."
python server_with_save.py --server-address 0.0.0.0:$PORT &

# Wait for server to start
sleep 10

# Start clients
echo "👥 Starting clients..."
python client.py --client-id 0 --server-address 0.0.0.0:$PORT --cloud-mode &
python client.py --client-id 1 --server-address 0.0.0.0:$PORT --cloud-mode &

# Wait for training to complete
wait

echo "🎉 Federated learning completed!"
