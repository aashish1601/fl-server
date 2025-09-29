#!/bin/bash
# deploy-multi-service.sh - Deploy federated learning with separate services

echo "🚂 Deploying Multi-Service Federated Learning to Railway"
echo "========================================================"

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Login to Railway
echo "🔐 Logging into Railway..."
railway login

# Initialize project if not already done
if [ ! -f ".railway/project.json" ]; then
    echo "🏗️ Initializing Railway project..."
    railway init
fi

echo ""
echo "🚀 Step 1: Deploying Server Service"
echo "===================================="
echo "📋 Server Configuration:"
echo "   - Service: federated-server"
echo "   - Command: python server_with_save.py --server-address 0.0.0.0:\$PORT --num-rounds 3"
echo "   - Environment: MODEL_SAVE_PATH=/app/models, PORT=8080"
echo ""

# Deploy server
echo "🏠 Deploying server..."
railway up --service federated-server --detach

# Wait for server to deploy
echo "⏳ Waiting for server to deploy (30 seconds)..."
sleep 30

# Get server URL
echo "🔍 Getting server URL..."
SERVER_URL=$(railway status --service federated-server --json | jq -r '.url' 2>/dev/null || echo "federated-server-production.up.railway.app")

if [ "$SERVER_URL" = "null" ] || [ -z "$SERVER_URL" ]; then
    echo "⚠️ Could not get server URL automatically. Please set SERVER_URL manually."
    echo "📝 You can find the server URL in the Railway dashboard."
    read -p "Enter server URL (without port): " SERVER_URL
fi

echo "🌐 Server URL: $SERVER_URL"
echo ""

echo "🚀 Step 2: Deploying Client Services"
echo "===================================="

# Update client configurations with server URL
echo "📝 Updating client configurations with server URL: $SERVER_URL:8080"

# Update railway-client0.json
sed -i "s/federated-server-production.up.railway.app:8080/$SERVER_URL:8080/g" railway-client0.json
sed -i "s/federated-server-production.up.railway.app:8080/$SERVER_URL:8080/g" railway-client1.json

echo "👥 Deploying Client 0..."
railway up --service federated-client-0 --detach

echo "👥 Deploying Client 1..."
railway up --service federated-client-1 --detach

echo ""
echo "📊 Checking deployment status..."
railway status

echo ""
echo "🎉 Multi-Service Deployment Complete!"
echo "====================================="
echo "🏠 Server: federated-server"
echo "👥 Client 0: federated-client-0"
echo "👥 Client 1: federated-client-1"
echo ""
echo "📋 Next Steps:"
echo "1. Check Railway dashboard for service status"
echo "2. Monitor logs: railway logs --service <service-name>"
echo "3. Download models from server when training completes"
echo ""
echo "🔗 Railway Dashboard: https://railway.app/dashboard"
