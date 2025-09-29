#!/bin/bash
# deploy-cli.sh - Railway CLI deployment for multi-service federated learning

echo "🚂 Railway CLI Multi-Service Deployment"
echo "======================================="

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Check if jq is installed (for JSON parsing)
if ! command -v jq &> /dev/null; then
    echo "❌ jq not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install jq
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y jq
    else
        echo "Please install jq manually: https://stedolan.github.io/jq/"
        exit 1
    fi
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
echo "🚀 Deploying Server Service..."
echo "=============================="

# Deploy server with specific configuration
railway up --service federated-server --detach

echo "⏳ Waiting for server to deploy (30 seconds)..."
sleep 30

# Get server URL
echo "🔍 Getting server URL..."
SERVER_URL=$(railway status --service federated-server --json 2>/dev/null | jq -r '.url' 2>/dev/null || echo "")

if [ -z "$SERVER_URL" ] || [ "$SERVER_URL" = "null" ]; then
    echo "⚠️ Could not get server URL automatically."
    echo "📝 Please check Railway dashboard for server URL."
    read -p "Enter server URL (without port): " SERVER_URL
fi

echo "🌐 Server URL: $SERVER_URL"
echo ""

echo "🚀 Deploying Client Services..."
echo "==============================="

# Deploy client 0
echo "👥 Deploying Client 0..."
railway up --service federated-client-0 --detach

# Deploy client 1
echo "👥 Deploying Client 1..."
railway up --service federated-client-1 --detach

echo ""
echo "📊 Deployment Status:"
echo "===================="
railway status

echo ""
echo "🎉 Multi-Service Deployment Complete!"
echo "====================================="
echo "🏠 Server: federated-server"
echo "👥 Client 0: federated-client-0" 
echo "👥 Client 1: federated-client-1"
echo ""
echo "📋 Monitor Training:"
echo "railway logs --service federated-server"
echo "railway logs --service federated-client-0"
echo "railway logs --service federated-client-1"
echo ""
echo "🔗 Railway Dashboard: https://railway.app/dashboard"
