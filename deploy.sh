#!/bin/bash
# deploy.sh - Railway deployment script

echo "🚂 Deploying Federated Learning to Railway"

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

# Deploy to Railway
echo "🚀 Deploying to Railway..."
railway up

# Check deployment status
echo "📊 Checking deployment status..."
railway status

# Show logs
echo "📋 Recent logs:"
railway logs --tail 50

echo "🎉 Deployment complete!"
echo "🌐 Your federated learning system is now running on Railway!"
