#!/bin/bash
# Quick deployment script for Railway

echo "🚂 Railway Deployment Helper"
echo "============================="
echo ""

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found!"
    echo ""
    echo "Install it with:"
    echo "  npm install -g @railway/cli"
    echo "  or"
    echo "  brew install railway"
    echo ""
    exit 1
fi

echo "✅ Railway CLI found"
echo ""

# Login check
echo "Checking Railway authentication..."
if ! railway whoami &> /dev/null; then
    echo "Please login to Railway:"
    railway login
fi

echo "✅ Authenticated"
echo ""

# Initialize or link project
echo "Setting up Railway project..."
if [ ! -f "railway.toml" ]; then
    railway init
else
    echo "✅ Railway project already configured"
fi

echo ""
echo "Deploying application..."
railway up

echo ""
echo "Setting environment variables..."
railway variables set FL_CONFIG=configs/mnist_config.py
railway variables set NUM_ROUNDS=5
railway variables set ALPHA=0.5
railway variables set MIN_CLIENTS=2

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Get your server address:"
echo "  railway status"
echo ""
echo "View logs:"
echo "  railway logs"
echo ""
echo "🎉 Your FL server is now running in the cloud!"
