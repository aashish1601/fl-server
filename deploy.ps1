# Quick deployment script for Railway (Windows PowerShell)

Write-Host "🚂 Railway Deployment Helper" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan
Write-Host ""

# Check if Railway CLI is installed
$railwayInstalled = Get-Command railway -ErrorAction SilentlyContinue
if (-not $railwayInstalled) {
    Write-Host "❌ Railway CLI not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Install it with:"
    Write-Host "  npm install -g @railway/cli" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Host "✅ Railway CLI found" -ForegroundColor Green
Write-Host ""

# Login check
Write-Host "Checking Railway authentication..."
$loginCheck = railway whoami 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Please login to Railway:" -ForegroundColor Yellow
    railway login
}

Write-Host "✅ Authenticated" -ForegroundColor Green
Write-Host ""

# Initialize or link project
Write-Host "Setting up Railway project..."
if (-not (Test-Path "railway.toml")) {
    railway init
} else {
    Write-Host "✅ Railway project already configured" -ForegroundColor Green
}

Write-Host ""
Write-Host "Deploying application..." -ForegroundColor Cyan
railway up

Write-Host ""
Write-Host "Setting environment variables..." -ForegroundColor Cyan
railway variables set FL_CONFIG=configs/mnist_config.py
railway variables set NUM_ROUNDS=5
railway variables set ALPHA=0.5
railway variables set MIN_CLIENTS=2

Write-Host ""
Write-Host "✅ Deployment complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Get your server address:" -ForegroundColor Yellow
Write-Host "  railway status"
Write-Host ""
Write-Host "View logs:" -ForegroundColor Yellow
Write-Host "  railway logs"
Write-Host ""
Write-Host "🎉 Your FL server is now running in the cloud!" -ForegroundColor Green

