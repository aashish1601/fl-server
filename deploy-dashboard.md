# 🚂 Railway Dashboard Deployment Guide

## Step-by-Step Multi-Service Deployment

### 1. Prepare Repository
```bash
git add .
git commit -m "Add multi-service Railway deployment"
git push origin main
```

### 2. Create Railway Project
1. Go to [railway.app](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository

### 3. Deploy Server Service

#### Service Configuration:
- **Service Name**: `federated-server`
- **Start Command**: `python server_with_save.py --server-address 0.0.0.0:$PORT --num-rounds 3`
- **Environment Variables**:
  - `PORT=8080`
  - `MODEL_SAVE_PATH=/app/models`

#### Steps:
1. Click "Add Service" → "GitHub Repo"
2. Select your repository
3. Set service name to `federated-server`
4. Go to "Settings" → "Deploy"
5. Set start command: `python server_with_save.py --server-address 0.0.0.0:$PORT --num-rounds 3`
6. Go to "Variables" tab
7. Add variables:
   - `PORT` = `8080`
   - `MODEL_SAVE_PATH` = `/app/models`
8. Click "Deploy"

### 4. Get Server URL
After server deploys, note the URL (e.g., `federated-server-production.up.railway.app`)

### 5. Deploy Client 0 Service

#### Service Configuration:
- **Service Name**: `federated-client-0`
- **Start Command**: `python client.py --client-id 0 --server-address $SERVER_URL --cloud-mode`
- **Environment Variables**:
  - `SERVER_URL=<your-server-url>:8080`

#### Steps:
1. Click "Add Service" → "GitHub Repo"
2. Select your repository
3. Set service name to `federated-client-0`
4. Go to "Settings" → "Deploy"
5. Set start command: `python client.py --client-id 0 --server-address $SERVER_URL --cloud-mode`
6. Go to "Variables" tab
7. Add variable:
   - `SERVER_URL` = `<your-server-url>:8080`
8. Click "Deploy"

### 6. Deploy Client 1 Service

#### Service Configuration:
- **Service Name**: `federated-client-1`
- **Start Command**: `python client.py --client-id 1 --server-address $SERVER_URL --cloud-mode`
- **Environment Variables**:
  - `SERVER_URL=<your-server-url>:8080`

#### Steps:
1. Click "Add Service" → "GitHub Repo"
2. Select your repository
3. Set service name to `federated-client-1`
4. Go to "Settings" → "Deploy"
5. Set start command: `python client.py --client-id 1 --server-address $SERVER_URL --cloud-mode`
6. Go to "Variables" tab
7. Add variable:
   - `SERVER_URL` = `<your-server-url>:8080`
8. Click "Deploy"

### 7. Monitor Training

#### Check Logs:
- **Server Logs**: Click on `federated-server` → "Logs"
- **Client 0 Logs**: Click on `federated-client-0` → "Logs"
- **Client 1 Logs**: Click on `federated-client-1` → "Logs"

#### Expected Output:
```
🏠 Server: Starting Flower Server with Model Saving
👥 Client 0: Starting Flower Client 0
👥 Client 1: Starting Flower Client 1
🔄 Round 1: Training in progress...
💾 Saved federated model from round 1
🔄 Round 2: Training in progress...
💾 Saved federated model from round 2
🔄 Round 3: Training in progress...
💾 Saved federated model from round 3
🏆 Final federated model saved
```

### 8. Download Models
Models are saved on the server service. You can access them through Railway's file system or add a download endpoint.

## Troubleshooting

### Common Issues:
1. **Connection Failed**: Check SERVER_URL is correct
2. **Client Timeout**: Ensure server is running first
3. **Build Failed**: Check requirements.txt and Dockerfile

### Debug Commands:
```bash
# Check service status
railway status

# View logs
railway logs --service federated-server
railway logs --service federated-client-0
railway logs --service federated-client-1
```
