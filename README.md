# 🚂 Federated Learning on Railway

This project implements federated learning using Flower and PyTorch, deployed on Railway.

## 🏗️ Architecture

- **Server**: Aggregates model parameters from clients
- **Clients**: Train on local data and send updates to server
- **Models**: Saved after each round and final model

## 🚀 Multi-Service Deployment

### Prerequisites
- Railway account
- Docker (for local testing)

### Option 1: Railway Dashboard (Recommended)

1. **Go to [railway.app](https://railway.app)**
2. **Create New Project** → **Deploy from GitHub**
3. **Deploy 3 Services**:

#### 🏠 Server Service
- **Name**: `federated-server`
- **Command**: `python server_with_save.py --server-address 0.0.0.0:$PORT --num-rounds 3`
- **Environment**:
  - `PORT=8080`
  - `MODEL_SAVE_PATH=/app/models`

#### 👥 Client 0 Service
- **Name**: `federated-client-0`
- **Command**: `python client.py --client-id 0 --server-address $SERVER_URL --cloud-mode`
- **Environment**:
  - `SERVER_URL=<server-url>:8080`

#### 👥 Client 1 Service
- **Name**: `federated-client-1`
- **Command**: `python client.py --client-id 1 --server-address $SERVER_URL --cloud-mode`
- **Environment**:
  - `SERVER_URL=<server-url>:8080`

### Option 2: Railway CLI

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login to Railway
railway login

# 3. Deploy all services
chmod +x deploy-cli.sh
./deploy-cli.sh
```

### Environment Variables

**Server Service:**
```
PORT=8080
MODEL_SAVE_PATH=/app/models
```

**Client Services:**
```
SERVER_URL=your-server.railway.app:8080
```

## 🔧 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python server_with_save.py --server-address 0.0.0.0:8080

# Run clients (in separate terminals)
python client.py --client-id 0 --server-address 127.0.0.1:8080
python client.py --client-id 1 --server-address 127.0.0.1:8080
```

## 📊 Monitoring

- Check Railway logs for training progress
- Models are saved to `/app/models/` directory
- Final model: `final_federated_model.pth`

## 🎯 Features

- ✅ Federated averaging
- ✅ Model saving
- ✅ Error handling
- ✅ Cloud deployment
- ✅ Docker support

## 🚂 Railway Deployment Steps

1. **Create Railway account** at [railway.app](https://railway.app)
2. **Create new project** and connect to GitHub
3. **Set environment variables** in Railway dashboard
4. **Deploy automatically** from GitHub
5. **Monitor training** through Railway logs

## 📁 File Structure

```
├── Dockerfile                    # Docker configuration
├── requirements.txt              # Python dependencies
├── .dockerignore                 # Docker ignore file
├── railway.json                  # Railway configuration
├── railway-server.json           # Server service config
├── railway-client0.json          # Client 0 service config
├── railway-client1.json          # Client 1 service config
├── railway-compose.yml           # Railway compose file
├── deploy-cli.sh                 # CLI deployment script
├── deploy-multi-service.sh       # Multi-service deployment
├── deploy-dashboard.md           # Dashboard deployment guide
├── run_federated.py              # Single-process runner
├── start.sh                      # Startup script
├── README.md                     # This file
├── server_with_save.py           # Server implementation
├── client.py                     # Client implementation
├── model.py                      # Neural network model
└── models/                       # Saved models directory
```

## 🎉 Benefits

- **Easy deployment** with Docker
- **Automatic scaling** based on demand
- **Built-in monitoring** and logs
- **Environment variables** for configuration
- **Persistent storage** for models
- **Global CDN** for fast access