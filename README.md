# 🚂 Federated Learning on Railway

This project implements federated learning using Flower and PyTorch, deployed on Railway.

## 🏗️ Architecture

- **Server**: Aggregates model parameters from clients
- **Clients**: Train on local data and send updates to server
- **Models**: Saved after each round and final model

## 🚀 Deployment

### Prerequisites
- Railway account
- Docker (for local testing)

### Deploy to Railway

1. **Fork this repository**
2. **Connect to Railway**
3. **Deploy automatically**

### Environment Variables

Set these in Railway dashboard:

```
PORT=8080
MODEL_SAVE_PATH=/app/models
NUM_ROUNDS=5
SERVER_ADDRESS=your-server.railway.app:8080
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
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
├── .dockerignore          # Docker ignore file
├── railway.json           # Railway configuration
├── railway.toml           # Railway TOML config
├── railway-compose.yml    # Railway compose file
├── start.sh               # Startup script
├── README.md              # This file
├── server_with_save.py    # Server implementation
├── client.py              # Client implementation
├── model.py               # Neural network model
└── models/                # Saved models directory
```

## 🎉 Benefits

- **Easy deployment** with Docker
- **Automatic scaling** based on demand
- **Built-in monitoring** and logs
- **Environment variables** for configuration
- **Persistent storage** for models
- **Global CDN** for fast access