# 🚀 Deploy in 2 Minutes - Quickstart

## ⚡ **Fastest Method: Railway Dashboard (No CLI needed)**

### **1. Push to GitHub (1 min)**
```bash
git init
git add .
git commit -m "FL server"
git branch -M main
# Create new repo at github.com/new, then:
git remote add origin https://github.com/YOUR_USERNAME/fl-server.git
git push -u origin main
```

### **2. Deploy to Railway (30 seconds)**
1. Go to: https://railway.app/new
2. Click **"Deploy from GitHub repo"**
3. Select your repo
4. Click **"Deploy"**

Railway auto-detects the `Dockerfile` ✅

### **3. Set Variables (30 seconds)**
In Railway dashboard → **Variables** tab:
```
FL_CONFIG    = configs/mnist_config.py
NUM_ROUNDS   = 5
ALPHA        = 0.5
MIN_CLIENTS  = 2
```
Click **"Redeploy"**

### **4. Get Server Address (10 seconds)**
Railway shows:
```
TCP: tcp://containers.railway.app:12345
```
**Copy this address!**

---

## 👥 **Run Local Clients**

On your computer (or any computer):
```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1
python client_generic.py --config configs/mnist_config.py --server-address containers.railway.app:12345 --client-id 0

# Mac/Linux
source venv/bin/activate
python client_generic.py --config configs/mnist_config.py --server-address containers.railway.app:12345 --client-id 0
```

Run on **2 different computers** (or 2 terminals) with different `--client-id` values!

---

## ✅ **Success!**

Watch the Railway logs:
```
✅ 2 clients connected!
📊 Round 1 - Blending models...
💾 Saved model
📊 Round 2 - Blending models...
```

**You now have federated learning in the cloud!** 🎉

---

## 🔧 **Alternative: Railway CLI Method**

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login & deploy
railway login
railway init
railway up

# Set variables
railway variables set FL_CONFIG=configs/mnist_config.py
railway variables set NUM_ROUNDS=5
railway variables set ALPHA=0.5
railway variables set MIN_CLIENTS=2

# Get address
railway status
```

---

## 📊 **Deploy Different Models**

Repeat steps 2-4 but set different `FL_CONFIG`:

| Service | FL_CONFIG | Use Case |
|---------|-----------|----------|
| Service 1 | `configs/mnist_config.py` | MNIST digits |
| Service 2 | `configs/cifar10_config.py` | CIFAR-10 images |
| Service 3 | `configs/custom_cnn_config.py` | Your custom model |

Each gets its own TCP address!

---

## 🎯 **Files Created for You**

- ✅ `Dockerfile` - Container config
- ✅ `railway.toml` - Railway config
- ✅ `.dockerignore` - Exclude unnecessary files
- ✅ `DEPLOY_RAILWAY.md` - Full detailed guide
- ✅ `deploy.sh` / `deploy.ps1` - Automated scripts

---

## 💡 **Next Steps**

1. **Deploy** (2 minutes using this guide)
2. **Run 2+ clients** from different machines
3. **Watch federated learning** happen in real-time
4. **Download trained model** from Railway logs

**That's it!** 🚀

