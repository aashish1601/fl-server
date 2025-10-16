# 🚂 Deploy to Railway - 2 Minute Guide

## ✅ Prerequisites
- Railway account (free): https://railway.app
- GitHub account

---

## 🚀 **OPTION 1: Deploy via Railway Dashboard (Easiest)**

### **Step 1: Push to GitHub**
```bash
# In your project folder
git init
git add .
git commit -m "FL server ready for deployment"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin master
```

### **Step 2: Deploy to Railway**

1. Go to https://railway.app/new
2. Click **"Deploy from GitHub repo"**
3. Select your repository
4. Railway auto-detects `Dockerfile` ✅
5. Click **"Deploy"**

### **Step 3: Configure Environment Variables**

In Railway dashboard → **"Variables"** tab, add:

| Variable | Value | Description |
|----------|-------|-------------|
| `FL_CONFIG` | `configs/mnist_config.py` | Which model to train |
| `NUM_ROUNDS` | `5` | Number of FL rounds |
| `ALPHA` | `0.5` | Server-client blending |
| `MIN_CLIENTS` | `2` | Clients needed to start |

Click **"Deploy"** again.

### **Step 4: Get Your Server Address**

Railway shows:
```
✅ Service running!

HTTP: https://yourapp.up.railway.app  ← ignore
TCP:  tcp://containers.rail.app:12345 ← USE THIS!
```

Copy the **TCP address**.

---

## 🖥️ **OPTION 2: Deploy via Railway CLI (Power Users)**

### **Step 1: Install Railway CLI**
```bash
npm install -g @railway/cli
# or
brew install railway
```

### **Step 2: Login**
```bash
railway login
```

### **Step 3: Deploy**
```bash
# In your project folder
railway init
railway up

# Set environment variables
railway variables set FL_CONFIG=configs/mnist_config.py
railway variables set NUM_ROUNDS=5
railway variables set ALPHA=0.5
railway variables set MIN_CLIENTS=2

# Check status
railway status
```

### **Step 4: Get TCP Address**
```bash
railway domain
# Shows: tcp://containers.<id>.railway.app:xxxxx
```

---

## 👥 **Connect Clients from Local Machine**

Once deployed, run clients locally:

```bash
# Activate venv
.\venv\Scripts\Activate.ps1  # Windows
# or
source venv/bin/activate     # Mac/Linux

# Run client
python client_generic.py \
    --config configs/mnist_config.py \
    --server-address containers.<id>.railway.app:12345 \
    --client-id 0
```

Run this on **2+ machines** (or terminals) with different `--client-id` values!

---

## 🎯 **Deploy Multiple Models**

Want MNIST **and** CIFAR-10 servers?

### **Create Second Service:**
1. Railway dashboard → **"New"** → **"GitHub Repo"**
2. Select **same repo**
3. Set **different environment variables**:

```
Service 1 (MNIST):
  FL_CONFIG = configs/mnist_config.py
  TCP: tcp://containers.abc123.railway.app:11111

Service 2 (CIFAR-10):
  FL_CONFIG = configs/cifar10_config.py  
  TCP: tcp://containers.xyz789.railway.app:22222
```

Now you have **two independent FL servers**! 🎉

---

## 📊 **Monitor Logs**

```bash
# Via CLI
railway logs

# Or in dashboard → "Deployments" → "View Logs"
```

You'll see:
```
🌐 GENERIC FEDERATED LEARNING SERVER
⏳ Waiting for clients to connect...
✅ 2 clients connected! Starting training...
📊 Round 1 - Received 2 client(s)
🔀 Blending models...
💾 Saved model to './models/Net_round_1.pth'
```

---

## 🔧 **Troubleshooting**

### **"Service crashed"**
- Check logs: `railway logs`
- Verify `requirements.txt` has all dependencies
- Make sure `Dockerfile` is correct

### **"Can't connect from client"**
- Use **TCP** address, not HTTP
- Check firewall allows outbound TCP
- Verify `--server-address` matches Railway's TCP URL

### **"Waiting for clients forever"**
- Check `MIN_CLIENTS` matches number of clients you're running
- Clients must use same config as server
- Verify clients can reach the TCP port

---

## 💰 **Cost**

Railway free tier includes:
- ✅ $5 free credit/month
- ✅ ~500 hours runtime
- ✅ Enough for testing/small projects

For production:
- ~$0.01/hour for basic container
- ~$7/month for 24/7 server

---

## 🎉 **Success Checklist**

- [ ] Pushed code to GitHub
- [ ] Created Railway project
- [ ] Set environment variables
- [ ] Got TCP address
- [ ] Ran 2+ local clients
- [ ] Saw "Round 1 completed" in logs
- [ ] Downloaded trained model

---

## 🚀 **You're Done!**

Your FL server is now:
- ✅ Running in the cloud (Railway)
- ✅ Accessible from anywhere via TCP
- ✅ Training models with federated learning
- ✅ Saving results automatically

**Clients can connect from:**
- Your laptop
- Friend's computer
- Edge devices
- Raspberry Pi
- Anywhere with internet!

🔒 **Privacy preserved** - only model weights travel, never raw data!

