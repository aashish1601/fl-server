# 🚀 Deploy to Render.com (Free Alternative)

## ✅ **Why Render?**
- ✅ **100% FREE** (no credit card needed)
- ✅ Supports Docker & long-running processes
- ✅ Perfect for federated learning servers
- ✅ Easy deployment from GitHub

---

## 📋 **Quick Deploy (2 Minutes)**

### **Step 1: Push to GitHub**
```bash
git init
git add .
git commit -m "FL server for Render"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fl-server.git
git push -u origin main
```

### **Step 2: Deploy to Render**

1. Go to: **https://dashboard.render.com**
2. Click **"New +"** → **"Blueprint"**
3. Connect your GitHub repository
4. Render auto-detects `render.yaml` ✅
5. Click **"Apply"**

### **Step 3: Wait for Build**
Render will:
- Build your Docker container
- Deploy the FL server
- Assign a public URL

### **Step 4: Get Server Address**
Render shows:
```
Service URL: https://fl-mnist-server.onrender.com
```

**Important:** For Flower, you need to connect via **TCP**. Render provides both:
- HTTPS: `fl-mnist-server.onrender.com` (port 443)
- Direct: Use the URL shown in Render dashboard

---

## 🔧 **Alternative: Manual Deployment**

### **Without render.yaml:**

1. Go to https://dashboard.render.com
2. Click **"New +"** → **"Web Service"**
3. Select **"Deploy from Git"**
4. Choose your repository
5. Configure:
   ```
   Name: fl-mnist-server
   Runtime: Docker
   Dockerfile Path: ./Dockerfile
   Plan: Free
   ```
6. Add **Environment Variables**:
   ```
   FL_CONFIG    = configs/mnist_config.py
   NUM_ROUNDS   = 5
   ALPHA        = 0.5
   MIN_CLIENTS  = 2
   PORT         = 8080
   ```
7. Click **"Create Web Service"**

---

## 👥 **Connect Clients**

Once deployed, run clients locally:

```powershell
# Windows
.\venv\Scripts\Activate.ps1
python client_generic.py `
    --config configs/mnist_config.py `
    --server-address fl-mnist-server.onrender.com:443 `
    --client-id 0
```

```bash
# Mac/Linux
source venv/bin/activate
python client_generic.py \
    --config configs/mnist_config.py \
    --server-address fl-mnist-server.onrender.com:443 \
    --client-id 0
```

---

## 📊 **Monitor Deployment**

View logs in Render dashboard:
```
🌐 GENERIC FEDERATED LEARNING SERVER
⏳ Waiting for 2 clients...
✅ 2 clients connected!
📊 Round 1 - Blending models...
💾 Saved model
```

---

## 💰 **Render Free Tier**

- ✅ **Completely FREE**
- ✅ No credit card required
- ✅ 750 hours/month (enough for 24/7)
- ✅ Supports Docker
- ✅ Auto-deploys from GitHub

**Limitations:**
- Spins down after 15 min of inactivity
- Takes ~30 seconds to wake up
- 512 MB RAM (enough for MNIST)

---

## 🎯 **Other Free Options**

| Platform | Free Tier | Docker | Best For |
|----------|-----------|--------|----------|
| **Render** | ✅ Yes | ✅ Yes | FL servers (RECOMMENDED) |
| **Fly.io** | ✅ Yes | ✅ Yes | Low latency, multiple regions |
| **Railway** | ⚠️ Needs verification | ✅ Yes | After adding payment method |
| **Heroku** | ❌ No free tier | ✅ Yes | Paid only now |
| **Vercel** | ❌ Not suitable | ❌ No | Only for serverless/static |

---

## 🚀 **Deploy Now!**

**Recommended path:**
1. Push to GitHub
2. Deploy via Render.com (free, no card needed)
3. Run clients locally
4. Start federated learning!

---

## ✅ **Next Steps**

1. Push code to GitHub
2. Visit: https://dashboard.render.com
3. Click "New +" → "Blueprint"
4. Select your repo → "Apply"
5. Wait 2-3 minutes for build
6. Get server URL
7. Run clients!

**That's it!** 🎉

