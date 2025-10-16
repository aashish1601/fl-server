# 🚀 Deploy to Fly.io (Another Free Option)

## ✅ **Why Fly.io?**
- ✅ **FREE tier** (3 shared CPU VMs)
- ✅ Fast global deployment
- ✅ Perfect for long-running processes
- ✅ True TCP support (better for Flower)

---

## 📋 **Quick Deploy via Fly.io CLI**

### **Step 1: Install Fly CLI**

**Windows (PowerShell):**
```powershell
iwr https://fly.io/install.ps1 -useb | iex
```

**Mac/Linux:**
```bash
curl -L https://fly.io/install.sh | sh
```

### **Step 2: Login**
```bash
fly auth login
```
(Opens browser for authentication)

### **Step 3: Create Fly Configuration**
```bash
fly launch --no-deploy
```

Answer prompts:
```
? App Name: fl-mnist-server
? Region: Choose closest to you
? Would you like to setup a database? No
? Would you like to deploy now? No
```

This creates `fly.toml`.

### **Step 4: Update fly.toml**
The file was created - now let me customize it:

