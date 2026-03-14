# DDoS Attack Detector — Render Deployment Guide

## 🚀 Deploy to Render in 5 Minutes

### Step 1: Prepare Your Repository

Ensure these files are in your repo root:
- ✅ `Procfile` — how to run the app
- ✅ `runtime.txt` — Python version
- ✅ `requirements.txt` — dependencies
- ✅ `app.py` — main FastAPI app
- ✅ `saved_model/` — BiLSTM weights & scaler
- ✅ `templates/` — HTML frontend

### Step 2: Push to GitHub

```bash
git init
git add .
git commit -m "Add DDoS detector for Render"
git remote add origin https://github.com/YOUR_USERNAME/ddos-detector
git push -u origin main
```

### Step 3: Deploy on Render

1. **Go to [render.com](https://render.com)**
2. **Sign in** with GitHub
3. **Click "New +"** → Select **"Web Service"**
4. **Connect your repo**
5. **Fill in settings:**
   - **Name:** `ddos-detector` (or any name)
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** (Leave blank, it will auto-detect Procfile)
   - **Instance Type:** `Standard` (free tier available)

6. **Click "Deploy"**
7. **Wait 2-3 minutes** for deployment to complete

### Step 4: Access Your App

Once deployed, you'll get a URL like:
```
https://ddos-detector.onrender.com
```

Open it in your browser!

---

## 📊 What Gets Deployed

| Component | Size | Status |
|-----------|------|--------|
| Model weights | ~45 MB | ✅ Included |
| Scaler & metadata | ~2 MB | ✅ Included |
| FastAPI app | <1 MB | ✅ Included |
| Frontend HTML | <100 KB | ✅ Included |
| **Total** | **~50 MB** | ✅ OK for Render |

---

## ⚙️ Configuration

**Environment Variables (Optional):**
If you want to use environment variables in the future, add them in Render dashboard:
- `Settings` → `Environment` → Add variables

**Current app doesn't need any env vars** — it works out of the box!

---

## 🔄 Auto-Redeploy on GitHub Push

Once connected, every time you push to `main`:
```bash
git push origin main
```
Render automatically redeploys your app! ✅

---

## ⚠️ Important Notes

1. **First request may be slow** — Model loads on startup (~10s)
2. **Free tier**: Service spins down after 15 mins of inactivity
3. **Cold start**: First request after spin-down takes ~30s
4. **To keep it always on**: Upgrade to Paid plan ($7/month)

---

## 🛠️ Troubleshooting

**Deployment failed?**
1. Check Logs: Render → Select your service → Logs
2. Verify Procfile syntax (no spaces before `web:`)
3. Ensure `requirements.txt` has all dependencies

**App is slow?**
1. Model is loading — normal on first request
2. Upgrade to paid instance for persistent memory

**Files not found?**
1. Check `.gitignore` isn't excluding important files
2. Verify `saved_model/` folder is committed

---

## 📱 Using Your Deployed App

**Upload CSV & Run Inference:**
1. Open your Render URL
2. Drop a CICFlowMeter CSV from archive (1) or (2)
3. Click "Run Inference"
4. View results, metrics, confusion matrix

**API Endpoints:**
- `GET /` — Web UI
- `GET /api/status` — Model status
- `POST /api/predict` — Run inference (form: file + max_rows)

---

**Your app is ready to deploy!** 🎉

Questions? Check [Render docs](https://docs.render.com)
