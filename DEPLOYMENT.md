# DEM Visualiser Web Tool - Deployment Guide

## 🌐 Live Demo
Your app will be hosted at: `https://dem-visualiser.onrender.com` (or custom URL)

## 🚀 Deploy to Render (Recommended - FREE)

### Option 1: Auto-Deploy from GitHub (Easiest)
1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account and select **`joycemalik/DEMVizWebtool`**
4. Render will auto-detect `render.yaml` and configure everything
5. Click **"Create Web Service"**
6. Wait 5-10 minutes for build to complete
7. Your app will be live! 🎉

### Option 2: Manual Configuration
If auto-detect doesn't work:
- **Name**: `dem-visualiser`
- **Environment**: `Python 3`
- **Build Command**: `./build.sh`
- **Start Command**: `cd "DEM Visualiser Web Tool" && gunicorn app:app`
- **Plan**: `Free`

## 🔧 Environment Variables (Optional)
If you need to configure:
- `PYTHON_VERSION`: `3.11.10`
- `PORT`: Auto-assigned by Render

## 📦 What's Included
- ✅ `Procfile` - Process configuration
- ✅ `render.yaml` - Render deployment config
- ✅ `build.sh` - Build script
- ✅ `runtime.txt` - Python version
- ✅ Updated `requirements.txt` with gunicorn

## 🌍 Alternative Hosting Options

### Railway (Also Free)
1. Go to [Railway](https://railway.app/)
2. Click **"Start a New Project"** → **"Deploy from GitHub repo"**
3. Select `joycemalik/DEMVizWebtool`
4. Railway auto-detects and deploys

### Heroku (Paid after free tier removal)
```bash
heroku create dem-visualiser
git push heroku main
```

### PythonAnywhere (Free with limitations)
- Upload files via web interface
- Configure WSGI manually
- Limited to 512MB storage on free tier

## ⚠️ Important Notes
- **File Uploads**: Render's free tier has ephemeral storage (files reset on restart)
- **Cold Starts**: First request after inactivity may take 30-60 seconds
- **Memory Limit**: 512MB RAM on free tier
- For production use with persistent storage, consider upgrading to paid tier

## 🐛 Troubleshooting
If deployment fails:
1. Check build logs in Render dashboard
2. Ensure all paths use forward slashes
3. Verify Python version compatibility
4. Check that rasterio builds successfully (may take 5-10 min)

## 📞 Support
Issues? Check:
- [Render Docs](https://render.com/docs)
- [GitHub Issues](https://github.com/joycemalik/DEMVizWebtool/issues)
