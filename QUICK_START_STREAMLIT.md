# 🎬 Quick Start Guide - Streamlit Video QA System

## ✅ What's Ready

Your Video QA System has been successfully converted to a **Streamlit web application**!

### Files Created:
- ✅ `streamlit_app.py` - Main Streamlit application
- ✅ `requirements.txt` - Updated with Streamlit & PEFT
- ✅ `.streamlit/config.toml` - UI configuration
- ✅ `STREAMLIT_DEPLOYMENT_GUIDE.md` - Complete deployment guide

---

## 🚀 Run Locally (Recommended First Step)

### Option 1: Run from PowerShell/Terminal

```bash
cd c:\Project\MTech\Project\Data
streamlit run streamlit_app.py
```

The app will open at: **http://localhost:8501**

### Option 2: Run from VS Code

1. Open integrated terminal (Ctrl + `)
2. Paste:
```bash
streamlit run c:\Project\MTech\Project\Data\streamlit_app.py
```

### What You'll See

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

---

## 🎯 Features in Your App

### 📤 Tab 1: Upload Video
- Upload MP4, AVI, MOV, or MKV files
- Real-time video preview
- Automatic analysis

### 📁 Tab 2: Local Video
- Browse videos from your system
- Select and analyze existing videos
- No upload needed

### ⚙️ Tab 3: Settings
- View current configuration
- Check model paths
- Monitor device (CPU/GPU)

---

## 🔍 Try It Now

### Example Workflow

1. **Click "Local Video" tab**
2. **Select**: `Arrest003_x264.mp4` (or any video in your directory)
3. **Click**: "🔍 Analyze Video"
4. **Wait**: ~30-60 seconds for processing
5. **See Results**:
   - Classification (Normal/Anomalous)
   - People/Weapon/Location detected
   - AI-generated summary
   - Confidence scores

### Ask Questions

After analysis, try these Q&A examples:
- ❓ "What is the status?" → Shows Normal/Anomalous
- ❓ "How many people?" → Shows people count
- ❓ "Is there a weapon?" → Shows weapon type
- ❓ "Where is this?" → Shows location
- ❓ "What's happening?" → Shows summary

---

## 📊 Performance Tips

For faster analysis:
- Use smaller videos (< 1 minute)
- Enable GPU if available
- Ensure 16GB+ RAM available
- Close other applications

---

## 🐛 If Something Goes Wrong

### Models not loading?
Check that files exist:
```
C:\Project\MTech\Project\Data\model\
  ├── binary_model.pkl ✓
  ├── model_config.pkl ✓
  ├── temporal_adapter.pt ✓
  ├── le_weapon.pkl ✓
  ├── le_location.pkl ✓
  ├── le_people.pkl ✓
  └── le_super.pkl ✓
```

### Memory error?
Increase Windows paging file to 30GB:
1. Right-click "This PC" → Properties
2. Advanced → Performance Settings
3. Advanced → Virtual Memory → Change
4. Set to 30000 MB (30GB)
5. Restart PC

### Video won't process?
- Ensure video format is supported (MP4/AVI/MOV/MKV)
- Check file isn't corrupted: `ffprobe video.mp4`
- Reduce number of frames in code

---

## 🌐 Deploy to Production

### Easy Option: Streamlit Cloud

1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Click "New app"
4. Select your repo & `streamlit_app.py`
5. Click Deploy

⏱️ Deployment takes ~5 minutes  
📍 Your app gets public URL like: `https://yourname-video-qa-system.streamlit.app`

### Advanced Option: Docker/AWS/Azure
See `STREAMLIT_DEPLOYMENT_GUIDE.md` for detailed instructions

---

## 📝 Customization

Edit `streamlit_app.py` to:
- Change title: `st.title("Your App Name")`
- Modify colors in `.streamlit/config.toml`
- Add more Q&A keywords in `answer_question()` function
- Change video directory paths

---

## 🔑 Key Commands

```bash
# Run app
streamlit run streamlit_app.py

# Run with custom port
streamlit run streamlit_app.py --server.port 8502

# Run in debug mode
streamlit run streamlit_app.py --logger.level=debug

# Clear cache
streamlit cache clear
```

---

## ✨ Next Steps

1. ✅ **Run locally** → `streamlit run streamlit_app.py`
2. ✅ **Test with videos** → Try Upload & Local Video tabs
3. ✅ **Deploy** → Push to GitHub → Deploy on Streamlit Cloud
4. ✅ **Share** → Get public URL and share with team

---

## 💡 Pro Tips

- **Cache Models**: Models are cached after first load (faster on subsequent runs)
- **Concurrent Users**: Deploy on paid Streamlit tier for multiple users
- **Videos Storage**: Store large videos outside repo for faster deployment
- **API Integration**: Can be extended to accept API requests

---

**You're all set! 🚀 Run the app now:**

```bash
streamlit run c:\Project\MTech\Project\Data\streamlit_app.py
```

Happy analyzing! 🎉
