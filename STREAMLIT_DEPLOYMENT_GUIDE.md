# 🚀 Streamlit Deployment Guide

## Overview
Your Video QA System has been converted to a Streamlit web application. This guide covers how to run and deploy it.

---

## 📋 Prerequisites

- Python 3.9+ (3.10+ recommended)
- 16GB+ RAM (8GB minimum)
- CUDA 11.8+ (for GPU acceleration)
- Models in `C:\Project\MTech\Project\Data\model\`
- Checkpoint in `C:\Project\MTech\Project\Data\checkpoint-140\`

---

## 🏃 Quick Start (Local Testing)

### 1. Install Dependencies

```bash
cd c:\Project\MTech\Project\Data
pip install -r requirements.txt
```

### 2. Run Locally

```bash
streamlit run streamlit_app.py
```

This opens the app at `http://localhost:8501`

### 3. Test the App

- **Upload Tab**: Upload a video file directly
- **Local Video Tab**: Select videos from `C:\Project\MTech\Project\Data\Anomaly-Videos-Part-1\`
- **Ask Questions**: Use the Q&A feature to query results

---

## 🌐 Deploy to Streamlit Cloud (Recommended)

### Step 1: Prepare GitHub Repository

```bash
cd c:\Project\MTech\Project\Data
git init
git add streamlit_app.py requirements.txt
git commit -m "Add Streamlit Video QA App"
git remote add origin https://github.com/YOUR_USERNAME/video-qa-system.git
git push -u origin main
```

### Step 2: Create `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#FF6B35"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableRunOnSave = true
```

### Step 3: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click **"New app"**
3. Choose your GitHub repository
4. Select `streamlit_app.py` as main file
5. Click **Deploy**

**⚠️ NOTE:** Streamlit Cloud has limitations:
- Free tier: 1GB RAM (may not be enough for large models)
- No persistent storage for downloaded models
- Requires public GitHub repo

---

## 🖥️ Deploy to Custom Server (AWS/Azure/Local)

### Option A: Windows Server

1. Install Python 3.10+ on the server
2. Copy project files
3. Create venv: `python -m venv venv`
4. Activate: `.\venv\Scripts\activate`
5. Install: `pip install -r requirements.txt`
6. Run: `streamlit run streamlit_app.py --server.port 8501`
7. Access at `http://SERVER_IP:8501`

### Option B: Docker (Recommended for Cloud)

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]
```

Build and run:

```bash
docker build -t video-qa-system .
docker run -p 8501:8501 video-qa-system
```

### Option C: AWS EC2

1. Launch EC2 instance (g3.4xlarge for GPU)
2. SSH into instance
3. Install Python and dependencies
4. Upload project files
5. Run with nginx reverse proxy
6. Configure security groups for port 8501

### Option D: Azure App Service

1. Create Python 3.10 App Service
2. Deploy via GitHub Actions
3. Configure startup command: `streamlit run streamlit_app.py --server.port 8000`
4. Set application settings (paths to models)

---

## ⚙️ Environment Variables (For Deployment)

Create a `.env` file:

```env
MODEL_DIR=C:\Project\MTech\Project\Data\model
CHECKPOINT=C:\Project\MTech\Project\Data\checkpoint-140
VIDEO_DIR=C:\Project\MTech\Project\Data\Anomaly-Videos-Part-1
SAVE_DIR=C:\Project\MTech\Project\Data\generated_summaries
DEVICE=cuda
```

Then update `streamlit_app.py` to use:

```python
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", r"C:\Project\MTech\Project\Data\model")
CHECKPOINT = os.getenv("CHECKPOINT", r"C:\Project\MTech\Project\Data\checkpoint-140")
```

---

## 🐛 Troubleshooting

### Issue: "Models not loading"
**Solution:** Ensure model files exist in the correct directory:
```
C:\Project\MTech\Project\Data\model\
├── binary_model.pkl
├── model_config.pkl
├── temporal_adapter.pt
├── le_weapon.pkl
├── le_location.pkl
├── le_people.pkl
└── le_super.pkl
```

### Issue: "Out of Memory"
**Solution:**
1. Increase paging file to 30GB (Windows)
2. Reduce FRAMES from 8 to 4 in `streamlit_app.py`
3. Use float16 instead of bfloat16

### Issue: "GPU out of memory"
**Solution:** Add to `streamlit_app.py`:
```python
torch.cuda.empty_cache()
```

### Issue: Slow video upload
**Solution:** Increase upload limit in `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 500  # MB
```

---

## 📊 Performance Optimization

### Model Caching
The app uses `@st.cache_resource` to load models once:
```python
@st.cache_resource
def load_all_models():
    # Models loaded once and reused
```

### Batch Processing
Process multiple videos:
```python
st.session_state.results = []
for video_file in uploaded_files:
    # Process each
```

### Reduce Inference Time
- Lower `max_new_tokens` in summarization (currently 120)
- Reduce number of frames (currently 8)
- Use int8 quantization instead of float16

---

## 📝 Features Included

✅ **Upload Video** - Upload MP4/AVI/MOV files  
✅ **Local Video Selection** - Browse local video directory  
✅ **Real-time Analysis** - Binary & multi-class classification  
✅ **AI Summarization** - Generate video summaries  
✅ **Interactive Q&A** - Ask questions about videos  
✅ **Export Results** - Save summaries as TXT  
✅ **Responsive UI** - Works on desktop/mobile  

---

## 🔒 Security Considerations

1. **Model Files**: Keep model directory private
2. **API Keys**: Use environment variables (not in code)
3. **Input Validation**: Limit video file sizes
4. **HTTPS**: Use HTTPS in production
5. **Authentication**: Add Streamlit authentication:

```python
import streamlit as st

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True

    password = st.text_input("Password:", type="password")
    if password == "your_password":
        st.session_state.password_correct = True
        return True
    return False

if not check_password():
    st.stop()

# Rest of app
```

---

## 📞 Support

For issues:
1. Check logs: `streamlit run streamlit_app.py --logger.level=debug`
2. Clear cache: `rm ~/.streamlit/`
3. Restart service: `streamlit run streamlit_app.py --server.runOnSave false`

---

## 📌 Next Steps

1. ✅ Test app locally
2. ✅ Push to GitHub
3. ✅ Deploy on Streamlit Cloud OR custom server
4. ✅ Configure domain name
5. ✅ Monitor performance

Good luck! 🚀
