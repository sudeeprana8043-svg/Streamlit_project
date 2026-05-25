# 🎬 Streamlit Video QA & Anomaly Detection System

A state-of-the-art web application for analyzing surveillance videos using deep learning. Detect anomalies, classify events, and get instant AI-powered summaries with interactive Q&A.

![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

- **🎥 Video Upload & Analysis** - Upload MP4, AVI, MOV, MKV files for real-time analysis
- **📊 Binary Classification** - Normal vs Anomalous activity detection with confidence scores
- **🔍 Multi-class Detection** - Identify people, weapons, locations, and event categories
- **🤖 AI Summarization** - Generate video summaries using fine-tuned Qwen3-VL model
- **💬 Interactive Q&A** - Ask natural language questions about video content
- **💾 Export Results** - Save analysis summaries as text files
- **⚡ GPU Accelerated** - CUDA support for fast inference
- **🎨 Responsive UI** - Beautiful web interface that works on desktop & mobile

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/sudeeprana8043-svg/streamlit-video-qa.git
cd streamlit-video-qa

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run streamlit_app.py
```

Opens at: **http://localhost:8501**

### Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to https://share.streamlit.io
3. Click "New app" → Select this repository
4. Your app gets a public URL instantly!

## 📋 Requirements

- Python 3.9+
- 16GB RAM minimum (8GB with optimizations)
- CUDA 11.8+ (optional, for GPU acceleration)
- Model files: Download from `https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct`

## 📦 Dependencies

See `requirements.txt` for full list:
- **streamlit** - Web framework
- **transformers** - Model loading
- **peft** - LoRA fine-tuning
- **torch** - Deep learning
- **opencv-python** - Video processing
- **scikit-learn** - ML utilities

## 🏗️ Architecture

```
Video Input
    ↓
Frame Extraction (8 frames)
    ↓
Embedding Generation (Qwen3-VL-Embedding)
    ↓
├─ Binary Classification (Normal/Anomalous)
├─ Multi-class Prediction (People/Weapon/Location/Category)
└─ Video Summarization (LoRA-finetuned)
    ↓
Interactive Q&A & Results
```

## 📁 Project Structure

```
streamlit-video-qa/
├── streamlit_app.py              # Main application
├── requirements.txt              # Dependencies
├── .streamlit/
│   └── config.toml              # Streamlit configuration
├── .gitignore
├── README.md                     # This file
├── QUICK_START_STREAMLIT.md     # Quick start guide
└── STREAMLIT_DEPLOYMENT_GUIDE.md # Deployment instructions
```

## 🔧 Configuration

Edit `streamlit_app.py` to customize:

```python
MODEL_DIR = r"C:\path\to\models"           # Model directory
CHECKPOINT = r"C:\path\to\checkpoint-140"  # LoRA checkpoint
BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"   # Base model
FRAMES = 8                                   # Number of frames to sample
```

## 💻 Usage Examples

### Upload Video
1. Go to "Upload Video" tab
2. Select an MP4/AVI/MOV/MKV file
3. Click "Analyze Video"
4. Get results in ~30-60 seconds

### Ask Questions
After analysis, try:
- ❓ "What is the status?" → Normal/Anomalous
- ❓ "How many people?" → People count
- ❓ "Is there a weapon?" → Weapon type
- ❓ "Where is this?" → Location
- ❓ "What's happening?" → Video summary

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Models not loading | Ensure model files exist in `MODEL_DIR` |
| Out of memory | Increase Windows paging file to 30GB; restart |
| Slow inference | Enable GPU; reduce `FRAMES` to 4 |
| Upload fails | Check file format (MP4/AVI/MOV/MKV); max 200MB |

## 📚 Documentation

- **[QUICK_START_STREAMLIT.md](QUICK_START_STREAMLIT.md)** - Get started in 5 minutes
- **[STREAMLIT_DEPLOYMENT_GUIDE.md](STREAMLIT_DEPLOYMENT_GUIDE.md)** - Production deployment guide

## 🌐 Deployment Options

### Streamlit Cloud (Recommended)
- Free hosting
- Instant deployment
- Automatic updates from GitHub

### Docker
```bash
docker build -t video-qa-system .
docker run -p 8501:8501 video-qa-system
```

### AWS/Azure/Custom Server
See `STREAMLIT_DEPLOYMENT_GUIDE.md` for detailed instructions

## 📊 Performance

| Metric | Value |
|--------|-------|
| Model Size | 4.26 GB (Qwen3-VL-Embedding) + 2B (Qwen3-VL-Instruct) |
| Inference Time | 30-60 seconds per video |
| Frame Processing | 8 evenly-spaced frames |
| Supported Formats | MP4, AVI, MOV, MKV |
| Max Upload | 200MB (configurable) |

## 🔐 Security

- ✅ Models loaded locally (no cloud dependencies)
- ✅ Videos processed in-memory (not stored)
- ✅ Input validation on all uploads
- ⚠️ Keep model files private in production
- ⚠️ Use HTTPS in production deployments

## 📝 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

**Author:** sudeeprana8043-svg  
**Email:** sudeep.rana8043@gmail.com  
**GitHub:** https://github.com/sudeeprana8043-svg

## 🙏 Acknowledgments

- **Qwen Team** - For Qwen3-VL models
- **Hugging Face** - For transformers library
- **Streamlit** - For amazing web framework
- **PyTorch** - For deep learning foundation

---

## 📌 Citation

If you use this system in your research, please cite:

```bibtex
@software{streamlit_video_qa_2024,
  author = {Sudeep Rana},
  title = {Streamlit Video QA & Anomaly Detection System},
  year = {2024},
  url = {https://github.com/sudeeprana8043-svg/streamlit-video-qa}
}
```

---

**Made with ❤️ using Streamlit & Qwen3-VL**
