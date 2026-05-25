# ==========================================================
# 🎬 STREAMLIT VIDEO QA SYSTEM
# ==========================================================
# Deployed version of the Video QA system with UI

import os
import re
import cv2
import torch
import joblib
import random
import numpy as np
import tempfile
import streamlit as st

from datetime import datetime
from PIL import Image

# Configure Streamlit
st.set_page_config(
    page_title="Video QA System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# IMPORTS - WITH FALLBACKS
# ==========================================================

from transformers import (
    AutoProcessor,
    AutoModel
)

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    from transformers import AutoModelForCausalLM as Qwen3VLForConditionalGeneration

from peft import PeftModel
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# SEED
# ==========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==========================================================
# CONFIG
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Updated to local paths
MODEL_DIR = r"C:\Project\MTech\Project\Data\model"
CHECKPOINT = r"C:\Project\MTech\Project\Data\checkpoint-140"
BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
FRAMES = 8

SAVE_DIR = r"C:\Project\MTech\Project\Data\generated_summaries"
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_MAP = {
    0: "ANOMALOUS",
    1: "NORMAL"
}

# ==========================================================
# STREAMLIT SESSION STATE
# ==========================================================

if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.binary_model = None
    st.session_state.config = None
    st.session_state.le_wpn = None
    st.session_state.le_loc = None
    st.session_state.le_ppl = None
    st.session_state.le_cat = None
    st.session_state.qwen = None
    st.session_state.summ_model = None
    st.session_state.qwen_processor = None
    st.session_state.summ_processor = None
    st.session_state.temporal_adapter = None

# ==========================================================
# LOAD MODELS (CACHED)
# ==========================================================

@st.cache_resource
def load_all_models():
    try:
        with st.spinner("🚦 Loading models (this may take a few minutes)..."):
            
            # Load model files
            binary_model = joblib.load(f"{MODEL_DIR}/binary_model.pkl")
            config = joblib.load(f"{MODEL_DIR}/model_config.pkl")
            le_wpn = joblib.load(f"{MODEL_DIR}/le_weapon.pkl")
            le_loc = joblib.load(f"{MODEL_DIR}/le_location.pkl")
            le_ppl = joblib.load(f"{MODEL_DIR}/le_people.pkl")
            le_cat = joblib.load(f"{MODEL_DIR}/le_super.pkl")
            
            config["n_cat"] = len(le_cat.classes_)
            
            # Load Qwen models
            qwen_processor = AutoProcessor.from_pretrained(BASE_MODEL)
            qwen = AutoModel.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.bfloat16
            ).to(DEVICE)
            qwen.eval()
            
            # Load summarization model with LoRA
            summ_processor = AutoProcessor.from_pretrained(BASE_MODEL)
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.bfloat16
            ).to(DEVICE)
            
            summ_model = PeftModel.from_pretrained(base_model, CHECKPOINT)
            summ_model.to(DEVICE)
            summ_model.eval()
            
            # Load temporal adapter
            temporal_adapter = TemporalAdapter(config)
            temporal_adapter.load_state_dict(
                torch.load(f"{MODEL_DIR}/temporal_adapter.pt", map_location=DEVICE)
            )
            temporal_adapter.to(DEVICE)
            temporal_adapter.eval()
            
            st.success("✅ All models loaded successfully!")
            
            return {
                "binary_model": binary_model,
                "config": config,
                "le_wpn": le_wpn,
                "le_loc": le_loc,
                "le_ppl": le_ppl,
                "le_cat": le_cat,
                "qwen": qwen,
                "summ_model": summ_model,
                "qwen_processor": qwen_processor,
                "summ_processor": summ_processor,
                "temporal_adapter": temporal_adapter
            }
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None

# ==========================================================
# TEMPORAL ADAPTER MODEL
# ==========================================================

class TemporalAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(
            cfg["in_dim"],
            cfg["n_ppl"] + cfg["n_wpn"] + cfg["n_loc"] + cfg["n_cat"]
        )
    
    def forward(self, x):
        return self.fc(x)

# ==========================================================
# FRAME SAMPLING
# ==========================================================

def sample_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total <= 0:
        return []
    
    idxs = np.linspace(0, total - 1, FRAMES, dtype=int)
    frames = []
    
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (448, 448))
        frames.append(Image.fromarray(frame))
    
    cap.release()
    return frames

# ==========================================================
# EMBED FRAMES
# ==========================================================

def embed_frames(frames, models):
    qwen = models["qwen"]
    qwen_processor = models["qwen_processor"]
    
    embs = []
    
    for img in frames:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe what you see."}
                ]
            }
        ]
        
        text = qwen_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = qwen_processor(
            text=[text],
            images=[img],
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            out = qwen(**inputs, output_hidden_states=True)
        
        emb = out.hidden_states[-1][:, -1, :]
        emb = F.normalize(emb, dim=-1)
        embs.append(emb.squeeze(0).float().cpu().numpy())
    
    return np.stack(embs)

# ==========================================================
# CLEAN OUTPUT
# ==========================================================

def clean_output(txt):
    txt = txt.strip()
    
    if "assistant" in txt.lower():
        txt = txt.split("assistant")[-1].strip()
    
    txt = " ".join(txt.split())
    
    sents = re.split(r'(?<=[.!?]) +', txt)
    cleaned = []
    
    banned = [
        "police", "court", "judge", "lawsuit",
        "investigation", "sentenced", "prison",
        "confessed", "reported"
    ]
    
    for s in sents:
        low = s.lower()
        if any(b in low for b in banned):
            continue
        if s not in cleaned:
            cleaned.append(s)
        if len(cleaned) >= 4:
            break
    
    txt = " ".join(cleaned)
    if not txt.endswith("."):
        txt += "."
    
    return txt

# ==========================================================
# SUMMARIZE VIDEO
# ==========================================================

def summarize_video(frames, models):
    summ_model = models["summ_model"]
    summ_processor = models["summ_processor"]
    
    if not frames:
        return "No frames to summarize"
    
    img = frames[0]
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe concisely."}
            ]
        }
    ]
    
    text = summ_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = summ_processor(
        text=[text],
        images=[img],
        return_tensors="pt"
    ).to(DEVICE)
    
    bad_words_ids = summ_processor.tokenizer(
        ["police", "court"],
        add_special_tokens=False
    ).input_ids
    
    with torch.no_grad():
        out = summ_model.generate(
            **inputs,
            max_new_tokens=120,
            min_new_tokens=40,
            do_sample=True,
            temperature=0.2,
            top_p=0.8,
            repetition_penalty=1.25,
            no_repeat_ngram_size=5,
            bad_words_ids=bad_words_ids,
            eos_token_id=summ_processor.tokenizer.eos_token_id,
            pad_token_id=summ_processor.tokenizer.eos_token_id,
        )
    
    generated_ids = out[0][inputs["input_ids"].shape[1]:]
    summary = summ_processor.decode(generated_ids, skip_special_tokens=True)
    summary = clean_output(summary)
    
    return summary

# ==========================================================
# BUILD CONTEXT
# ==========================================================

def build_context(video_path, models):
    frames = sample_frames(video_path)
    
    if not frames:
        return None, None
    
    # Get embeddings
    emb_seq = embed_frames(frames, models)
    pooled = np.mean(emb_seq, axis=0)
    
    # Binary classification
    binary_pred = models["binary_model"].predict([pooled])[0]
    binary_proba = models["binary_model"].predict_proba([pooled])[0]
    binary_conf = binary_proba[binary_pred]
    
    # Multi-class predictions
    multi_out = models["temporal_adapter"](torch.from_numpy(emb_seq).float().to(DEVICE))
    multi_out = multi_out.cpu().detach().numpy()
    
    n_ppl = len(models["le_ppl"].classes_)
    n_wpn = len(models["le_wpn"].classes_)
    n_loc = len(models["le_loc"].classes_)
    
    ppl_preds = np.argmax(multi_out[:, :n_ppl], axis=1)
    wpn_preds = np.argmax(multi_out[:, n_ppl:n_ppl+n_wpn], axis=1)
    loc_preds = np.argmax(multi_out[:, n_ppl+n_wpn:n_ppl+n_wpn+n_loc], axis=1)
    cat_preds = np.argmax(multi_out[:, n_ppl+n_wpn+n_loc:], axis=1)
    
    # Get summaries
    summary = summarize_video(frames, models)
    
    context = {
        "binary_class": CLASS_MAP[binary_pred],
        "binary_confidence": binary_conf,
        "people": models["le_ppl"].inverse_transform(ppl_preds)[0],
        "weapon": models["le_wpn"].inverse_transform(wpn_preds)[0],
        "location": models["le_loc"].inverse_transform(loc_preds)[0],
        "category": models["le_cat"].inverse_transform(cat_preds)[0],
        "summary": summary,
        "frames": frames
    }
    
    return context, frames

# ==========================================================
# STREAMLIT UI
# ==========================================================

st.title("🎬 Video QA & Anomaly Detection System")

st.markdown("""
Analyze surveillance videos for anomalies using deep learning:
- Binary classification (Normal/Anomalous)
- Multi-class detection (people, weapons, locations)
- Video summarization with AI
- Interactive Q&A
""")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
st.sidebar.info(
    f"""
**Models Path:** {MODEL_DIR}
**Checkpoint:** {CHECKPOINT}
**Device:** {DEVICE}
**Frames:** {FRAMES}
    """
)

# Main content
tab1, tab2, tab3 = st.tabs(["Upload Video", "Local Video", "Settings"])

# Load models
models = load_all_models()

if models is None:
    st.error("Failed to load models. Check your paths and dependencies.")
    st.stop()

# TAB 1: Upload Video
with tab1:
    st.header("📤 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"]
    )
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        
        st.video(uploaded_file)
        
        if st.button("🔍 Analyze Video", key="analyze_upload"):
            with st.spinner("Analyzing video..."):
                context, frames = build_context(tmp_path, models)
                
                if context:
                    st.success("✅ Analysis Complete!")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Classification Results")
                        st.metric("Status", context["binary_class"], f"({context['binary_confidence']:.2%})")
                        st.metric("People", context["people"])
                        st.metric("Weapon", context["weapon"])
                        st.metric("Location", context["location"])
                        st.metric("Category", context["category"])
                    
                    with col2:
                        st.subheader("Summary")
                        st.write(context["summary"])
                        
                        if st.button("💾 Save Summary"):
                            base = uploaded_file.name.split('.')[0]
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            txt_path = os.path.join(SAVE_DIR, f"{base}_{timestamp}.txt")
                            
                            with open(txt_path, "w", encoding="utf-8") as f:
                                f.write(context["summary"])
                            
                            st.success(f"✅ Saved to {txt_path}")
                    
                    # Q&A Section
                    st.subheader("❓ Ask Questions")
                    question = st.text_input("Ask about the video:")
                    
                    if question:
                        answer = answer_question(context, question)
                        st.info(f"**Q:** {question}\n\n**A:** {answer}")
        
        # Cleanup
        os.unlink(tmp_path)

# TAB 2: Local Video
with tab2:
    st.header("📁 Select Local Video")
    
    # Show available videos
    video_dir = r"C:\Project\MTech\Project\Data\Anomaly-Videos-Part-1"
    
    if os.path.exists(video_dir):
        video_files = []
        for root, dirs, files in os.walk(video_dir):
            for file in files:
                if file.endswith((".mp4", ".avi", ".mov", ".mkv")):
                    video_files.append(os.path.join(root, file))
        
        if video_files:
            selected_video = st.selectbox("Choose a video:", video_files)
            
            if st.button("🔍 Analyze Video", key="analyze_local"):
                with st.spinner("Analyzing video..."):
                    context, frames = build_context(selected_video, models)
                    
                    if context:
                        st.success("✅ Analysis Complete!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Classification Results")
                            st.metric("Status", context["binary_class"], f"({context['binary_confidence']:.2%})")
                            st.metric("People", context["people"])
                            st.metric("Weapon", context["weapon"])
                            st.metric("Location", context["location"])
                            st.metric("Category", context["category"])
                        
                        with col2:
                            st.subheader("Summary")
                            st.write(context["summary"])
                            
                            if st.button("💾 Save Summary", key="save_local"):
                                base = os.path.splitext(os.path.basename(selected_video))[0]
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                txt_path = os.path.join(SAVE_DIR, f"{base}_{timestamp}.txt")
                                
                                with open(txt_path, "w", encoding="utf-8") as f:
                                    f.write(context["summary"])
                                
                                st.success(f"✅ Saved to {txt_path}")
                        
                        # Q&A Section
                        st.subheader("❓ Ask Questions")
                        question = st.text_input("Ask about the video:", key="qa_local")
                        
                        if question:
                            answer = answer_question(context, question)
                            st.info(f"**Q:** {question}\n\n**A:** {answer}")
        else:
            st.warning("No video files found in the directory.")
    else:
        st.error(f"Video directory not found: {video_dir}")

# TAB 3: Settings
with tab3:
    st.header("⚙️ Settings")
    st.write("**Current Configuration:**")
    st.json({
        "MODEL_DIR": MODEL_DIR,
        "CHECKPOINT": CHECKPOINT,
        "DEVICE": DEVICE,
        "BASE_MODEL": BASE_MODEL,
        "FRAMES": FRAMES,
        "SAVE_DIR": SAVE_DIR
    })

# ==========================================================
# Q&A FUNCTION
# ==========================================================

def answer_question(context, question):
    """Route questions to relevant context"""
    q_lower = question.lower()
    
    # Status questions
    if any(w in q_lower for w in ["normal", "anomalous", "status", "safe"]):
        return f"The video shows **{context['binary_class']}** activity (confidence: {context['binary_confidence']:.2%})"
    
    # People questions
    if any(w in q_lower for w in ["people", "person", "how many", "number", "count"]):
        return f"People detected: **{context['people']}**"
    
    # Weapon questions
    if any(w in q_lower for w in ["weapon", "gun", "knife", "armed", "used"]):
        return f"Weapon type: **{context['weapon']}**"
    
    # Location questions
    if any(w in q_lower for w in ["location", "where", "place", "located"]):
        return f"Location type: **{context['location']}**"
    
    # Category questions
    if any(w in q_lower for w in ["category", "type", "event", "what is happening", "what"]):
        return f"Event category: **{context['category']}**. Summary: {context['summary']}"
    
    # Default summary
    return context["summary"]

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Video QA System v1.0**
    
    Built with Streamlit & Qwen3-VL
    
    📧 [Contact](mailto:support@example.com)
    """
)
