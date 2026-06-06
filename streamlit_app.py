# ==========================================================
# 🎬 STREAMLIT VIDEO QA SYSTEM
# ==========================================================
# Deployed version of the Video QA system with UI

import os
import tempfile
import streamlit as st

from datetime import datetime

import app_core as core
from app_core import (
    DEVICE,
    MODEL_DIR,
    CHECKPOINT,
    MODEL_REPO_ID,
    TEMPORAL_ADAPTER_GDRIVE_ID,
    TEMPORAL_ADAPTER_GDRIVE_URL,
    BASE_MODEL,
    EMBED_MODEL,
    FRAMES,
    SAVE_DIR,
    VIDEO_DIR,
    MODEL_FILES,
    build_context,
    answer_question,
)

# Configure Streamlit
st.set_page_config(
    page_title="Video QA System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD MODELS (CACHED) - delegates to app_core
# ==========================================================

@st.cache_resource
def load_all_models():
    try:
        with st.spinner("🚦 Loading models (this may take a few minutes)..."):
            models = core.load_all_models(progress=lambda m: None)
            st.success("✅ All models loaded successfully!")
            return models
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info(
            "Model artifacts are not included in the deployed app. "
            "Upload the required files under a `model/` folder in the repository, "
            "or create a Hugging Face model repository containing these files and set "
            "`MODEL_REPO_ID` in Streamlit secrets."
        )
        return None

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
    video_dir = VIDEO_DIR
    
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
        "MODEL_REPO_ID": MODEL_REPO_ID,
        "TEMPORAL_ADAPTER_GDRIVE_ID_SET": bool(TEMPORAL_ADAPTER_GDRIVE_ID),
        "TEMPORAL_ADAPTER_GDRIVE_URL_SET": bool(TEMPORAL_ADAPTER_GDRIVE_URL),
        "DEVICE": DEVICE,
        "BASE_MODEL": BASE_MODEL,
        "EMBED_MODEL": EMBED_MODEL,
        "FRAMES": FRAMES,
        "SAVE_DIR": SAVE_DIR,
        "VIDEO_DIR": VIDEO_DIR
    })
    st.write("**Required model files:**")
    st.code("\n".join(MODEL_FILES), language="text")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Video QA System v1.0**
    
    Built with Streamlit & Qwen3-VL
    
    📧 [Contact](mailto:support@example.com)
    """
)
