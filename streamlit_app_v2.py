# ==========================================================
# 🎬 STREAMLIT VIDEO QA SYSTEM (v2)
# ==========================================================
# Same UI as streamlit_app.py, but the binary classification is
# replaced with the SimpleAdapter + LoRA ensemble (binary_ensemble.py).
# Multi-class detection and summarization still come from app_core.

import os
import tempfile
import streamlit as st
import numpy as np
import torch

from datetime import datetime

import app_core as core
import binary_ensemble
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
# LOAD MODELS (CACHED)
# ==========================================================

@st.cache_resource
def load_all_models():
    try:
        with st.spinner("🚦 Loading models (this may take a few minutes)..."):
            models = core.load_all_models(progress=lambda m: None)
            if models is None:
                return None
            # Load the ensemble binary classifier heads
            models["binary_ensemble"] = binary_ensemble.load_binary_models(DEVICE)
            st.success("✅ All models loaded successfully!")
            return models
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info(
            "Model artifacts are not included in the deployed app. "
            "Ensure `input_dim.pkl`, `simple_adapter.pt`, `lora_model.pt` are in "
            "`BINARY_MODEL_DIR`, plus the multi-class/summary files in `MODEL_DIR` "
            "(or set `MODEL_REPO_ID`)."
        )
        return None


# ==========================================================
# BUILD CONTEXT (ensemble binary + existing multi-class/summary)
# ==========================================================

def build_context_v2(video_path, models):
    frames = core.sample_frames(video_path)

    if not frames:
        return None, None

    # ---- Binary classification: SimpleAdapter + LoRA ensemble ----
    binary_class, binary_conf, binary_parts = binary_ensemble.predict_binary(
        frames,
        models["binary_ensemble"],
        models["summ_model"],
        models["summ_processor"],
        DEVICE,
    )

    # ---- Summarization (also used to enrich the multi-class embedding) ----
    summary = core.summarize_video(frames, models)

    # ---- Multi-class attributes ----
    if models.get("mc_bundle") is not None:
        # Preferred: trained clf_* heads on the fused (visual + summary) embedding
        people, weapon, location, category, actions = core.predict_multiclass(
            frames, summary, models
        )
    else:
        # Legacy fallback: temporal adapter. Slice using the SAME class counts
        # the adapter was built with (config), not len(le_*), to avoid an
        # out-of-range (empty) slice -> argmax([]) crash.
        emb_seq = core.embed_frames(frames, models)
        multi_out = models["temporal_adapter"](
            torch.from_numpy(emb_seq).float().to(DEVICE)
        ).cpu().detach().numpy()

        cfg = models["config"]
        total = multi_out.shape[1]
        n_ppl = cfg.get("n_ppl", len(models["le_ppl"].classes_))
        n_wpn = cfg.get("n_wpn", len(models["le_wpn"].classes_))
        n_loc = cfg.get("n_loc", len(models["le_loc"].classes_))
        n_cat = cfg.get("n_cat", max(total - n_ppl - n_wpn - n_loc, 0))

        def _seg_argmax(start, length):
            seg = multi_out[:, start:start + length]
            if seg.shape[1] == 0:
                return np.zeros(multi_out.shape[0], dtype=int)
            return np.argmax(seg, axis=1)

        def _safe_inv(le, preds):
            idx = np.clip(preds, 0, len(le.classes_) - 1)
            return le.inverse_transform(idx)

        people = _safe_inv(models["le_ppl"], _seg_argmax(0, n_ppl))[0]
        weapon = _safe_inv(models["le_wpn"], _seg_argmax(n_ppl, n_wpn))[0]
        location = _safe_inv(models["le_loc"], _seg_argmax(n_ppl + n_wpn, n_loc))[0]
        category = _safe_inv(models["le_cat"], _seg_argmax(n_ppl + n_wpn + n_loc, n_cat))[0]
        actions = None

    context = {
        "binary_class": binary_class,
        "binary_confidence": float(binary_conf),
        "binary_parts": binary_parts,
        "people": people,
        "weapon": weapon,
        "location": location,
        "category": category,
        "actions": actions,
        "summary": summary,
        "frames": frames,
    }

    return context, frames


# ==========================================================
# STREAMLIT UI
# ==========================================================

st.title("🎬 Video QA & Anomaly Detection System")

st.markdown("""
Analyze surveillance videos for anomalies using deep learning:
- Binary classification (Normal/Anomalous) — **SimpleAdapter + LoRA ensemble**
- Multi-class detection (people, weapons, locations)
- Video summarization with AI
- Interactive Q&A
""")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
st.sidebar.info(
    f"""
**Models Path:** {MODEL_DIR}
**Binary Models:** {binary_ensemble.BINARY_MODEL_DIR}
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
                context, frames = build_context_v2(tmp_path, models)

                if context:
                    st.success("✅ Analysis Complete!")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Classification Results")
                        st.metric("Status", context["binary_class"], f"({context['binary_confidence']:.2%})")
                        st.caption(
                            f"SimpleAdapter: {context['binary_parts']['simple']:.4f} | "
                            f"LoRA: {context['binary_parts']['lora']:.4f}"
                        )
                        st.metric("People", context["people"])
                        st.metric("Weapon", context["weapon"])
                        st.metric("Location", context["location"])
                        st.metric("Category", context["category"])
                        if context.get("actions"):
                            st.metric("Detected Actions", ", ".join(context["actions"]))

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
                    context, frames = build_context_v2(selected_video, models)

                    if context:
                        st.success("✅ Analysis Complete!")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Classification Results")
                            st.metric("Status", context["binary_class"], f"({context['binary_confidence']:.2%})")
                            st.caption(
                                f"SimpleAdapter: {context['binary_parts']['simple']:.4f} | "
                                f"LoRA: {context['binary_parts']['lora']:.4f}"
                            )
                            st.metric("People", context["people"])
                            st.metric("Weapon", context["weapon"])
                            st.metric("Location", context["location"])
                            st.metric("Category", context["category"])
                            if context.get("actions"):
                                st.metric("Detected Actions", ", ".join(context["actions"]))

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
        "BINARY_MODEL_DIR": binary_ensemble.BINARY_MODEL_DIR,
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
    st.write("**Required model files (multi-class / summary):**")
    st.code("\n".join(MODEL_FILES), language="text")
    st.write("**Required binary ensemble files (in BINARY_MODEL_DIR):**")
    st.code("input_dim.pkl\nsimple_adapter.pt\nlora_model.pt", language="text")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Video QA System v2.0**
    
    Built with Streamlit & Qwen3-VL
    
    Binary: SimpleAdapter + LoRA ensemble
    """
)
