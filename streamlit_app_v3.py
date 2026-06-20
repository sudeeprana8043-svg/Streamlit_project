# ==========================================================
# 🎬 STREAMLIT VIDEO QA SYSTEM (v3)
# ==========================================================
# Flow:
#   1. "Analyze Video" -> shows ONLY Normal/Anomalous + Summary (fast).
#   2. A VQA section then lets the user ask for details (people, weapon,
#      location, category, actions). The multi-class attributes are computed
#      lazily (only when first asked) and cached in session state.
#
# Binary classification = SimpleAdapter + LoRA ensemble (binary_ensemble.py).
# Multi-class detection  = trained clf_* bundle (app_core.predict_multiclass).
# Summarization          = Qwen3-VL + LoRA checkpoint (app_core).

import os
import tempfile
import streamlit as st

from datetime import datetime

import app_core as core
import binary_ensemble
from app_core import (
    DEVICE,
    MODEL_DIR,
    CHECKPOINT,
    FRAMES,
    SAVE_DIR,
    VIDEO_DIR,
    answer_question,
)

# Configure Streamlit
st.set_page_config(
    page_title="Video QA System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
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
            models["binary_ensemble"] = binary_ensemble.load_binary_models(DEVICE)
            st.success("✅ All models loaded successfully!")
            return models
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info(
            "Ensure `input_dim.pkl`, `simple_adapter.pt`, `lora_model.pt` are in "
            "`BINARY_MODEL_DIR`, plus the multi-class/summary files in `MODEL_DIR`."
        )
        return None


# ==========================================================
# STAGE 1: BASIC ANALYSIS (binary + summary only)
# ==========================================================

def analyze_basic(video_path, models):
    """Fast pass: sample frames, run the binary ensemble and the summarizer.
    Multi-class attributes are intentionally NOT computed here."""
    frames = core.sample_frames(video_path)
    if not frames:
        return None

    binary_class, binary_conf, binary_parts = binary_ensemble.predict_binary(
        frames,
        models["binary_ensemble"],
        models["summ_model"],
        models["summ_processor"],
        DEVICE,
    )
    summary = core.summarize_video(frames, models)

    return {
        "binary_class": binary_class,
        "binary_confidence": float(binary_conf),
        "binary_parts": binary_parts,
        "summary": summary,
        "frames": frames,
        "attrs_done": False,
    }


# ==========================================================
# STAGE 2: LAZY MULTI-CLASS ATTRIBUTES (computed on first VQA)
# ==========================================================

def ensure_attrs(state_key, models):
    """Compute people/weapon/location/category (+ actions) once, on demand."""
    ctx = st.session_state[state_key]
    if ctx.get("attrs_done"):
        return ctx

    with st.spinner("Detecting attributes..."):
        people, weapon, location, category, actions = core.predict_multiclass(
            ctx["frames"], ctx["summary"], models
        )

    ctx.update({
        "people": people,
        "weapon": weapon,
        "location": location,
        "category": category,
        "actions": actions,
        "attrs_done": True,
    })
    st.session_state[state_key] = ctx
    return ctx


# ==========================================================
# RESULTS + VQA RENDERER
# ==========================================================

def render_results(state_key, models, source_label):
    ctx = st.session_state[state_key]

    st.success("✅ Analysis Complete!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Classification Results")
        st.metric(
            "Status",
            ctx["binary_class"],
            f"({ctx['binary_confidence']:.2%})",
        )
        st.caption(
            f"SimpleAdapter: {ctx['binary_parts']['simple']:.4f} | "
            f"LoRA: {ctx['binary_parts']['lora']:.4f}"
        )

    with col2:
        st.subheader("Summary")
        st.write(ctx["summary"])
        if st.button("💾 Save Summary", key=f"{state_key}_save"):
            base = os.path.splitext(os.path.basename(source_label))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            txt_path = os.path.join(SAVE_DIR, f"{base}_{timestamp}.txt")
            os.makedirs(SAVE_DIR, exist_ok=True)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(ctx["summary"])
            st.success(f"✅ Saved to {txt_path}")

    # ---- VQA Section ----
    st.divider()
    st.subheader("❓ Ask Questions (VQA)")
    st.caption(
        "Detailed attributes are detected on demand. Use a quick button or type a question."
    )

    b1, b2, b3, b4, b5 = st.columns(5)
    quick = None
    if b1.button("👥 People", key=f"{state_key}_b_ppl"):
        quick = ("Number of people", "people")
    if b2.button("🔪 Weapon", key=f"{state_key}_b_wpn"):
        quick = ("Weapon", "weapon")
    if b3.button("📍 Location", key=f"{state_key}_b_loc"):
        quick = ("Location", "location")
    if b4.button("🏷️ Category", key=f"{state_key}_b_cat"):
        quick = ("Event category", "category")
    if b5.button("🎬 Actions", key=f"{state_key}_b_act"):
        quick = ("Detected actions", "actions")

    question = st.text_input("Or type a question:", key=f"{state_key}_q")

    if quick is not None:
        ctx = ensure_attrs(state_key, models)
        label, field = quick
        if field == "actions":
            value = ", ".join(ctx.get("actions") or []) or "None detected"
        else:
            value = ctx.get(field, "N/A")
        st.info(f"**{label}:** {value}")

    if question:
        ctx = ensure_attrs(state_key, models)
        answer = answer_question(ctx, question)
        st.info(f"**Q:** {question}\n\n**A:** {answer}")


# ==========================================================
# STREAMLIT UI
# ==========================================================

st.title("🎬 Video QA & Anomaly Detection System")

st.markdown(
    """
Analyze surveillance videos:
1. **Analyze** -> Normal/Anomalous status + AI summary.
2. **Ask** -> people, weapon, location, category, actions (computed on demand).
"""
)

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

tab1, tab2, tab3 = st.tabs(["Upload Video", "Local Video", "Settings"])

models = load_all_models()
if models is None:
    st.error("Failed to load models. Check your paths and dependencies.")
    st.stop()

# TAB 1: Upload Video
with tab1:
    st.header("📤 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded_file is not None:
        st.video(uploaded_file)

        if st.button("🔍 Analyze Video", key="analyze_upload"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            try:
                with st.spinner("Analyzing video (status + summary)..."):
                    ctx = analyze_basic(tmp_path, models)
            finally:
                os.unlink(tmp_path)

            if ctx is None:
                st.error("Could not read frames from the video.")
            else:
                ctx["source_label"] = uploaded_file.name
                st.session_state["ctx_upload"] = ctx

        if "ctx_upload" in st.session_state:
            render_results(
                "ctx_upload", models,
                st.session_state["ctx_upload"].get("source_label", "video"),
            )

# TAB 2: Local Video
with tab2:
    st.header("📁 Select Local Video")

    if os.path.exists(VIDEO_DIR):
        video_files = []
        for root, _dirs, files in os.walk(VIDEO_DIR):
            for file in files:
                if file.endswith((".mp4", ".avi", ".mov", ".mkv")):
                    video_files.append(os.path.join(root, file))

        if video_files:
            selected_video = st.selectbox("Choose a video:", video_files)

            if st.button("🔍 Analyze Video", key="analyze_local"):
                with st.spinner("Analyzing video (status + summary)..."):
                    ctx = analyze_basic(selected_video, models)
                if ctx is None:
                    st.error("Could not read frames from the video.")
                else:
                    ctx["source_label"] = selected_video
                    st.session_state["ctx_local"] = ctx

            if "ctx_local" in st.session_state:
                render_results(
                    "ctx_local", models,
                    st.session_state["ctx_local"].get("source_label", "video"),
                )
        else:
            st.warning("No video files found in the directory.")
    else:
        st.error(f"Video directory not found: {VIDEO_DIR}")

# TAB 3: Settings
with tab3:
    st.header("⚙️ Settings")
    st.write("**Current Configuration:**")
    st.json({
        "MODEL_DIR": MODEL_DIR,
        "BINARY_MODEL_DIR": binary_ensemble.BINARY_MODEL_DIR,
        "CHECKPOINT": CHECKPOINT,
        "DEVICE": DEVICE,
        "FRAMES": FRAMES,
    })
