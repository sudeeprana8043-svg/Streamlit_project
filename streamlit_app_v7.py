# ==========================================================
# STREAMLIT VIDEO QA SYSTEM - HYBRID V7 CACHED-QWEN SINGLE-BACKBONE ARCHITECTURE
# ==========================================================

import os
import tempfile
from datetime import datetime

import streamlit as st

import app_core_v7 as core
import binary_ensemble

from app_core_v7 import (
    DEVICE,
    MULTITASK_MODEL_DIR,
    SUMMARY_CHECKPOINT,
    FRAMES,
    SAVE_DIR,
    VIDEO_DIR,
    answer_question,
)


st.set_page_config(
    page_title="Video QA System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_all_models():
    try:
        status = st.empty()

        def progress(message):
            # Show the exact loading stage in Streamlit and in streamlit.log.
            print(f"[MODEL LOAD] {message}", flush=True)
            status.info(message)

        with st.spinner(
            "Loading models. This may take a few minutes..."
        ):
            models = core.load_all_models(
                progress=progress
            )

            progress("9/9 Loading binary anomaly detection ensemble...")
            models["binary_ensemble"] = (
                binary_ensemble.load_binary_models(
                    DEVICE
                )
            )
            progress("READY: All V7 cached-Qwen models loaded successfully.")

        status.empty()
        print("[MODEL LOAD] All models loaded successfully.", flush=True)
        st.success(
            "✅ All models loaded successfully!"
        )

        return models

    except Exception as error:
        st.error(
            f"❌ Error loading models: {error}"
        )

        st.info(
            "Check BINARY_MODEL_DIR for the binary ensemble, "
            "MULTITASK_MODEL_DIR for the new temporal/fusion PKL files, "
            "and SUMMARY_CHECKPOINT for the LoRA summarization checkpoint."
        )

        return None


# ==========================================================
# STAGE 1 - BINARY + UI SUMMARY
# ==========================================================

def analyze_basic(video_path, models):
    frames = core.sample_frames(video_path)

    if not frames:
        return None

    binary_class, binary_conf, binary_parts = (
        binary_ensemble.predict_binary(
            frames,
            models["binary_ensemble"],
            models["summ_model"],
            models["summ_processor"],
            DEVICE,
        )
    )

    summary = core.summarize_video(
        frames,
        models,
    )

    return {
        "binary_class": binary_class,
        "binary_confidence": float(binary_conf),
        "binary_parts": binary_parts,
        "summary": summary,
        "frames": frames,
        "attrs_done": False,
    }


# ==========================================================
# STAGE 2 - HYBRID MULTICLASS MODEL
# ==========================================================

def ensure_attrs(state_key, models):
    ctx = st.session_state[state_key]

    if ctx.get("attrs_done"):
        return ctx

    with st.spinner(
        "Running hybrid multiclass classification..."
    ):
        (
            people,
            weapon,
            location,
            activity,
            actions,
        ) = core.predict_multiclass(
            ctx["frames"],
            ctx["summary"],
            models,
        )

    ctx.update({
        "people": people,
        "weapon": weapon,
        "location": location,
        "category": activity,
        "activity": activity,
        "actions": actions,
        "attrs_done": True,
    })

    st.session_state[state_key] = ctx

    return ctx


# ==========================================================
# RESULTS
# ==========================================================

def render_results(
    state_key,
    models,
    source_label,
):
    ctx = st.session_state[state_key]

    st.success("✅ Analysis Complete!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Binary Classification")

        st.metric(
            "Status",
            ctx["binary_class"],
            f"({ctx['binary_confidence']:.2%})",
        )


    with col2:
        st.subheader("Video Summary")
        st.write(ctx["summary"])

        if st.button(
            "💾 Save Summary",
            key=f"{state_key}_save",
        ):
            base = os.path.splitext(
                os.path.basename(source_label)
            )[0]

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            txt_path = os.path.join(
                SAVE_DIR,
                f"{base}_{timestamp}.txt",
            )

            os.makedirs(
                SAVE_DIR,
                exist_ok=True,
            )

            with open(
                txt_path,
                "w",
                encoding="utf-8",
            ) as f:
                f.write(ctx["summary"])

            st.success(
                f"✅ Saved to {txt_path}"
            )

    st.divider()

    st.subheader(
        "❓ Structured Video Question Answering"
    )

    st.caption(
        "Activity and weapon use the previous trained classifiers; "
        "people and location use the Temporal Adapter + "
        "generated-summary fusion model."
    )

    b1, b2, b3, b4 = st.columns(4)

    quick = None

    if b1.button(
        "👥 People",
        key=f"{state_key}_b_ppl",
    ):
        quick = (
            "Number of people",
            "people",
        )

    if b2.button(
        "🔪 Weapon",
        key=f"{state_key}_b_wpn",
    ):
        quick = (
            "Weapon",
            "weapon",
        )

    if b3.button(
        "📍 Location",
        key=f"{state_key}_b_loc",
    ):
        quick = (
            "Location",
            "location",
        )

    if b4.button(
        "🎬 Activity",
        key=f"{state_key}_b_act",
    ):
        quick = (
            "Activity",
            "activity",
        )

    question = st.text_input(
        "Or type a question:",
        key=f"{state_key}_q",
    )

    if quick is not None:
        ctx = ensure_attrs(
            state_key,
            models,
        )

        label, field = quick

        st.info(
            f"**{label}:** "
            f"{ctx.get(field, 'N/A')}"
        )

    if question:
        ctx = ensure_attrs(
            state_key,
            models,
        )

        answer = answer_question(
            ctx,
            question,
        )

        st.info(
            f"**Q:** {question}\n\n"
            f"**A:** {answer}"
        )


# ==========================================================
# UI
# ==========================================================

st.title(
    "🎬 Video QA & Anomaly Detection System"
)

st.markdown(
    """
1. **Analyze** → Normal/Anomalous status and video summary.
2. **Ask** → Activity and weapon use the previous trained classifiers;
   people and location use the Temporal Adapter + Summary Fusion model.
"""
)

st.sidebar.header("⚙️ Configuration")

st.sidebar.info(
    f"""
**Multitask Model:** {MULTITASK_MODEL_DIR}

**Binary Models:** {binary_ensemble.BINARY_MODEL_DIR}

**Summary Checkpoint:** {SUMMARY_CHECKPOINT}

**Device:** {DEVICE}

**Frames:** {FRAMES}
"""
)

tab1, tab2, tab3 = st.tabs([
    "Upload Video",
    "Local Video",
    "Settings",
])

models = load_all_models()

if models is None:
    st.error(
        "Failed to load models. "
        "Check model paths and dependencies."
    )
    st.stop()


# ==========================================================
# UPLOAD VIDEO
# ==========================================================

with tab1:
    st.header("📤 Upload Video")

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv",
        ],
    )

    if uploaded_file is not None:
        st.video(uploaded_file)

        if st.button(
            "🔍 Analyze Video",
            key="analyze_upload",
        ):
            suffix = os.path.splitext(
                uploaded_file.name
            )[1] or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as tmp:
                tmp.write(
                    uploaded_file.getbuffer()
                )
                tmp_path = tmp.name

            try:
                with st.spinner(
                    "Analyzing binary status and summary..."
                ):
                    ctx = analyze_basic(
                        tmp_path,
                        models,
                    )

            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            if ctx is None:
                st.error(
                    "Could not read frames from the video."
                )

            else:
                ctx["source_label"] = (
                    uploaded_file.name
                )

                st.session_state[
                    "ctx_upload"
                ] = ctx

        if "ctx_upload" in st.session_state:
            render_results(
                "ctx_upload",
                models,
                st.session_state[
                    "ctx_upload"
                ].get(
                    "source_label",
                    "video",
                ),
            )


# ==========================================================
# LOCAL VIDEO
# ==========================================================

with tab2:
    st.header("📁 Select Local Video")

    video_files = core.list_local_videos(
        VIDEO_DIR
    )

    if video_files:
        selected_video = st.selectbox(
            "Choose a video:",
            video_files,
        )

        if st.button(
            "🔍 Analyze Video",
            key="analyze_local",
        ):
            with st.spinner(
                "Analyzing binary status and summary..."
            ):
                ctx = analyze_basic(
                    selected_video,
                    models,
                )

            if ctx is None:
                st.error(
                    "Could not read frames from the video."
                )

            else:
                ctx["source_label"] = (
                    selected_video
                )

                st.session_state[
                    "ctx_local"
                ] = ctx

        if "ctx_local" in st.session_state:
            render_results(
                "ctx_local",
                models,
                st.session_state[
                    "ctx_local"
                ].get(
                    "source_label",
                    "video",
                ),
            )

    else:
        st.warning(
            f"No video files found in {VIDEO_DIR}"
        )


# ==========================================================
# SETTINGS
# ==========================================================

with tab3:
    st.header("⚙️ Settings")

    metadata = models.get(
        "metadata",
        {},
    )

    st.write(
        "**Current Deployment Configuration:**"
    )

    st.json({
        "MULTITASK_MODEL_DIR":
            MULTITASK_MODEL_DIR,

        "BINARY_MODEL_DIR":
            binary_ensemble.BINARY_MODEL_DIR,

        "SUMMARY_CHECKPOINT":
            SUMMARY_CHECKPOINT,

        "DEVICE":
            DEVICE,

        "FRAMES":
            FRAMES,

        "MULTITASK_BACKBONE":
            metadata.get(
                "model_id",
                "Unknown",
            ),

        "MULTITASK_CLASSES":
            metadata.get(
                "classes",
                {},
            ),
    })
