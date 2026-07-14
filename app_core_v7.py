# ==========================================================
# VIDEO QA SYSTEM - CORE LOGIC V7
# CACHED-QWEN MULTITASK TEMPORAL + SUMMARY-FUSION ARCHITECTURE
# ==========================================================

import os
import re
import cv2
import json
import pickle
import random
import time
from contextlib import nullcontext

# ==========================================================
# HUGGING FACE DOWNLOAD / CACHE CONFIGURATION
# Set before importing Transformers / huggingface_hub.
# ==========================================================

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

HF_LOCAL_FILES_ONLY = (
    os.getenv("HF_LOCAL_FILES_ONLY", "1").strip().lower()
    in {"1", "true", "yes", "on"}
)

import numpy as np
import joblib

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    from transformers import AutoModelForCausalLM as Qwen3VLForConditionalGeneration


# ==========================================================
# SEED / CONFIG
# ==========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    APP_DIR = os.getcwd()

MULTITASK_MODEL_DIR = os.getenv(
    "MULTITASK_MODEL_DIR",
    os.path.join(APP_DIR, "multitask"),
)

MULTITASK_MODEL_PATH = os.getenv(
    "MULTITASK_MODEL_PATH",
    os.path.join(MULTITASK_MODEL_DIR, "multitask_temporal_model.pkl"),
)

MULTITASK_METADATA_PATH = os.getenv(
    "MULTITASK_METADATA_PATH",
    os.path.join(MULTITASK_MODEL_DIR, "training_metadata.json"),
)

MULTITASK_ENCODERS_PATH = os.getenv(
    "MULTITASK_ENCODERS_PATH",
    os.path.join(MULTITASK_MODEL_DIR, "label_encoders.pkl"),
)

SUMMARY_CHECKPOINT = os.getenv(
    "SUMMARY_CHECKPOINT",
    os.getenv(
        "LORA_CHECKPOINT",
        os.path.join(APP_DIR, "summary"),
    ),
)

BASE_MODEL = os.getenv(
    "BASE_MODEL",
    "Qwen/Qwen3-VL-2B-Instruct",
)

# Training metadata should also contain the same model_id.
MULTITASK_BACKBONE = os.getenv(
    "MULTITASK_BACKBONE",
    BASE_MODEL,
)

FRAMES = int(os.getenv("NUM_FRAMES", "8"))

SAVE_DIR = os.getenv(
    "SAVE_DIR",
    os.path.join(APP_DIR, "generated_summaries"),
)

VIDEO_DIR = os.getenv(
    "VIDEO_DIR",
    os.path.join(APP_DIR, "Anomaly-Videos-Part-1"),
)

os.makedirs(SAVE_DIR, exist_ok=True)

# Backward-compatible aliases for the Streamlit UI.
MODEL_DIR = MULTITASK_MODEL_DIR
CHECKPOINT = SUMMARY_CHECKPOINT

TASKS = ["activity", "weapon", "people", "location"]

# Hybrid deployment:
#   Activity + Weapon -> OLD trained clf_* bundle
#   People + Location -> NEW Temporal Adapter + Summary Fusion model
OLD_MULTICLASS_BUNDLE_PATH = os.getenv(
    "OLD_MULTICLASS_BUNDLE_PATH",
    os.path.join(APP_DIR, "old_multiclass", "multiclass_bundle.pkl"),
)


# ==========================================================
# MODEL ARCHITECTURE - MUST MATCH TRAINING CODE
# ==========================================================

class TemporalAdapter(nn.Module):
    def __init__(
        self,
        input_dim,
        temporal_dim=512,
        nhead=8,
        num_layers=2,
        dropout=0.20,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim,
            nhead=nhead,
            dim_feedforward=temporal_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_norm = nn.LayerNorm(temporal_dim)

    def forward(self, frame_features, frame_mask):
        x = self.input_projection(frame_features)

        x = self.temporal_encoder(
            x,
            src_key_padding_mask=~frame_mask,
        )

        mask = frame_mask.unsqueeze(-1).to(x.dtype)

        pooled = (
            (x * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

        return self.output_norm(pooled)


class SummaryProjector(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=512,
        dropout=0.20,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class FusionModule(nn.Module):
    def __init__(
        self,
        temporal_dim=512,
        summary_dim=512,
        fusion_dim=512,
        dropout=0.20,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(temporal_dim + summary_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, temporal_feature, summary_feature):
        fused = torch.cat(
            [temporal_feature, summary_feature],
            dim=-1,
        )
        return self.network(fused)


class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        dropout=0.20,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class MultiTaskTemporalModel(nn.Module):
    def __init__(self, metadata):
        super().__init__()

        backbone_dim = int(metadata["backbone_dim"])
        temporal_dim = int(metadata.get("temporal_dim", 512))
        summary_dim = int(metadata.get("summary_dim", 512))
        fusion_dim = int(metadata.get("fusion_dim", 512))
        nhead = int(metadata.get("nhead", 8))
        num_layers = int(metadata.get("num_temporal_layers", 2))
        dropout = float(metadata.get("dropout", 0.20))

        classes = metadata["classes"]

        self.temporal_adapter = TemporalAdapter(
            input_dim=backbone_dim,
            temporal_dim=temporal_dim,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.summary_projector = SummaryProjector(
            input_dim=backbone_dim,
            output_dim=summary_dim,
            dropout=dropout,
        )

        self.fusion = FusionModule(
            temporal_dim=temporal_dim,
            summary_dim=summary_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
        )

        self.heads = nn.ModuleDict({
            task: ClassificationHead(
                input_dim=fusion_dim,
                num_classes=len(classes[task]),
                dropout=dropout,
            )
            for task in TASKS
        })

    def forward(
        self,
        frame_features,
        frame_mask,
        summary_feature,
    ):
        temporal_feature = self.temporal_adapter(
            frame_features,
            frame_mask,
        )

        summary_feature = self.summary_projector(
            summary_feature,
        )

        fused_feature = self.fusion(
            temporal_feature,
            summary_feature,
        )

        return {
            task: self.heads[task](fused_feature)
            for task in TASKS
        }


# ==========================================================
# FILE VALIDATION
# ==========================================================

MULTITASK_REQUIRED = [
    "temporal_adapter.pkl",
    "summary_projector.pkl",
    "fusion.pkl",
    "activity_head.pkl",
    "weapon_head.pkl",
    "people_head.pkl",
    "location_head.pkl",
    "label_encoders.pkl",
    "multitask_temporal_model.pkl",
    "training_metadata.json",
]


def ensure_model_files():
    missing = [
        name
        for name in MULTITASK_REQUIRED
        if not os.path.exists(
            os.path.join(MULTITASK_MODEL_DIR, name)
        )
    ]

    if missing:
        raise FileNotFoundError(
            "Missing NEW multitask files in MULTITASK_MODEL_DIR: "
            + ", ".join(missing)
            + f". Current MULTITASK_MODEL_DIR: {MULTITASK_MODEL_DIR}"
        )

    if not os.path.exists(OLD_MULTICLASS_BUNDLE_PATH):
        raise FileNotFoundError(
            "Old multiclass bundle not found: "
            f"{OLD_MULTICLASS_BUNDLE_PATH}. "
            "Set OLD_MULTICLASS_BUNDLE_PATH to multiclass_bundle.pkl."
        )

    if not os.path.isdir(SUMMARY_CHECKPOINT):
        raise FileNotFoundError(
            f"Summary checkpoint directory not found: {SUMMARY_CHECKPOINT}"
        )


# ==========================================================
# CHECKPOINT HELPERS
# ==========================================================

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_state_dict_from_pkl(path):
    checkpoint = load_pickle(path)

    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    raise KeyError(
        f"No model_state_dict/state_dict found in {path}"
    )


# ==========================================================
# LOAD MODELS
# ==========================================================

_MODELS = None


def load_all_models(progress=None):
    """
    V7 cached-Qwen deployment.

    The V6 single-backbone architecture is preserved:
      - ONE Qwen3-VL backbone
      - summarization LoRA attached to the shared backbone
      - frozen base-model feature extraction with LoRA disabled
      - OLD Activity / Weapon classifiers
      - NEW People / Location temporal + summary-fusion model

    V7 changes only the Hugging Face loading path:
      - Xet is disabled by the deployment notebook/process environment
      - the Qwen checkpoint is pre-downloaded before Streamlit starts
      - local_files_only=True is used by default
      - detailed loading timings expose cache / RAM / CUDA stages
    """
    global _MODELS

    if _MODELS is not None:
        return _MODELS

    def log(message):
        print(f"[CORE LOAD] {message}", flush=True)
        if progress:
            progress(message)

    log("1/8 Checking model files...")
    ensure_model_files()

    log("2/8 Loading multitask metadata and label encoders...")
    with open(
        MULTITASK_METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    encoders = load_pickle(MULTITASK_ENCODERS_PATH)

    log("3/8 Loading old Activity / Weapon classifier bundle...")
    old_mc_bundle = joblib.load(OLD_MULTICLASS_BUNDLE_PATH)

    required_old_keys = [
        "clf_action_fine",
        "le_action_fine",
        "clf_weapon",
        "le_weapon",
    ]
    missing_old_keys = [
        key for key in required_old_keys
        if key not in old_mc_bundle
    ]
    if missing_old_keys:
        raise KeyError(
            "Old multiclass bundle is missing required keys: "
            + ", ".join(missing_old_keys)
        )

    backbone_id = metadata.get("model_id", MULTITASK_BACKBONE)

    if backbone_id != BASE_MODEL:
        raise ValueError(
            "Single-backbone deployment requires the multitask backbone "
            f"and summarization base model to match. metadata model_id={backbone_id!r}, "
            f"BASE_MODEL={BASE_MODEL!r}."
        )

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    log(
        f"4/8 Loading cached shared processor: {backbone_id} "
        f"(local_files_only={HF_LOCAL_FILES_ONLY})"
    )
    processor_start = time.time()
    shared_processor = AutoProcessor.from_pretrained(
        backbone_id,
        trust_remote_code=True,
        local_files_only=HF_LOCAL_FILES_ONLY,
    )
    log(
        "4/8 Cached shared processor ready in "
        f"{time.time() - processor_start:.2f} sec."
    )

    log(
        "5A/8 Loading cached Qwen3-VL checkpoint into memory "
        f"(local_files_only={HF_LOCAL_FILES_ONLY})..."
    )
    checkpoint_start = time.time()
    shared_base_model = (
        Qwen3VLForConditionalGeneration
        .from_pretrained(
            backbone_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=HF_LOCAL_FILES_ONLY,
        )
    )
    log(
        "5B/8 Cached Qwen checkpoint loaded in "
        f"{time.time() - checkpoint_start:.2f} sec."
    )

    log(f"5C/8 Moving ONE shared Qwen3-VL backbone to {DEVICE}...")
    device_start = time.time()
    shared_base_model = shared_base_model.to(DEVICE)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        reserved_gb = torch.cuda.memory_reserved() / 1024**3
        log(
            "5D/8 Shared Qwen moved to CUDA in "
            f"{time.time() - device_start:.2f} sec. "
            f"GPU allocated={allocated_gb:.2f} GB, "
            f"reserved={reserved_gb:.2f} GB."
        )
    else:
        log(
            "5D/8 Shared Qwen moved to CPU in "
            f"{time.time() - device_start:.2f} sec."
        )

    shared_base_model.eval()

    for parameter in shared_base_model.parameters():
        parameter.requires_grad = False

    log("6/8 Attaching summarization LoRA to the shared backbone...")
    try:
        from peft import PeftModel

        shared_model = PeftModel.from_pretrained(
            shared_base_model,
            SUMMARY_CHECKPOINT,
            is_trainable=False,
        )
        shared_model.to(DEVICE)
        shared_model.eval()

    except Exception as error:
        log(
            "WARNING: LoRA checkpoint could not be loaded; "
            f"using the shared base model. Error: {error}"
        )
        shared_model = shared_base_model

    log("7/8 Loading trained multitask temporal / fusion model...")
    multitask_model = MultiTaskTemporalModel(metadata)
    state_dict = load_state_dict_from_pkl(MULTITASK_MODEL_PATH)
    multitask_model.load_state_dict(state_dict, strict=True)
    multitask_model.to(DEVICE)
    multitask_model.eval()

    log("8/8 Finalizing V7 cached-Qwen hybrid model bundle...")

    _MODELS = {
        "metadata": metadata,
        "encoders": encoders,
        "qwen": shared_model,
        "qwen_processor": shared_processor,
        "multitask_model": multitask_model,
        "old_mc_bundle": old_mc_bundle,
        "summ_model": shared_model,
        "summ_processor": shared_processor,
    }

    log(
        "READY: V7 cached shared Qwen backbone + LoRA summary adapter + "
        "OLD Activity/Weapon + NEW People/Location."
    )

    return _MODELS


# ==========================================================
# FRAME SAMPLING
# ==========================================================

def sample_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    total = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total <= 0:
        cap.release()
        return []

    count = min(FRAMES, total)

    indices = np.linspace(
        0,
        total - 1,
        num=count,
    ).astype(int)

    frames = []

    for index in indices:
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index),
        )

        ok, frame = cap.read()

        if not ok or frame is None:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        # IMPORTANT:
        # Do not force 448x448 here. Training passed PIL images
        # directly to the Qwen processor.
        frames.append(
            Image.fromarray(frame)
        )

    cap.release()

    return frames


# ==========================================================
# FEATURE EXTRACTION - MATCHES TRAINING
# ==========================================================

def model_input_device(model):
    return next(model.parameters()).device


def adapter_disabled(model):
    """
    Disable the summarization LoRA temporarily when the shared Qwen backbone
    is used for frozen feature extraction / old classifiers.
    """
    disable = getattr(model, "disable_adapter", None)
    return disable() if callable(disable) else nullcontext()


def qwen_base_model(model):
    """
    Return the underlying Qwen model from a PEFT wrapper.
    """
    getter = getattr(model, "get_base_model", None)
    return getter() if callable(getter) else model


def masked_mean(hidden, attention_mask):
    mask = (
        attention_mask
        .unsqueeze(-1)
        .to(hidden.dtype)
    )

    numerator = (
        hidden * mask
    ).sum(dim=1)

    denominator = (
        mask.sum(dim=1)
        .clamp(min=1)
    )

    return numerator / denominator


@torch.inference_mode()
def extract_frame_feature(model, processor, image):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": "Describe the visible scene and activity.",
            },
        ],
    }]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    inputs = processor(
        text=[prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    device = model_input_device(model)

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with adapter_disabled(model):
        base_model = qwen_base_model(model)
        outputs = base_model.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    pooled = masked_mean(
        outputs.last_hidden_state,
        inputs["attention_mask"],
    )

    return pooled[0].float()


@torch.inference_mode()
def generate_multitask_summary(
    model,
    processor,
    frames,
):
    """Generate the SAME style of summary used during multitask training."""

    content = [
        {"type": "image"}
        for _ in frames
    ]

    content.append({
        "type": "text",
        "text": (
            "Provide a concise 2-3 sentence factual summary "
            "of the surveillance video. Describe the main "
            "activity, visible weapon or object if any, "
            "approximate number of people, and scene/location. "
            "Describe only visually supported information."
        ),
    })

    messages = [{
        "role": "user",
        "content": content,
    }]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt],
        images=frames,
        padding=True,
        return_tensors="pt",
    )

    device = model_input_device(model)

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with adapter_disabled(model):
        generated = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids
        in zip(
            inputs["input_ids"],
            generated,
        )
    ]

    return (
        processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        .strip()
    )


@torch.inference_mode()
def extract_summary_feature(
    model,
    processor,
    summary,
):
    tokenizer = processor.tokenizer

    inputs = tokenizer(
        summary,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    device = model_input_device(model)

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with adapter_disabled(model):
        base_model = qwen_base_model(model)
        outputs = base_model.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )

    pooled = masked_mean(
        outputs.last_hidden_state,
        inputs["attention_mask"],
    )

    return pooled[0].float()


# ==========================================================
# HYBRID MULTITASK PREDICTION
# Activity + Weapon: OLD clf_* classifiers
# People + Location: NEW Temporal + Summary-Fusion model
# ==========================================================

@torch.inference_mode()
def _old_embed_text(text, models):
    """Training-style text embedding used by the old clf_* bundle."""
    qwen = models["qwen"]
    processor = models["qwen_processor"]

    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": text}],
    }]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    inputs = processor(
        text=[prompt],
        return_tensors="pt",
    )

    device = model_input_device(qwen)
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with adapter_disabled(qwen):
        outputs = qwen(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    embedding = F.normalize(
        outputs.hidden_states[-1][:, -1, :],
        dim=-1,
    )

    return (
        embedding[0]
        .float()
        .cpu()
        .numpy()
    )


@torch.inference_mode()
def embed_for_old_classifiers(
    frames,
    summary,
    models,
):
    """
    Reproduce the old multiclass training feature:
      per-frame last-token embedding
      -> L2 normalize
      -> mean visual feature
      -> 0.7 visual + 0.3 summary text embedding
      -> final L2 normalization
    """

    qwen = models["qwen"]
    processor = models["qwen_processor"]
    bundle = models["old_mc_bundle"]

    base_instruction = bundle.get(
        "embed_instruction",
        "Represent this surveillance video for crime activity classification.",
    )

    summary = (summary or "").strip()

    image_instruction = (
        f"{base_instruction}\n\nVideo Summary: {summary}"
        if summary
        else base_instruction
    )

    frame_embeddings = []

    for image in frames:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": image_instruction,
                },
            ],
        }]

        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        inputs = processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
        )

        device = model_input_device(qwen)
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        outputs = qwen(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        embedding = F.normalize(
            outputs.hidden_states[-1][:, -1, :],
            dim=-1,
        )

        frame_embeddings.append(
            embedding[0]
            .float()
            .cpu()
            .numpy()
        )

    visual_feature = np.mean(
        frame_embeddings,
        axis=0,
    )

    if summary:
        text_feature = _old_embed_text(
            f"Video Summary: {summary}\n\n{base_instruction}",
            models,
        )

        combined = (
            0.7 * visual_feature
            + 0.3 * text_feature
        )

        norm = np.linalg.norm(combined)

        if norm > 0:
            combined = combined / norm

        feature = combined
    else:
        feature = visual_feature

    expected_dim = bundle.get("feature_dim")

    if (
        expected_dim is not None
        and int(expected_dim) != int(feature.shape[0])
    ):
        raise ValueError(
            "Old classifier feature dimension mismatch. "
            f"Bundle expects {expected_dim}, "
            f"but current Qwen backbone produced {feature.shape[0]}. "
            "The old classifiers must use the same embedding backbone "
            "used during their training."
        )

    return feature


@torch.inference_mode()
def predict_multiclass(
    frames,
    summary,
    models,
):
    """
    HYBRID inference:
      Activity -> OLD clf_action_fine
      Weapon   -> OLD clf_weapon
      People   -> NEW people head
      Location -> NEW location head

    The old Activity/Weapon classifiers use the old 70/30
    visual-summary feature construction.

    The new People/Location heads use the exact trained
    Temporal Adapter + Summary Projector + Fusion architecture.
    """

    if not frames:
        return (
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
            None,
        )

    qwen = models["qwen"]
    processor = models["qwen_processor"]
    multitask_model = models["multitask_model"]
    encoders = models["encoders"]
    old_bundle = models["old_mc_bundle"]

    # ------------------------------------------------------
    # OLD ACTIVITY + WEAPON
    # ------------------------------------------------------

    old_feature = embed_for_old_classifiers(
        frames,
        summary,
        models,
    )

    old_X = old_feature.reshape(1, -1)

    activity_index = old_bundle[
        "clf_action_fine"
    ].predict(old_X)

    activity = old_bundle[
        "le_action_fine"
    ].inverse_transform(
        activity_index
    )[0]

    weapon_index = old_bundle[
        "clf_weapon"
    ].predict(old_X)

    weapon = old_bundle[
        "le_weapon"
    ].inverse_transform(
        weapon_index
    )[0]

    # Optional old multi-label actions for Q&A compatibility.
    actions = None

    try:
        clf_ml = old_bundle["clf_action_multilabel"]
        mlb = old_bundle["mlb_action"]

        probabilities = clf_ml.predict_proba(
            old_X
        )[0]

        threshold = float(
            old_bundle.get(
                "ml_best_threshold",
                0.5,
            )
        )

        mask = probabilities >= threshold

        if not mask.any():
            mask[int(np.argmax(probabilities))] = True

        actions = list(
            np.asarray(mlb.classes_)[mask]
        )

    except Exception:
        actions = None

    # ------------------------------------------------------
    # NEW PEOPLE + LOCATION
    # ------------------------------------------------------

    frame_features = torch.stack([
        extract_frame_feature(
            qwen,
            processor,
            image,
        )
        for image in frames
    ])

    classifier_summary = generate_multitask_summary(
        qwen,
        processor,
        frames,
    )

    summary_feature = extract_summary_feature(
        qwen,
        processor,
        classifier_summary,
    )

    frame_features = (
        frame_features
        .unsqueeze(0)
        .to(DEVICE)
    )

    frame_mask = torch.ones(
        1,
        frame_features.shape[1],
        dtype=torch.bool,
        device=DEVICE,
    )

    summary_feature = (
        summary_feature
        .unsqueeze(0)
        .to(DEVICE)
    )

    logits = multitask_model(
        frame_features,
        frame_mask,
        summary_feature,
    )

    people_index = int(
        logits["people"]
        .argmax(dim=1)
        .item()
    )

    people = encoders[
        "people"
    ].inverse_transform(
        [people_index]
    )[0]

    location_index = int(
        logits["location"]
        .argmax(dim=1)
        .item()
    )

    location = encoders[
        "location"
    ].inverse_transform(
        [location_index]
    )[0]

    # Preserve Streamlit return order:
    # people, weapon, location, category/activity, actions
    return (
        str(people),
        str(weapon),
        str(location),
        str(activity),
        actions,
    )


# ==========================================================
# UI SUMMARY - EXISTING QWEN + LoRA PATH
# ==========================================================

def clean_output(text):
    text = text.strip()

    if "assistant" in text.lower():
        text = text.split("assistant")[-1].strip()

    text = " ".join(text.split())

    sentences = re.split(
        r"(?<=[.!?]) +",
        text,
    )

    cleaned = []

    banned = [
        "police",
        "court",
        "judge",
        "lawsuit",
        "investigation",
        "sentenced",
        "prison",
        "confessed",
        "reported",
    ]

    for sentence in sentences:
        low = sentence.lower()

        if any(
            word in low
            for word in banned
        ):
            continue

        if sentence not in cleaned:
            cleaned.append(sentence)

        if len(cleaned) >= 4:
            break

    text = " ".join(cleaned)

    if text and not text.endswith("."):
        text += "."

    return text or "Unable to generate a reliable summary."


@torch.inference_mode()
def summarize_video(frames, models):
    summ_model = models["summ_model"]
    summ_processor = models["summ_processor"]

    if not frames:
        return "No frames to summarize."

    # Preserve the existing deployed LoRA summarization behavior.
    image = frames[len(frames) // 2]

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": "Describe concisely.",
            },
        ],
    }]

    prompt = summ_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = summ_processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
    ).to(DEVICE)

    bad_words_ids = (
        summ_processor.tokenizer(
            ["police", "court"],
            add_special_tokens=False,
        )
        .input_ids
    )

    output = summ_model.generate(
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

    generated_ids = output[
        0,
        inputs["input_ids"].shape[1]:,
    ]

    summary = summ_processor.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return clean_output(summary)


# ==========================================================
# Q&A
# ==========================================================

def answer_question(context, question):
    q_lower = question.lower()

    if any(
        word in q_lower
        for word in [
            "normal",
            "anomalous",
            "status",
            "safe",
        ]
    ):
        return (
            f"The video shows {context['binary_class']} activity "
            f"(confidence: {context['binary_confidence']:.2%})."
        )

    if any(
        word in q_lower
        for word in [
            "people",
            "person",
            "how many",
            "number",
            "count",
        ]
    ):
        return (
            f"People detected: {context['people']}"
        )

    if any(
        word in q_lower
        for word in [
            "weapon",
            "gun",
            "knife",
            "armed",
            "used",
        ]
    ):
        return (
            f"Weapon type: {context['weapon']}"
        )

    if any(
        word in q_lower
        for word in [
            "location",
            "where",
            "place",
            "located",
        ]
    ):
        return (
            f"Location type: {context['location']}"
        )

    if any(
        word in q_lower
        for word in [
            "activity",
            "category",
            "type",
            "event",
            "what is happening",
            "what",
        ]
    ):
        return (
            f"Activity: {context['category']}. "
            f"Summary: {context['summary']}"
        )

    return context["summary"]


def list_local_videos(video_dir=None):
    video_dir = video_dir or VIDEO_DIR
    videos = []

    if os.path.exists(video_dir):
        for root, _dirs, files in os.walk(video_dir):
            for filename in files:
                if filename.lower().endswith(
                    (".mp4", ".avi", ".mov", ".mkv")
                ):
                    videos.append(
                        os.path.join(root, filename)
                    )

    return sorted(videos)
