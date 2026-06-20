# ==========================================================
# VIDEO QA SYSTEM - CORE LOGIC (UI-agnostic)
# ==========================================================
# All model loading, frame sampling, embedding, summarization,
# classification and Q&A logic. No UI framework dependencies so
# it can be reused by Streamlit, NiceGUI, CLI, etc.

import os
import re
import cv2
import torch
import joblib
import random
import numpy as np
import gdown

from PIL import Image
from huggingface_hub import hf_hub_download

from transformers import AutoProcessor, AutoModel

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    from transformers import AutoModelForCausalLM as Qwen3VLForConditionalGeneration

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

try:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    APP_DIR = os.getcwd()

MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(APP_DIR, "model"))
CHECKPOINT = os.getenv("CHECKPOINT", os.path.join(MODEL_DIR, "checkpoint-140"))
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "")
TEMPORAL_ADAPTER_GDRIVE_ID = os.getenv("TEMPORAL_ADAPTER_GDRIVE_ID", "")
TEMPORAL_ADAPTER_GDRIVE_URL = os.getenv("TEMPORAL_ADAPTER_GDRIVE_URL", "")
BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
EMBED_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
FRAMES = 8

SAVE_DIR = os.getenv("SAVE_DIR", os.path.join(APP_DIR, "generated_summaries"))
VIDEO_DIR = os.getenv("VIDEO_DIR", os.path.join(APP_DIR, "Anomaly-Videos-Part-1"))
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_MAP = {
    0: "ANOMALOUS",
    1: "NORMAL",
}

MODEL_FILES = [
    "binary_model.pkl",
    "model_config.pkl",
    "le_weapon.pkl",
    "le_location.pkl",
    "le_people.pkl",
    "le_super.pkl",
    "temporal_adapter.pt",
    "checkpoint-140/adapter_config.json",
    "checkpoint-140/adapter_model.safetensors",
]


# ==========================================================
# TEMPORAL ADAPTER MODEL
# ==========================================================

class TemporalAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(
            cfg["in_dim"],
            cfg["n_ppl"] + cfg["n_wpn"] + cfg["n_loc"] + cfg["n_cat"],
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.fc(x)


# ==========================================================
# MODEL FILE MANAGEMENT
# ==========================================================

def extract_gdrive_file_id(value):
    if not value:
        return ""

    match = re.search(r"/file/d/([^/]+)", value)
    if match:
        return match.group(1)

    match = re.search(r"[?&]id=([^&]+)", value)
    if match:
        return match.group(1)

    return value.strip()


def download_temporal_adapter_from_gdrive():
    adapter_path = os.path.join(MODEL_DIR, "temporal_adapter.pt")
    if os.path.exists(adapter_path):
        return

    if TEMPORAL_ADAPTER_GDRIVE_URL:
        file_id = extract_gdrive_file_id(TEMPORAL_ADAPTER_GDRIVE_URL)
        gdown.download(id=file_id, output=adapter_path, quiet=False)
    elif TEMPORAL_ADAPTER_GDRIVE_ID:
        file_id = extract_gdrive_file_id(TEMPORAL_ADAPTER_GDRIVE_ID)
        gdown.download(id=file_id, output=adapter_path, quiet=False)


def ensure_model_files():
    os.makedirs(MODEL_DIR, exist_ok=True)
    download_temporal_adapter_from_gdrive()
    missing = [n for n in MODEL_FILES if not os.path.exists(os.path.join(MODEL_DIR, n))]

    if missing and MODEL_REPO_ID:
        for name in missing:
            downloaded_path = hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=name,
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False,
            )
            if not os.path.exists(downloaded_path):
                raise FileNotFoundError(f"Could not download {name} from {MODEL_REPO_ID}")
        missing = [n for n in MODEL_FILES if not os.path.exists(os.path.join(MODEL_DIR, n))]

    if missing:
        raise FileNotFoundError(
            "Missing model files in MODEL_DIR: " + ", ".join(missing) +
            f". Current MODEL_DIR: {MODEL_DIR}. Add these files under model/ "
            "or set MODEL_REPO_ID to a Hugging Face repo containing them."
        )


# ==========================================================
# LOAD MODELS (module-level cache)
# ==========================================================

_MODELS = None


def load_all_models(progress=None):
    """Load and cache all models. `progress` is an optional callable(str)
    used to report status messages to whatever UI is calling this."""
    global _MODELS
    if _MODELS is not None:
        return _MODELS

    def log(msg):
        if progress:
            progress(msg)

    log("Checking model files...")
    ensure_model_files()

    log("Loading classifier artifacts...")
    binary_model = joblib.load(f"{MODEL_DIR}/binary_model.pkl")
    config = joblib.load(f"{MODEL_DIR}/model_config.pkl")
    le_wpn = joblib.load(f"{MODEL_DIR}/le_weapon.pkl")
    le_loc = joblib.load(f"{MODEL_DIR}/le_location.pkl")
    le_ppl = joblib.load(f"{MODEL_DIR}/le_people.pkl")
    le_cat = joblib.load(f"{MODEL_DIR}/le_super.pkl")

    config["n_cat"] = len(le_cat.classes_)

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    log("Loading embedding model (Qwen3-VL-Embedding)...")
    qwen_processor = AutoProcessor.from_pretrained(EMBED_MODEL)
    qwen = AutoModel.from_pretrained(EMBED_MODEL, torch_dtype=dtype).to(DEVICE)
    qwen.eval()

    log("Loading summarization model (Qwen3-VL)...")
    summ_processor = AutoProcessor.from_pretrained(BASE_MODEL)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=dtype
    ).to(DEVICE)

    try:
        from peft import PeftModel
        summ_model = PeftModel.from_pretrained(base_model, CHECKPOINT)
        summ_model.to(DEVICE)
    except Exception as peft_error:
        log(f"LoRA checkpoint could not be loaded; using base model. {peft_error}")
        summ_model = base_model
    summ_model.eval()

    # Prefer the trained multi-class bundle (clf_*) over the temporal adapter.
    mc_bundle = None
    bundle_path = f"{MODEL_DIR}/multiclass_bundle.pkl"
    if os.path.exists(bundle_path):
        log("Loading multi-class bundle (clf_*)...")
        mc_bundle = joblib.load(bundle_path)

    temporal_adapter = None
    if mc_bundle is None:
        log("Loading temporal adapter...")
        temporal_adapter = TemporalAdapter(config)
        temporal_adapter.load_state_dict(
            torch.load(f"{MODEL_DIR}/temporal_adapter.pt", map_location=DEVICE, weights_only=True),
            strict=False,
        )
        temporal_adapter.to(DEVICE)
        temporal_adapter.eval()

    log("All models loaded.")

    _MODELS = {
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
        "temporal_adapter": temporal_adapter,
        "mc_bundle": mc_bundle,
    }
    return _MODELS


# ==========================================================
# FRAME SAMPLING
# ==========================================================

def sample_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
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
                    {"type": "text", "text": "Describe what you see."},
                ],
            }
        ]

        text = qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = qwen_processor(
            text=[text], images=[img], return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            out = qwen(**inputs, output_hidden_states=True)

        emb = out.hidden_states[-1][:, -1, :]
        emb = F.normalize(emb, dim=-1)
        embs.append(emb.squeeze(0).float().cpu().numpy())

    return np.stack(embs)


# ==========================================================
# MULTI-CLASS EMBEDDING + PREDICTION (trained clf_* bundle)
# ==========================================================

def _embed_text(text, models):
    """Last-token, L2-normalized embedding of a text-only prompt."""
    qwen = models["qwen"]
    proc = models["qwen_processor"]
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    chat = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = proc(text=[chat], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = qwen(**inputs, output_hidden_states=True)
    emb = F.normalize(out.hidden_states[-1][:, -1, :], dim=-1)
    return emb.squeeze(0).float().cpu().numpy()


def embed_for_multiclass(frames, summary, models):
    """Reproduce the training-time feature: per-frame visual embeddings
    (instruction + optional summary) averaged, fused 70/30 with a text-only
    summary embedding. Matches extract_qwen_features in the training script."""
    qwen = models["qwen"]
    proc = models["qwen_processor"]
    bundle = models["mc_bundle"]
    base_instr = bundle.get(
        "embed_instruction",
        "Represent this surveillance video for crime activity classification.",
    )
    summary = (summary or "").strip()
    img_instr = f"{base_instr}\n\nVideo Summary: {summary}" if summary else base_instr

    frame_embs = []
    for img in frames:
        messages = [
            {"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": img_instr}]}
        ]
        chat = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = proc(text=[chat], images=[img], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = qwen(**inputs, output_hidden_states=True)
        emb = F.normalize(out.hidden_states[-1][:, -1, :], dim=-1)
        frame_embs.append(emb.squeeze(0).float().cpu().numpy())

    visual = np.mean(frame_embs, axis=0)

    if summary:
        text_emb = _embed_text(f"Video Summary: {summary}\n\n{base_instr}", models)
        combined = 0.7 * visual + 0.3 * text_emb
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        return combined
    return visual


def predict_multiclass(frames, summary, models):
    """Predict people/weapon/location/category (+ multi-label actions) using
    the trained clf_* bundle. Returns (people, weapon, location, category, actions)."""
    bundle = models["mc_bundle"]
    vec = embed_for_multiclass(frames, summary, models)
    X = [vec]

    def _pred(clf_key, le_key):
        return bundle[le_key].inverse_transform(bundle[clf_key].predict(X))[0]

    people = _pred("clf_people", "le_people")
    weapon = _pred("clf_weapon", "le_weapon")
    location = _pred("clf_location", "le_location")
    category = _pred("clf_action_super", "le_super")

    actions = None
    try:
        clf_ml = bundle["clf_action_multilabel"]
        mlb = bundle["mlb_action"]
        proba = clf_ml.predict_proba(X)[0]
        thr = bundle.get("ml_best_threshold", 0.5)
        mask = proba >= thr
        if not mask.any():
            mask[int(np.argmax(proba))] = True
        actions = list(np.array(mlb.classes_)[mask])
    except Exception:
        actions = None

    return people, weapon, location, category, actions


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
        "confessed", "reported",
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

    # Use a middle frame; the first frame is often black/blank in surveillance clips.
    img = frames[len(frames) // 2]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe concisely."},
            ],
        }
    ]

    text = summ_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = summ_processor(
        text=[text], images=[img], return_tensors="pt"
    ).to(DEVICE)

    bad_words_ids = summ_processor.tokenizer(
        ["police", "court"], add_special_tokens=False
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

    emb_seq = embed_frames(frames, models)
    pooled = np.mean(emb_seq, axis=0)

    binary_pred = models["binary_model"].predict([pooled])[0]
    binary_proba = models["binary_model"].predict_proba([pooled])[0]
    binary_conf = binary_proba[binary_pred]

    multi_out = models["temporal_adapter"](
        torch.from_numpy(emb_seq).float().to(DEVICE)
    )
    multi_out = multi_out.cpu().detach().numpy()

    # Slice using the class counts the adapter was built with (config), not
    # len(le_*); a config/encoder mismatch would otherwise yield an empty slice.
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

    ppl_preds = _seg_argmax(0, n_ppl)
    wpn_preds = _seg_argmax(n_ppl, n_wpn)
    loc_preds = _seg_argmax(n_ppl + n_wpn, n_loc)
    cat_preds = _seg_argmax(n_ppl + n_wpn + n_loc, n_cat)

    summary = summarize_video(frames, models)

    context = {
        "binary_class": CLASS_MAP[binary_pred],
        "binary_confidence": float(binary_conf),
        "people": _safe_inv(models["le_ppl"], ppl_preds)[0],
        "weapon": _safe_inv(models["le_wpn"], wpn_preds)[0],
        "location": _safe_inv(models["le_loc"], loc_preds)[0],
        "category": _safe_inv(models["le_cat"], cat_preds)[0],
        "summary": summary,
        "frames": frames,
    }

    return context, frames


# ==========================================================
# Q&A FUNCTION
# ==========================================================

def answer_question(context, question):
    """Route questions to relevant context."""
    q_lower = question.lower()

    if any(w in q_lower for w in ["normal", "anomalous", "status", "safe"]):
        return (
            f"The video shows {context['binary_class']} activity "
            f"(confidence: {context['binary_confidence']:.2%})"
        )

    if any(w in q_lower for w in ["people", "person", "how many", "number", "count"]):
        return f"People detected: {context['people']}"

    if any(w in q_lower for w in ["weapon", "gun", "knife", "armed", "used"]):
        return f"Weapon type: {context['weapon']}"

    if any(w in q_lower for w in ["location", "where", "place", "located"]):
        return f"Location type: {context['location']}"

    if any(w in q_lower for w in ["category", "type", "event", "what is happening", "what"]):
        return f"Event category: {context['category']}. Summary: {context['summary']}"

    return context["summary"]


def list_local_videos(video_dir=None):
    video_dir = video_dir or VIDEO_DIR
    videos = []
    if os.path.exists(video_dir):
        for root, _dirs, files in os.walk(video_dir):
            for file in files:
                if file.endswith((".mp4", ".avi", ".mov", ".mkv")):
                    videos.append(os.path.join(root, file))
    return videos
