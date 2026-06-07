# ==========================================================
# BINARY CLASSIFICATION - ENSEMBLE (SimpleAdapter + LoRA)
# ==========================================================
# Replaces the sklearn binary_model with an ensemble of two
# torch heads (SimpleAdapter + LoRALinear) operating on a
# single video-level embedding from the base Qwen3-VL model.
#
# Required files (in BINARY_MODEL_DIR):
#   - input_dim.pkl
#   - simple_adapter.pt
#   - lora_model.pt
# ==========================================================

import os
import joblib
import torch
import torch.nn as nn

# Directory holding the binary ensemble artifacts. Defaults to the
# main MODEL_DIR but can be overridden (e.g. a Google Drive folder).
BINARY_MODEL_DIR = os.getenv(
    "BINARY_MODEL_DIR",
    os.getenv("MODEL_DIR", os.path.join(os.getcwd(), "model")),
)


# ==========================================================
# MODEL DEFINITIONS
# ==========================================================

class SimpleAdapter(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.down = nn.Linear(d, 256)
        self.up = nn.Linear(256, d)
        self.relu = nn.ReLU()
        self.out = nn.Linear(d, 1)

    def forward(self, x):
        x = x + self.up(self.relu(self.down(x)))
        return self.out(x).squeeze(-1)


class LoRALinear(nn.Module):
    def __init__(self, d, r=8):
        super().__init__()
        self.base = nn.Linear(d, 1)
        self.A = nn.Linear(d, r, bias=False)
        self.B = nn.Linear(r, 1, bias=False)

    def forward(self, x):
        return (self.base(x) + self.B(self.A(x))).squeeze(-1)


# ==========================================================
# LOADING
# ==========================================================

def load_binary_models(device, model_dir=None):
    """Load the ensemble heads. `model_dir` defaults to BINARY_MODEL_DIR."""
    model_dir = model_dir or BINARY_MODEL_DIR

    d = joblib.load(os.path.join(model_dir, "input_dim.pkl"))

    simple = SimpleAdapter(d)
    simple.load_state_dict(
        torch.load(os.path.join(model_dir, "simple_adapter.pt"), map_location=device)
    )
    simple.to(device).eval()

    lora = LoRALinear(d)
    lora.load_state_dict(
        torch.load(os.path.join(model_dir, "lora_model.pt"), map_location=device)
    )
    lora.to(device).eval()

    return {"simple": simple, "lora": lora, "dim": d}


# ==========================================================
# EMBEDDING (video-level, base Qwen3-VL)
# ==========================================================

def extract_binary_embedding(frames, qwen_model, processor, device):
    """Build a single video-level embedding from all frames using the
    base Qwen3-VL model. If `qwen_model` is a PEFT/LoRA-wrapped model,
    its adapter is disabled so the embedding matches the base model the
    ensemble heads were trained on."""
    convo = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the video briefly."},
                *[{"type": "image"} for _ in frames],
            ],
        }
    ]

    text = processor.apply_chat_template(
        convo, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(images=list(frames), text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        if hasattr(qwen_model, "disable_adapter"):
            with qwen_model.disable_adapter():
                outputs = qwen_model(**inputs, output_hidden_states=True)
        else:
            outputs = qwen_model(**inputs, output_hidden_states=True)

    emb = outputs.hidden_states[-1][:, -1, :]
    return emb.squeeze(0).float()


# ==========================================================
# PREDICTION
# ==========================================================

def predict_binary(frames, binary_models, qwen_model, processor, device):
    """Return (status, confidence) using the averaged ensemble probability.

    status is 'ANOMALOUS' if final prob >= 0.5 else 'NORMAL'.
    confidence is the probability of the predicted class.
    """
    emb = extract_binary_embedding(frames, qwen_model, processor, device)
    emb = emb.unsqueeze(0).to(device)

    with torch.no_grad():
        p1 = torch.sigmoid(binary_models["simple"](emb)).item()
        p2 = torch.sigmoid(binary_models["lora"](emb)).item()

    final = (p1 + p2) / 2.0

    if final >= 0.5:
        return "ANOMALOUS", final, {"simple": p1, "lora": p2}
    return "NORMAL", 1.0 - final, {"simple": p1, "lora": p2}
