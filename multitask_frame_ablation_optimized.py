# ==============================================================================
# UCFCrime - MULTI-TASK FRAME ABLATION + MULTI-OBJECTIVE OPTIMIZATION
# Tasks: Action + Weapon + Location + People Count
# Backbone: Qwen3-VL-Embedding-2B (frozen) -> MultiTaskTemporalAdapter
#
# For each frame count this script:
#   1. Extracts frozen per-frame embeddings and trains the multi-task adapter.
#   2. Evaluates all 4 tasks (accuracy / macro-F1 / weighted-F1) and measures:
#        quality : mean accuracy / mean macro-F1 / mean weighted-F1 (maximize)
#        latency : embedding extraction sec/video                    (minimize)
#        power   : real GPU watts via NVML (pynvml)                  (minimize)
#        energy  : power x latency = Joules per video                (minimize)
#   3. Picks the optimum frame count with FOUR algorithms:
#        Pareto frontier | TOPSIS | Kneedle knee | desirability
# ==============================================================================

# pip install -q "transformers>=4.51.0" opencv-python scikit-learn tqdm pillow \
#     torch torchvision pynvml pandas matplotlib

import os
import json
import re
import time
import threading
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModel

# ==============================================================================
# CONFIG
# ==============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIDEO_ROOT = "/content/drive/MyDrive/Project_VLM/All_Videos"
TRAIN_JSON = "/content/UCFCrime_Train_Summary_wind_UPDATED_FINAL.json"
TEST_JSON = "/content/UCFCrime_Test_Summary_wind_UPDATED_FINAL.json"

OUT_DIR = "/content/drive/MyDrive/Project_VLM/multitask_frame_ablation"
os.makedirs(OUT_DIR, exist_ok=True)

FRAME_SETTINGS = [4, 8, 16, 32, 64]

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4

QWEN_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"

EMBED_INSTRUCTION = (
    "Represent this surveillance video frame for crime activity retrieval."
)

# Power model fallback (only used if NVML is unavailable)
GPU_IDLE_WATTS = 40.0
GPU_PEAK_WATTS = 300.0

# Multi-objective definition: +1 maximize, -1 minimize
OBJECTIVES = {
    "accuracy":    {"direction": +1, "weight": 0.25},
    "macro_f1":    {"direction": +1, "weight": 0.20},
    "weighted_f1": {"direction": +1, "weight": 0.15},
    "latency":     {"direction": -1, "weight": 0.15},
    "power":       {"direction": -1, "weight": 0.10},
    "energy":      {"direction": -1, "weight": 0.15},
}

TASKS = ["action", "weapon", "location", "people"]

# ==============================================================================
# SUPER CLASS MAP
# ==============================================================================

SUPER_CLASS_MAP = {
    "Abuse": "Violence", "Assault": "Violence", "Fighting": "Violence",
    "Shooting": "Violence",
    "Stealing": "Theft", "Robbery": "Theft", "Burglary": "Theft",
    "Shoplifting": "Theft",
    "Arson": "PropertyDamage", "Vandalism": "PropertyDamage",
    "Explosion": "PropertyDamage",
    "Arrest": "Police",
    "RoadAccidents": "Accident",
}


def to_super(label):
    return SUPER_CLASS_MAP.get(label, label)

# ==============================================================================
# LOAD QWEN (FROZEN)
# ==============================================================================

print("Loading Qwen embedding model...")
processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
qwen_model = AutoModel.from_pretrained(
    QWEN_MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    trust_remote_code=True,
).to(DEVICE)
qwen_model.eval()
for p in qwen_model.parameters():
    p.requires_grad = False
print("Qwen frozen")

# ==============================================================================
# WARM-UP RUN (to exclude initialization time from latency)
# ==============================================================================
print("Warming up model (excluding initialization time from latency)...")
warmup_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
warmup_txt = processor.apply_chat_template(
    [{"role": "user", "content": [
        {"type": "image", "image": warmup_img},
        {"type": "text", "text": "Warm up."]}]},
    tokenize=False, add_generation_prompt=False)
warmup_inp = processor(text=[warmup_txt], images=[warmup_img], return_tensors="pt").to(DEVICE)
with torch.no_grad():
    _ = qwen_model(**warmup_inp, output_hidden_states=True)
print("Warm-up complete.")

# ==============================================================================
# PROBE EMBEDDING DIM
# ==============================================================================

with torch.no_grad():
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    txt = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": EMBED_INSTRUCTION}]}],
        tokenize=False, add_generation_prompt=False)
    inp = processor(text=[txt], images=[img], return_tensors="pt").to(DEVICE)
    out = qwen_model(**inp, output_hidden_states=True)
    HIDDEN_DIM = out.hidden_states[-1].shape[-1]
print("Embedding dim:", HIDDEN_DIM)

# ==============================================================================
# VIDEO INDEX
# ==============================================================================

def normalize_name(x):
    x = x.lower()
    x = re.sub(r"\([^)]*\)", "", x)
    x = re.sub(r"_slow\d*|_skip\d*", "", x)
    x = re.sub(r"\.mp4|\.avi|\.mov|\.mkv", "", x)
    return x.strip()


print("Indexing videos...")
video_index = {}
for root, _, files in os.walk(VIDEO_ROOT):
    for f in files:
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_index[normalize_name(f)] = os.path.join(root, f)
print("Videos indexed:", len(video_index))


def find_video(name):
    return video_index.get(normalize_name(name), None)

# ==============================================================================
# FRAME SAMPLING + EMBEDDING
# ==============================================================================

def sample_frames(path, k):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        indices = np.linspace(0, total - 1, k, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    cap.release()
    if not frames:
        frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(k)]
    return frames


def embed_frames(frames):
    embs = []
    for frame in frames:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        txt = processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": EMBED_INSTRUCTION}]}],
            tokenize=False, add_generation_prompt=False)
        inp = processor(text=[txt], images=[img], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = qwen_model(**inp, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :]
            h = F.normalize(h.float(), dim=-1)
        embs.append(h.squeeze(0).cpu())
    return torch.stack(embs)

# ==============================================================================
# NVML POWER SAMPLER
# ==============================================================================

class PowerSampler:
    """Polls GPU power draw in a background thread; reports mean watts."""
    def __init__(self, interval=0.2, gpu_index=0):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self.ok = False
        try:
            import pynvml
            self.pynvml = pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.ok = True
        except Exception as e:
            print(f"NVML unavailable ({e}); will use modeled power.")

    def _loop(self):
        while not self._stop.is_set():
            try:
                mw = self.pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.samples.append(mw / 1000.0)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        if not self.ok:
            return
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.ok:
            return None
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return float(np.mean(self.samples)) if self.samples else None


def modeled_power(frames):
    fmin, fmax = min(FRAME_SETTINGS), max(FRAME_SETTINGS)
    frac = 0.0 if fmax == fmin else (frames - fmin) / (fmax - fmin)
    return GPU_IDLE_WATTS + frac * (GPU_PEAK_WATTS - GPU_IDLE_WATTS)

# ==============================================================================
# BUILD DATASET (with optional latency + power measurement)
# ==============================================================================

def build_dataset(json_path, frames, measure_cost=False):
    with open(json_path) as f:
        records = json.load(f)

    seqs = []
    action_labels, weapon_labels, location_labels, people_labels = [], [], [], []
    latencies = []

    sampler = PowerSampler() if measure_cost else None
    if sampler:
        sampler.start()

    for item in tqdm(records):
        video = item["video_id"]
        path = item.get("video_path")
        if not path or not os.path.exists(path):
            path = find_video(video)
        if not path:
            continue
        try:
            t0 = time.time()
            sampled_frames = sample_frames(path, frames)
            seq = embed_frames(sampled_frames)
            latencies.append(time.time() - t0)

            seqs.append(seq)
            action_labels.append(to_super(item["action_type"].split(",")[0]))
            weapon_labels.append(item.get("weapon_used", "None"))
            location_labels.append(item.get("location", "Unknown"))
            people_labels.append(str(item.get("people_count", "Unknown")))
        except Exception as e:
            print(video, e)

    avg_power = sampler.stop() if sampler else None
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    return (seqs, action_labels, weapon_labels, location_labels, people_labels,
            mean_latency, avg_power)

# ==============================================================================
# DATASET / COLLATE
# ==============================================================================

class VideoDataset(Dataset):
    def __init__(self, seqs, action, weapon, location, people):
        self.seqs = seqs
        self.action = action
        self.weapon = weapon
        self.location = location
        self.people = people

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return {
            "frames": self.seqs[i],
            "action": torch.tensor(self.action[i]),
            "weapon": torch.tensor(self.weapon[i]),
            "location": torch.tensor(self.location[i]),
            "people": torch.tensor(self.people[i]),
        }


def collate(batch):
    return {
        "frames": torch.stack([b["frames"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "weapon": torch.stack([b["weapon"] for b in batch]),
        "location": torch.stack([b["location"] for b in batch]),
        "people": torch.stack([b["people"] for b in batch]),
    }

# ==============================================================================
# MULTITASK TEMPORAL ADAPTER
# ==============================================================================

class MultiTaskTemporalAdapter(nn.Module):
    def __init__(self, in_dim, proj_dim, nhead, layers, dropout, max_frames,
                 action_classes, weapon_classes, location_classes, people_classes):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, proj_dim))
        self.proj = nn.Linear(in_dim, proj_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_frames + 1, proj_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.action_head = nn.Linear(proj_dim, action_classes)
        self.weapon_head = nn.Linear(proj_dim, weapon_classes)
        self.location_head = nn.Linear(proj_dim, location_classes)
        self.people_head = nn.Linear(proj_dim, people_classes)

    def forward(self, x):
        B, T, _ = x.shape
        x = self.proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :T + 1]
        x = self.encoder(x)
        x = x[:, 0]
        return {
            "action": self.action_head(x),
            "weapon": self.weapon_head(x),
            "location": self.location_head(x),
            "people": self.people_head(x),
        }

# ==============================================================================
# MULTI-OBJECTIVE OPTIMIZATION
# ==============================================================================

def to_matrix(df):
    cols = list(OBJECTIVES.keys())
    X = df[cols].to_numpy(dtype=float).copy()
    directions = np.array([OBJECTIVES[c]["direction"] for c in cols])
    weights = np.array([OBJECTIVES[c]["weight"] for c in cols])
    return X, directions, weights / weights.sum(), cols


def pareto_mask(df):
    X, directions, _, _ = to_matrix(df)
    Y = X * directions
    n = len(Y)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(Y[j] >= Y[i]) and np.any(Y[j] > Y[i]):
                mask[i] = False
                break
    return mask


def topsis(df):
    X, directions, weights, _ = to_matrix(df)
    norm = np.sqrt((X ** 2).sum(axis=0)); norm[norm == 0] = 1.0
    V = (X / norm) * weights
    best = np.where(directions > 0, V.max(0), V.min(0))
    worst = np.where(directions > 0, V.min(0), V.max(0))
    d_best = np.sqrt(((V - best) ** 2).sum(1))
    d_worst = np.sqrt(((V - worst) ** 2).sum(1))
    denom = d_best + d_worst; denom[denom == 0] = 1.0
    return d_worst / denom


def desirability(df):
    X, directions, weights, _ = to_matrix(df)
    d = np.zeros_like(X)
    for j in range(X.shape[1]):
        col = X[:, j]; lo, hi = col.min(), col.max(); rng = hi - lo
        if rng == 0:
            d[:, j] = 1.0
        elif directions[j] > 0:
            d[:, j] = (col - lo) / rng
        else:
            d[:, j] = (hi - col) / rng
    d = np.clip(d, 1e-6, 1.0)
    return np.exp((weights * np.log(d)).sum(1) / weights.sum())


def knee_point(df, cost="energy", gain="accuracy"):
    d = df.sort_values(cost).reset_index(drop=True)
    x = d[cost].to_numpy(float); y = d[gain].to_numpy(float)
    xr = (x - x.min()) / (x.max() - x.min() + 1e-12)
    yr = (y - y.min()) / (y.max() - y.min() + 1e-12)
    x1, y1, x2, y2 = xr[0], yr[0], xr[-1], yr[-1]
    denom = np.hypot(x2 - x1, y2 - y1) + 1e-12
    dist = np.abs((y2 - y1) * xr - (x2 - x1) * yr + x2 * y1 - y2 * x1) / denom
    return int(d.iloc[int(np.argmax(dist))]["frames"])

# ==============================================================================
# MAIN FRAME STUDY
# ==============================================================================

all_results = []     # detailed per-task rows (original output)
opt_rows = []        # aggregated rows for multi-objective optimization

for FRAME_COUNT in FRAME_SETTINGS:
    print("\n" + "=" * 80)
    print(f"FRAME COUNT: {FRAME_COUNT}")
    print("=" * 80)

    print("\nExtracting TRAIN embeddings...")
    (train_seqs, train_action, train_weapon, train_location, train_people,
     _, _) = build_dataset(TRAIN_JSON, FRAME_COUNT, measure_cost=False)

    print("\nExtracting TEST embeddings (measuring latency + power)...")
    (test_seqs, test_action, test_weapon, test_location, test_people,
     lat, pw) = build_dataset(TEST_JSON, FRAME_COUNT, measure_cost=True)

    if len(train_seqs) == 0 or len(test_seqs) == 0:
        print(f"[frames={FRAME_COUNT}] no embeddings - skipping")
        continue

    # ---- label encoders ----
    action_le, weapon_le = LabelEncoder(), LabelEncoder()
    location_le, people_le = LabelEncoder(), LabelEncoder()
    y_action_train = action_le.fit_transform(train_action)
    y_weapon_train = weapon_le.fit_transform(train_weapon)
    y_location_train = location_le.fit_transform(train_location)
    y_people_train = people_le.fit_transform(train_people)

    # guard against unseen labels in test set
    def safe_transform(le, labels):
        known = set(le.classes_)
        fallback = le.transform([le.classes_[0]])[0]
        return np.array([le.transform([l])[0] if l in known else fallback
                         for l in labels])

    y_action_test = safe_transform(action_le, test_action)
    y_weapon_test = safe_transform(weapon_le, test_weapon)
    y_location_test = safe_transform(location_le, test_location)
    y_people_test = safe_transform(people_le, test_people)

    train_dl = DataLoader(
        VideoDataset(train_seqs, y_action_train, y_weapon_train,
                     y_location_train, y_people_train),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    test_dl = DataLoader(
        VideoDataset(test_seqs, y_action_test, y_weapon_test,
                     y_location_test, y_people_test),
        batch_size=BATCH_SIZE, collate_fn=collate)

    model = MultiTaskTemporalAdapter(
        in_dim=HIDDEN_DIM, proj_dim=128, nhead=2, layers=1, dropout=0.1,
        max_frames=FRAME_COUNT,
        action_classes=len(action_le.classes_),
        weapon_classes=len(weapon_le.classes_),
        location_classes=len(location_le.classes_),
        people_classes=len(people_le.classes_)).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # ---- train ----
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_dl:
            x = batch["frames"].to(DEVICE)
            out = model(x)
            loss = (loss_fn(out["action"], batch["action"].to(DEVICE))
                    + loss_fn(out["weapon"], batch["weapon"].to(DEVICE))
                    + loss_fn(out["location"], batch["location"].to(DEVICE))
                    + loss_fn(out["people"], batch["people"].to(DEVICE)))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} Loss: {total_loss:.4f}")

    # ---- evaluate ----
    model.eval()
    results = {t: {"preds": [], "gts": []} for t in TASKS}
    with torch.no_grad():
        for batch in test_dl:
            out = model(batch["frames"].to(DEVICE))
            for task in TASKS:
                results[task]["preds"].extend(out[task].argmax(1).cpu().numpy())
                results[task]["gts"].extend(batch[task].numpy())

    # ---- metrics (per task) ----
    accs, macros, weighteds = [], [], []
    for task in TASKS:
        gts, preds = results[task]["gts"], results[task]["preds"]
        acc = accuracy_score(gts, preds)
        macro_f1 = f1_score(gts, preds, average="macro")
        weighted_f1 = f1_score(gts, preds, average="weighted")
        accs.append(acc); macros.append(macro_f1); weighteds.append(weighted_f1)
        print(f"\n{task.upper()}  acc={acc:.4f}  macroF1={macro_f1:.4f}  "
              f"weightedF1={weighted_f1:.4f}")
        all_results.append({"frames": FRAME_COUNT, "task": task,
                            "accuracy": round(acc, 4),
                            "macro_f1": round(macro_f1, 4),
                            "weighted_f1": round(weighted_f1, 4)})

    # ---- aggregated row for optimization (mean across the 4 tasks) ----
    if pw is None:
        pw = modeled_power(FRAME_COUNT)
        power_src = "modeled"
    else:
        power_src = "nvml"
    opt_rows.append({
        "frames": FRAME_COUNT,
        "accuracy": round(float(np.mean(accs)), 4),
        "macro_f1": round(float(np.mean(macros)), 4),
        "weighted_f1": round(float(np.mean(weighteds)), 4),
        "latency": round(lat, 4),
        "power": round(pw, 2),
        "energy": round(pw * lat, 2),
        "power_src": power_src,
    })

# ==============================================================================
# SAVE DETAILED PER-TASK RESULTS (original output)
# ==============================================================================

results_df = pd.DataFrame(all_results)
print("\n" + "=" * 80)
print("PER-TASK RESULTS")
print("=" * 80)
print(results_df.to_string(index=False))
detail_csv = os.path.join(OUT_DIR, "multitask_frame_ablation.csv")
results_df.to_csv(detail_csv, index=False)
print(f"\nSaved -> {detail_csv}")

# ==============================================================================
# MULTI-OBJECTIVE OPTIMIZATION (over aggregated metrics)
# ==============================================================================

if not opt_rows:
    raise SystemExit("No frame count completed - nothing to optimize.")

df = pd.DataFrame(opt_rows).sort_values("frames").reset_index(drop=True)
df["pareto"] = pareto_mask(df)
df["topsis"] = topsis(df).round(4)
df["desirability"] = desirability(df).round(4)

best_topsis = int(df.sort_values("topsis", ascending=False).iloc[0]["frames"])
best_desir = int(df.sort_values("desirability", ascending=False).iloc[0]["frames"])
best_knee = knee_point(df, cost="energy", gain="accuracy")

print("\n" + "=" * 90)
print("MULTI-TASK FRAME OPTIMIZATION (mean over action/weapon/location/people)")
print("=" * 90)
print(df.to_string(index=False))

opt_csv = os.path.join(OUT_DIR, "multitask_frame_optimization.csv")
df.to_csv(opt_csv, index=False)
print(f"\nSaved -> {opt_csv}")

print("\nOPTIMUM FRAME COUNT BY ALGORITHM")
print(f"  Pareto-optimal set : {sorted(df[df['pareto']]['frames'].tolist())}")
print(f"  TOPSIS             : {best_topsis}")
print(f"  Desirability       : {best_desir}")
print(f"  Knee (Acc/J)       : {best_knee}")

from collections import Counter
consensus = Counter([best_topsis, best_desir, best_knee]).most_common(1)[0][0]
print(f"\nRECOMMENDED FRAME COUNT (consensus): {consensus}")

# ==============================================================================
# PLOTS
# ==============================================================================

import matplotlib.pyplot as plt

frames = df["frames"].to_numpy()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

ax = axes[0, 0]
ax.scatter(df["energy"], df["accuracy"], s=120,
           c=["#d62728" if p else "#7f7f7f" for p in df["pareto"]], zorder=3)
pf = df[df["pareto"]].sort_values("energy")
ax.plot(pf["energy"], pf["accuracy"], "--", color="#d62728", alpha=0.7,
        label="Pareto front")
for _, r in df.iterrows():
    ax.annotate(f"{int(r['frames'])}f", (r["energy"], r["accuracy"]),
                textcoords="offset points", xytext=(6, 6))
ax.set_xlabel("Energy / video (J)", fontweight="bold")
ax.set_ylabel("Mean accuracy", fontweight="bold")
ax.set_title("Pareto: Accuracy vs Energy", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(df["latency"], df["accuracy"], "o-", color="#1f77b4")
for _, r in df.iterrows():
    ax.annotate(f"{int(r['frames'])}f", (r["latency"], r["accuracy"]),
                textcoords="offset points", xytext=(6, 6))
ax.set_xlabel("Latency / video (s)", fontweight="bold")
ax.set_ylabel("Mean accuracy", fontweight="bold")
ax.set_title("Accuracy vs Latency", fontweight="bold")
ax.grid(alpha=0.3)

ax = axes[0, 2]
w = 0.35; x = np.arange(len(df))
ax.bar(x - w / 2, df["topsis"], w, label="TOPSIS")
ax.bar(x + w / 2, df["desirability"], w, label="Desirability")
ax.set_xticks(x); ax.set_xticklabels([f"{int(f)}f" for f in frames])
ax.set_title("MCDM scores (higher=better)", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3, axis="y")

ax = axes[1, 0]
ax.plot(df["frames"], df["latency"], "o-", label="Latency (s)")
ax.plot(df["frames"], df["energy"] / df["energy"].max(), "s-", label="Energy (norm)")
ax.plot(df["frames"], df["power"] / df["power"].max(), "^-", label="Power (norm)")
ax.set_xlabel("Frames", fontweight="bold")
ax.set_title("Cost factors vs Frames", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1, 1]
for m in ["accuracy", "macro_f1", "weighted_f1"]:
    ax.plot(df["frames"], df[m], "o-", label=m)
ax.set_xlabel("Frames", fontweight="bold")
ax.set_title("Mean quality vs Frames", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1, 2]
d = df.sort_values("energy")
ax.plot(d["energy"], d["accuracy"], "o-", color="#9467bd")
krow = df[df["frames"] == best_knee].iloc[0]
ax.scatter([krow["energy"]], [krow["accuracy"]], s=260, facecolors="none",
           edgecolors="red", linewidths=2.5, label=f"Knee = {best_knee}f")
ax.set_xlabel("Energy / video (J)", fontweight="bold")
ax.set_ylabel("Mean accuracy", fontweight="bold")
ax.set_title("Knee-point (best trade-off)", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)

fig.suptitle(f"Multi-Task Frame-Count Optimization  -  recommended: {consensus} frames",
             fontsize=15, fontweight="bold")
fig.tight_layout()
out_plot = os.path.join(OUT_DIR, "multitask_frame_optimization.png")
fig.savefig(out_plot, dpi=160, bbox_inches="tight")
print(f"Saved -> {out_plot}")
