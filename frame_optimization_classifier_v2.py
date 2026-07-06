# ==========================================================
# OPTIMUM NUMBER OF FRAMES - BINARY ANOMALY CLASSIFIER
# Frozen Qwen3-VL-2B backbone -> embeddings -> trainable heads
# (LoRA head used as the PRIMARY optimization target)
#
# For each frame count in FRAME_COUNTS this script:
#   1. Extracts video embeddings with the FROZEN Qwen3-VL backbone
#      using that many uniformly sampled frames.
#   2. Trains binary heads on UCFCrime_Train (normal=0 / anomaly=1):
#        classical: LogReg / RandomForest / SVM
#        neural   : SimpleAdapter / LoRA / TemporalAdapter
#   3. Evaluates on UCFCrime_Test and measures:
#        quality : Accuracy / F1 / AUC   (maximize)   [PRIMARY = LoRA]
#        latency : embedding extraction sec/video      (minimize)
#        power   : real GPU watts via NVML (pynvml)     (minimize)
#        energy  : power x latency = Joules per video   (minimize)
#   4. Picks the optimum frame count with FOUR algorithms:
#        Pareto frontier | TOPSIS | Kneedle knee | desirability
# ==========================================================

!pip install -q transformers accelerate opencv-python pillow scikit-learn \
    pynvml pandas matplotlib

import os, json, cv2, re, gc, time, threading
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- CONFIG ----------------

# Drive may be mounted at /content/drive OR /content/drive2 depending on the
# session. Auto-resolve the Project_VLM root across the common candidates so
# the script works without manual edits.
def _resolve_project_root():
    candidates = [
        "/content/drive2/MyDrive/Project_VLM",
        "/content/drive/MyDrive/Project_VLM",
        "/content/drive/My Drive/Project_VLM",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "UCFCrime_Train.json")):
            return c
    # nothing found yet - try mounting Google Drive (Colab), then re-check
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    except Exception:
        pass
    for c in candidates:
        if os.path.exists(os.path.join(c, "UCFCrime_Train.json")):
            return c
    raise FileNotFoundError(
        "Could not find UCFCrime_Train.json under any of: " + ", ".join(candidates)
        + "\nMount Google Drive or set PROJECT_ROOT manually.")

PROJECT_ROOT = _resolve_project_root()
print("Using PROJECT_ROOT:", PROJECT_ROOT)

VIDEO_ROOT = os.path.join(PROJECT_ROOT, "All_Videos")
TRAIN_JSON = os.path.join(PROJECT_ROOT, "UCFCrime_Train.json")
TEST_JSON = os.path.join(PROJECT_ROOT, "UCFCrime_Test.json")

OUT_ROOT = os.path.join(PROJECT_ROOT, "frame_optimization_classifier")
os.makedirs(OUT_ROOT, exist_ok=True)

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

# Run only the NEW frame counts now; they will be merged with any previously
# saved results (e.g. the earlier 4/8/16/32 run) before optimization.
FRAME_COUNTS = [1, 2]

# If True, load the previously saved CSVs and combine with this run's results
# so the optimization is performed over ALL frame counts together. Newly
# computed frame counts overwrite any stale duplicates.
MERGE_WITH_EXISTING = True

# Optional caps to control cost (None = use all videos in the JSON)
MAX_TRAIN = None
MAX_TEST = None

# Which trained head drives the multi-objective optimization
PRIMARY_MODEL = "LoRA"

NN_EPOCHS = 10
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Power model fallback (only if NVML unavailable)
GPU_IDLE_WATTS = 40.0
GPU_PEAK_WATTS = 300.0

OBJECTIVES = {
    "accuracy": {"direction": +1, "weight": 0.30},
    "f1":       {"direction": +1, "weight": 0.20},
    "auc":      {"direction": +1, "weight": 0.15},
    "latency":  {"direction": -1, "weight": 0.15},
    "power":    {"direction": -1, "weight": 0.05},
    "energy":   {"direction": -1, "weight": 0.15},
}

# ----------------------------------------------------------
# LOAD QWEN (FROZEN)
# ----------------------------------------------------------
print("Loading Qwen model...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()
print("Frozen Qwen loaded.")

# ----------------------------------------------------------
# WARM-UP RUN (to exclude initialization time from latency)
# ----------------------------------------------------------
print("Warming up model (excluding initialization time from latency)...")
warmup_frames = sample_frames(find_video(list(video_index.keys())[0]), min(FRAME_COUNTS))
if warmup_frames:
    warmup_imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in warmup_frames[:1]]
    warmup_convo = [{"role": "user", "content": [
        {"type": "text", "text": "Warm up."},
        {"type": "image", "image": warmup_imgs[0]}]}]
    warmup_text = processor.apply_chat_template(warmup_convo, tokenize=False,
                                                 add_generation_prompt=True)
    warmup_inputs = processor(images=[warmup_imgs[0]], text=warmup_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        _ = model(**warmup_inputs, output_hidden_states=True)
print("Warm-up complete.")

# ----------------------------------------------------------
# VIDEO UTILITIES
# ----------------------------------------------------------
def normalize(x):
    return re.sub(r"\.mp4|\.avi|\.mov|\.mkv", "", x.lower())

video_index = {}
for root, _, files in os.walk(VIDEO_ROOT):
    for f in files:
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_index[normalize(f)] = os.path.join(root, f)
print("Indexed videos:", len(video_index))


def find_video(name):
    return video_index.get(normalize(name), None)


def sample_frames(path, k):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        idxs = [int(i * total / k) for i in range(k)]
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if ok:
                frames.append(f)
    cap.release()
    return frames


@torch.no_grad()
def extract_embedding(video_path, k):
    frames = sample_frames(video_path, k)
    if len(frames) == 0:
        return None
    imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    convo = [{"role": "user", "content": [
        {"type": "text", "text": "Describe the video briefly."},
        *[{"type": "image"} for _ in imgs]]}]
    text = processor.apply_chat_template(convo, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(images=imgs, text=text, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs, output_hidden_states=True)
    emb = outputs.hidden_states[-1][:, -1, :]
    return emb.squeeze(0).float().cpu().numpy()

# ============================================================
# NVML POWER SAMPLER
# ============================================================

class PowerSampler:
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
    fmin, fmax = min(FRAME_COUNTS), max(FRAME_COUNTS)
    frac = 0.0 if fmax == fmin else (frames - fmin) / (fmax - fmin)
    return GPU_IDLE_WATTS + frac * (GPU_PEAK_WATTS - GPU_IDLE_WATTS)

# ============================================================
# DATASET BUILDER (with latency + power measurement)
# ============================================================

def build_dataset(json_path, k, measure_cost=False, cap=None):
    with open(json_path) as f:
        data = json.load(f)
    keys = list(data.keys())
    if cap:
        keys = keys[:cap]

    X, y, latencies = [], [], []
    sampler = PowerSampler() if measure_cost else None
    if sampler:
        sampler.start()

    for video in tqdm(keys, desc=f"emb k={k} {os.path.basename(json_path)}"):
        path = find_video(video)
        if path is None:
            continue
        t0 = time.time()
        emb = extract_embedding(path, k)
        dt = time.time() - t0
        if emb is None:
            continue
        X.append(emb)
        y.append(0 if video.lower().startswith("normal") else 1)
        latencies.append(dt)

    avg_power = sampler.stop() if sampler else None
    return (np.array(X), np.array(y),
            float(np.mean(latencies)) if latencies else 0.0, avg_power)

# ============================================================
# CLASSIFIER HEADS (from reference)
# ============================================================

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
    def __init__(self, in_dim, r=8):
        super().__init__()
        self.base = nn.Linear(in_dim, 1)
        self.A = nn.Linear(in_dim, r, bias=False)
        self.B = nn.Linear(r, 1, bias=False)

    def forward(self, x):
        return (self.base(x) + self.B(self.A(x))).squeeze(-1)


class TemporalAdapter(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.conv = nn.Conv1d(d, 256, 3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(256 * 3, 1)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, 8, 1)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv(x))
        mean = x.mean(2)
        maxp = x.max(2).values
        std = x.std(2)
        z = torch.cat([mean, maxp, std], 1)
        return self.fc(z).squeeze(-1)


def train_nn(net, X_train, y_train, X_test, y_test):
    net.to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    crit = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)),
        batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)), batch_size=32)

    for _ in range(NN_EPOCHS):
        net.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = crit(net(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

    net.eval()
    probs, preds, gts = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            p = torch.sigmoid(net(xb.to(DEVICE)))
            probs += p.cpu().tolist()
            preds += (p > 0.5).float().cpu().tolist()
            gts += yb.tolist()
    return (accuracy_score(gts, preds), f1_score(gts, preds, zero_division=0),
            roc_auc_score(gts, probs))


def train_all_heads(X_train, y_train, X_test, y_test):
    res = {}
    for name, clf in {
        "LogReg": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=SEED),
        "SVM": SVC(probability=True, random_state=SEED),
    }.items():
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)
        res[name] = (accuracy_score(y_test, preds),
                     f1_score(y_test, preds, zero_division=0),
                     roc_auc_score(y_test, probs))
    d = X_train.shape[1]
    res["SimpleAdapter"] = train_nn(SimpleAdapter(d), X_train, y_train, X_test, y_test)
    res["LoRA"] = train_nn(LoRALinear(d), X_train, y_train, X_test, y_test)
    res["TemporalAdapter"] = train_nn(TemporalAdapter(d), X_train, y_train, X_test, y_test)
    return res

# ============================================================
# RUN ONE FRAME COUNT
# ============================================================

all_head_rows = []   # detailed per-head metrics across frame counts


def run_frame_count(k):
    print("\n" + "=" * 70)
    print(f"FRAME COUNT = {k}")
    print("=" * 70)

    Xtr, ytr, _, _ = build_dataset(TRAIN_JSON, k, measure_cost=False, cap=MAX_TRAIN)
    Xte, yte, lat, pw = build_dataset(TEST_JSON, k, measure_cost=True, cap=MAX_TEST)
    print(f"[k={k}] train {Xtr.shape} test {Xte.shape} "
          f"latency={lat:.3f}s/video power={pw}")

    if Xtr.size == 0 or Xte.size == 0:
        print(f"[k={k}] no embeddings - skipping")
        return None

    heads = train_all_heads(Xtr, ytr, Xte, yte)
    for hname, (acc, f1, auc) in heads.items():
        all_head_rows.append({"frames": k, "head": hname,
                              "accuracy": round(acc, 4), "f1": round(f1, 4),
                              "auc": round(auc, 4)})

    if pw is None:
        pw = modeled_power(k)
        power_src = "modeled"
    else:
        power_src = "nvml"

    acc, f1, auc = heads[PRIMARY_MODEL]
    row = {
        "frames": k,
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "latency": round(lat, 4),
        "power": round(pw, 2),
        "energy": round(pw * lat, 2),
        "power_src": power_src,
    }
    print(f"[k={k}] PRIMARY({PRIMARY_MODEL}) {row}")
    gc.collect(); torch.cuda.empty_cache()
    return row

# ============================================================
# MULTI-OBJECTIVE OPTIMIZATION
# ============================================================

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

# ============================================================
# RUN SWEEP + OPTIMIZE
# ============================================================

rows = []
for k in FRAME_COUNTS:
    try:
        r = run_frame_count(k)
        if r:
            rows.append(r)
    except Exception as e:
        print(f"!! frames={k} failed: {e}")
        import traceback; traceback.print_exc()

if not rows:
    raise SystemExit("No frame count completed - nothing to optimize.")

df = pd.DataFrame(rows)
heads_df = pd.DataFrame(all_head_rows)

out_csv = os.path.join(OUT_ROOT, "frame_optimization_classifier.csv")
heads_csv = os.path.join(OUT_ROOT, "frame_optimization_classifier_allheads.csv")

# Merge this run's frame counts with previously saved results so optimization
# runs over the full set. Newly computed frames overwrite stale duplicates.
if MERGE_WITH_EXISTING:
    base_cols = ["frames", "accuracy", "f1", "auc",
                 "latency", "power", "energy", "power_src"]
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)[ [c for c in base_cols if c in
                                      pd.read_csv(out_csv).columns] ]
        df = pd.concat([prev, df[base_cols]], ignore_index=True)
        df = df.drop_duplicates(subset="frames", keep="last")
        print(f"Merged with existing -> frame counts now: "
              f"{sorted(df['frames'].astype(int).tolist())}")
    if os.path.exists(heads_csv) and not heads_df.empty:
        prev_h = pd.read_csv(heads_csv)
        heads_df = pd.concat([prev_h, heads_df], ignore_index=True)
        heads_df = heads_df.drop_duplicates(subset=["frames", "head"],
                                            keep="last")

# Keep only the objective columns + frames for optimization
df = df.sort_values("frames").reset_index(drop=True)
df["pareto"] = pareto_mask(df)
df["topsis"] = topsis(df).round(4)
df["desirability"] = desirability(df).round(4)

best_topsis = int(df.sort_values("topsis", ascending=False).iloc[0]["frames"])
best_desir = int(df.sort_values("desirability", ascending=False).iloc[0]["frames"])
best_knee = knee_point(df, cost="energy", gain="accuracy")

print("\n" + "=" * 90)
print(f"FRAME OPTIMIZATION - BINARY CLASSIFIER (primary head = {PRIMARY_MODEL})")
print("=" * 90)
print(df.to_string(index=False))

# df/heads_df already include any merged prior results (see merge block above)
df.to_csv(out_csv, index=False)
heads_df.to_csv(heads_csv, index=False)
print(f"\nSaved -> {out_csv}")
print(f"Saved -> {heads_csv}")

print("\nALL-HEAD ACCURACY BY FRAME COUNT")
print(heads_df.pivot(index="frames", columns="head", values="accuracy")
      .to_string())

print("\nOPTIMUM FRAME COUNT BY ALGORITHM")
print(f"  Pareto-optimal set : {sorted(df[df['pareto']]['frames'].tolist())}")
print(f"  TOPSIS             : {best_topsis}")
print(f"  Desirability       : {best_desir}")
print(f"  Knee (Acc/J)       : {best_knee}")

from collections import Counter
consensus = Counter([best_topsis, best_desir, best_knee]).most_common(1)[0][0]
print(f"\nRECOMMENDED FRAME COUNT (consensus): {consensus}")

# ============================================================
# PLOT
# ============================================================

import matplotlib.pyplot as plt

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
ax.set_ylabel("Accuracy", fontweight="bold")
ax.set_title("Pareto: Accuracy vs Energy", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(df["latency"], df["accuracy"], "o-", color="#1f77b4")
for _, r in df.iterrows():
    ax.annotate(f"{int(r['frames'])}f", (r["latency"], r["accuracy"]),
                textcoords="offset points", xytext=(6, 6))
ax.set_xlabel("Latency / video (s)", fontweight="bold")
ax.set_ylabel("Accuracy", fontweight="bold")
ax.set_title("Accuracy vs Latency", fontweight="bold")
ax.grid(alpha=0.3)

ax = axes[0, 2]
w = 0.35; x = np.arange(len(df))
ax.bar(x - w / 2, df["topsis"], w, label="TOPSIS")
ax.bar(x + w / 2, df["desirability"], w, label="Desirability")
ax.set_xticks(x); ax.set_xticklabels([f"{int(f)}f" for f in df["frames"]])
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
for m in ["accuracy", "f1", "auc"]:
    ax.plot(df["frames"], df[m], "o-", label=m)
ax.set_xlabel("Frames", fontweight="bold")
ax.set_title(f"{PRIMARY_MODEL} quality vs Frames", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1, 2]
d = df.sort_values("energy")
ax.plot(d["energy"], d["accuracy"], "o-", color="#9467bd")
krow = df[df["frames"] == best_knee].iloc[0]
ax.scatter([krow["energy"]], [krow["accuracy"]], s=260, facecolors="none",
           edgecolors="red", linewidths=2.5, label=f"Knee = {best_knee}f")
ax.set_xlabel("Energy / video (J)", fontweight="bold")
ax.set_ylabel("Accuracy", fontweight="bold")
ax.set_title("Knee-point (best trade-off)", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)

fig.suptitle(f"Frame-Count Optimization - Binary Classifier ({PRIMARY_MODEL})  "
             f"- recommended: {consensus} frames",
             fontsize=15, fontweight="bold")
fig.tight_layout()
out_plot = os.path.join(OUT_ROOT, "frame_optimization_classifier.png")
fig.savefig(out_plot, dpi=160, bbox_inches="tight")
print(f"Saved -> {out_plot}")
