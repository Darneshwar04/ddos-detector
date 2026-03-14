"""
DDoS Attack Detector — BiLSTM Training Script (PyTorch)
========================================================
Training data  : archive (1)  CIC-IDS-2018  (10 CSV files ~6 GB)
                 archive (2)  CIC-DDoS-2019 (18 CSV files ~25 GB)
Test data      : archive (4)/csv/data.csv   (see testing/test_bilstm.py)

Model          : Bidirectional LSTM (64 → 32 units) + Dense head
Task           : Binary classification  — Benign (0)  vs  Attack (1)

Usage:
    python train_bilstm.py

Outputs saved to ./saved_model/
    bilstm_ddos.pt         — trained model weights (PyTorch state_dict)
    scaler.pkl             — fitted StandardScaler
    label_encoder.pkl      — metadata dict (feature cols, class names)
    training_history.png   — loss / accuracy curves
"""

import os
import glob
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# local utilities
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    FEATURE_COLS, normalize_columns, get_label,
    clean_features, create_sequences
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — tweak these to suit your machine's RAM
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ARCHIVE1_DIR = os.path.join(BASE_DIR, "archive (1)")
ARCHIVE2_DIR = os.path.join(BASE_DIR, "archive (2)")
MODEL_DIR    = os.path.join(BASE_DIR, "saved_model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Rows sampled per CSV file during loading (None = load full file)
# With 28 files × 150 000 = ~4.2 M rows, which trains comfortably on 16 GB RAM.
# Increase to 500_000 if you have more RAM / GPU VRAM.
ROWS_PER_FILE   = 150_000

SEQUENCE_LEN    = 10        # timesteps per BiLSTM input
BATCH_SIZE      = 512
EPOCHS          = 30
LEARNING_RATE   = 1e-3
VAL_SPLIT       = 0.15      # fraction held out for validation
SEED            = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def gather_csv_paths() -> list[str]:
    """Collect all training CSV files from archives 1 and 2."""
    paths = (
        glob.glob(os.path.join(ARCHIVE1_DIR, "*.csv")) +
        glob.glob(os.path.join(ARCHIVE2_DIR, "**", "*.csv"), recursive=True)
    )
    paths = sorted(paths)
    print(f"[*] Found {len(paths)} training CSV files")
    for p in paths:
        size_mb = os.path.getsize(p) / 1e6
        print(f"    {size_mb:7.1f} MB  {os.path.relpath(p, BASE_DIR)}")
    return paths


def load_single_csv(path: str, nrows: int | None) -> pd.DataFrame | None:
    """Load one CSV, normalize columns, clean features, extract label."""
    try:
        df = pd.read_csv(
            path, nrows=nrows,
            encoding="utf-8", on_bad_lines="skip",
            low_memory=False
        )
        df = normalize_columns(df)

        # ensure all feature columns exist (fill missing ones with 0)
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

        df = clean_features(df)
        df["binary_label"] = get_label(df)
        return df[FEATURE_COLS + ["binary_label"]]
    except Exception as exc:
        print(f"    [!] Skipped {os.path.basename(path)}: {exc}")
        return None


def build_dataset(paths: list[str], rows_per_file: int | None):
    """Stream through all CSVs, returning (X_all, y_all) numpy arrays."""
    X_parts, y_parts = [], []
    total_benign = total_attack = 0

    for i, path in enumerate(paths, 1):
        fname = os.path.relpath(path, BASE_DIR)
        print(f"  [{i:2d}/{len(paths)}] Loading {fname} …", end=" ", flush=True)
        df = load_single_csv(path, nrows=rows_per_file)
        if df is None:
            continue

        y = df["binary_label"].values.astype(np.int8)
        X = df[FEATURE_COLS].values.astype(np.float32)
        X_parts.append(X)
        y_parts.append(y)

        b = (y == 0).sum()
        a = (y == 1).sum()
        total_benign += b
        total_attack += a
        print(f"rows={len(df):,}  benign={b:,}  attack={a:,}")

    print(f"\n[*] Total — benign: {total_benign:,}  attack: {total_attack:,}")
    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    return X_all, y_all


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class BiLSTMDetector(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.bilstm1 = nn.LSTM(input_size=n_features, hidden_size=64, num_layers=1,
                               batch_first=True, bidirectional=True)
        self.drop1   = nn.Dropout(0.2)
        self.bilstm2 = nn.LSTM(input_size=128, hidden_size=32, num_layers=1,
                               batch_first=True, bidirectional=True)
        self.drop2   = nn.Dropout(0.2)
        self.dense   = nn.Linear(64, 64)
        self.relu    = nn.ReLU()
        self.drop3   = nn.Dropout(0.3)
        self.output  = nn.Linear(64, 2)

    def forward(self, x):
        out, _ = self.bilstm1(x)
        out     = self.drop1(out)
        out, _ = self.bilstm2(out)
        out     = self.drop2(out[:, -1, :])
        out     = self.drop3(self.relu(self.dense(out)))
        return self.output(out)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(history: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train loss")
    axes[0].plot(history["val_loss"],   label="val loss")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(history["train_acc"], label="train acc")
    axes[1].plot(history["val_acc"],   label="val acc")
    axes[1].set_title("Accuracy"); axes[1].legend()

    plt.tight_layout()
    fig.savefig(out_path)
    print(f"[*] Training curve saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  BiLSTM DDoS Detector — Training")
    print("=" * 65)

    # ── collect files ────────────────────────────────────────────────────────
    paths = gather_csv_paths()
    if not paths:
        raise RuntimeError("No training CSV files found — check ARCHIVE1_DIR / ARCHIVE2_DIR")

    # ── load data ────────────────────────────────────────────────────────────
    print("\n[*] Loading data …")
    X_raw, y_raw = build_dataset(paths, rows_per_file=ROWS_PER_FILE)
    print(f"\n[*] Raw dataset shape: X={X_raw.shape}, y={y_raw.shape}")

    # ── shuffle before splitting ─────────────────────────────────────────────
    idx = np.random.permutation(len(X_raw))
    X_raw, y_raw = X_raw[idx], y_raw[idx]

    # ── scale ─────────────────────────────────────────────────────────────────
    print("[*] Fitting StandardScaler …")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)
    del X_raw   # free memory

    # ── create sequences ──────────────────────────────────────────────────────
    print(f"[*] Creating sequences (len={SEQUENCE_LEN}) …")
    X_seq, y_seq = create_sequences(X_scaled, y_raw, SEQUENCE_LEN)
    del X_scaled, y_raw
    print(f"    Sequences: {X_seq.shape}, labels: {y_seq.shape}")

    # ── train / val split ─────────────────────────────────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_seq, y_seq, test_size=VAL_SPLIT, random_state=SEED, stratify=y_seq
    )
    print(f"[*] Train: {X_tr.shape}  Val: {X_val.shape}")
    del X_seq, y_seq

    # ── class weights (handles imbalance) ─────────────────────────────────────
    classes = np.unique(y_tr)
    cw = compute_class_weight("balanced", classes=classes, y=y_tr)
    class_weight_tensor = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
    print(f"[*] Class weights: { {int(c): round(float(w),4) for c,w in zip(classes,cw)} }")

    # ── build model ───────────────────────────────────────────────────────────
    n_features = X_tr.shape[2]
    model = BiLSTMDetector(n_features).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Model on {DEVICE}  |  parameters: {total_params:,}")

    # ── dataloaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.int64))),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val.astype(np.int64))),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6, verbose=True
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    model_path = os.path.join(MODEL_DIR, "bilstm_ddos.pt")

    # ── train ─────────────────────────────────────────────────────────────────
    print(f"\n[*] Training for up to {EPOCHS} epochs …")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss    += loss.item() * len(yb)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total   += len(yb)

        model.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits  = model(Xb)
                loss    = criterion(logits, yb)
                vl_loss    += loss.item() * len(yb)
                vl_correct += (logits.argmax(1) == yb).sum().item()
                vl_total   += len(yb)

        tr_loss /= tr_total;  vl_loss /= vl_total
        tr_acc   = tr_correct / tr_total
        vl_acc   = vl_correct / vl_total
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        scheduler.step(vl_loss)

        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={tr_loss:.4f}  acc={tr_acc*100:.2f}%  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc*100:.2f}%")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"    ✓ checkpoint saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # ── save artefacts ────────────────────────────────────────────────────────
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(
        {"feature_cols": FEATURE_COLS, "class_names": ["Benign", "Attack"],
         "seq_len": SEQUENCE_LEN, "n_features": n_features},
        os.path.join(MODEL_DIR, "label_encoder.pkl")
    )
    plot_history(history, os.path.join(MODEL_DIR, "training_history.png"))

    best_acc = max(history["val_acc"]) * 100
    print(f"\n[✓] Best val_accuracy : {best_acc:.2f}%")
    print(f"[✓] Model saved       → {model_path}")
    print(f"[✓] Scaler saved      → {os.path.join(MODEL_DIR, 'scaler.pkl')}")
    print("\nNext step: python testing/test_bilstm.py")


if __name__ == "__main__":
    main()
