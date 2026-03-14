"""
DDoS Attack Detector — FastAPI Web Server
==========================================
This file is the DEPLOYMENT component only.
Training is done separately in: train_bilstm.ipynb

Workflow
--------
  1. Train model  →  open train_bilstm.ipynb and run all cells
  2. Start server →  python -m uvicorn app:app --host 0.0.0.0 --port 8000
  3. Open browser →  http://localhost:8000

Requires (in saved_model/):
  bilstm_ddos.pt      — trained model weights (PyTorch state_dict)
  scaler.pkl          — fitted StandardScaler
  label_encoder.pkl   — feature list + metadata

Features
--------
- Upload any CICFlowMeter CSV (archive-1/2/4 format auto-detected)
- Bi-LSTM inference with real-time progress bar
- Full metrics (accuracy, F1, ROC-AUC) when file has labels
- Confusion-matrix image rendered in browser
- Per-class classification report table
- Threat-level badge (Low / Medium / High)
"""

import os, sys, io, base64, traceback
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import FEATURE_COLS, normalize_columns, get_label, clean_features

import torch
import torch.nn as nn
import joblib


class _BiLSTMDetector(nn.Module):
    """Mirror of the training architecture — must match train_bilstm.ipynb."""
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


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "saved_model")
TMPL_DIR    = os.path.join(BASE_DIR, "templates")

# ── global model state ─────────────────────────────────────────────────────────
_model  = None
_scaler = None
_meta: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global _model, _scaler, _meta
    model_path  = os.path.join(MODEL_DIR, "bilstm_ddos.pt")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    meta_path   = os.path.join(MODEL_DIR, "label_encoder.pkl")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print("[*] Loading BiLSTM model ...")
        _scaler = joblib.load(scaler_path)
        _meta   = joblib.load(meta_path) if os.path.exists(meta_path) else {}
        n_feat  = _meta.get("n_features", len(FEATURE_COLS))
        net     = _BiLSTMDetector(n_feat).to(_DEVICE)
        net.load_state_dict(torch.load(model_path, map_location=_DEVICE))
        net.eval()
        _model  = net
        total_p = sum(p.numel() for p in net.parameters())
        print(f"[OK] Model ready on {_DEVICE} - parameters: {total_p:,}")
    else:
        print("[!] Trained model not found.")
        print("    Run notebook:  jupyter lab train_bilstm.ipynb")
    
    yield
    
    # Shutdown
    print("[*] Shutting down...")


app = FastAPI(title="DDoS Attack Detector", version="1.0.0", lifespan=lifespan)
templates = Jinja2Templates(directory=TMPL_DIR)


# ── routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def status():
    trained = _model is not None
    params  = sum(p.numel() for p in _model.parameters()) if trained else 0
    return {
        "model_loaded"    : trained,
        "torch_version"   : torch.__version__,
        "device"          : str(_DEVICE),
        "python_version"  : sys.version.split()[0],
        "parameters"      : params,
        "seq_len"         : _meta.get("seq_len", 10),
        "n_features"      : len(FEATURE_COLS),
        "class_names"     : _meta.get("class_names", ["Benign", "Attack"]),
        "model_path"      : os.path.join(MODEL_DIR, "bilstm_ddos.pt"),
    }


@app.post("/api/predict")
async def predict(
    file    : UploadFile = File(...),
    max_rows: int        = Form(100_000),
):
    # ── guards ────────────────────────────────────────────────────────────────

    if _model is None:
        raise HTTPException(
            503,
            "Model not loaded. Train it first:\n  python train_bilstm.py"
        )
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported.")

    # ── parse CSV ─────────────────────────────────────────────────────────────
    raw_bytes = await file.read()
    try:
        df_raw = pd.read_csv(
            io.BytesIO(raw_bytes),
            nrows=max(1, int(max_rows)),
            low_memory=False,
            on_bad_lines="skip",
        )
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}")

    if df_raw.empty:
        raise HTTPException(400, "CSV is empty or unreadable.")

    try:
        # ── normalise columns & clean ─────────────────────────────────────────
        df = normalize_columns(df_raw.copy())
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        df = clean_features(df)

        # ── detect labels ─────────────────────────────────────────────────────
        has_labels = any(c.strip().lower() == "label" for c in df_raw.columns)

        # ── scale ─────────────────────────────────────────────────────────────
        X = _scaler.transform(df[FEATURE_COLS].values.astype(np.float32))

        seq_len     = _meta.get("seq_len", 10)
        class_names = _meta.get("class_names", ["Benign", "Attack"])

        n = (len(X) // seq_len) * seq_len
        if n == 0:
            raise HTTPException(
                400,
                f"Need at least {seq_len} rows to form one sequence. Got {len(X)}."
            )

        X_seq = X[:n].reshape(-1, seq_len, len(FEATURE_COLS))

        y_seq = None
        if has_labels:
            y_raw = get_label(df).values
            y_seq = y_raw[:n].reshape(-1, seq_len)[:, -1]

        # ── inference ─────────────────────────────────────────────────────────
        all_preds, all_probs = [], []
        batch = 512
        with torch.no_grad():
            for i in range(0, len(X_seq), batch):
                Xb     = torch.from_numpy(X_seq[i:i+batch]).to(_DEVICE)
                logits = _model(Xb)
                probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds_b = logits.argmax(1).cpu().numpy()
                all_preds.extend(preds_b)
                all_probs.extend(probs)
        preds       = np.array(all_preds)
        attack_prob = np.array(all_probs)

        # ── build response ────────────────────────────────────────────────────
        benign_cnt = int((preds == 0).sum())
        attack_cnt = int((preds == 1).sum())

        result: dict = {
            "filename"            : file.filename,
            "total_rows_loaded"   : len(df_raw),
            "sequences_evaluated" : len(preds),
            "predictions": {
                "benign_count"      : benign_cnt,
                "attack_count"      : attack_cnt,
                "attack_percentage" : round(float(attack_cnt / len(preds) * 100), 2),
            },
        }

        if has_labels and y_seq is not None:
            acc    = float(accuracy_score(y_seq, preds))
            f1     = float(f1_score(y_seq, preds, average="weighted", zero_division=0))
            try:
                auc = float(roc_auc_score(y_seq, attack_prob))
            except Exception:
                auc = None

            cm     = confusion_matrix(y_seq, preds)
            rpt    = classification_report(
                y_seq, preds,
                target_names=class_names,
                digits=4, output_dict=True,
                zero_division=0,
            )

            result["metrics"] = {
                "accuracy"               : round(acc * 100, 2),
                "f1_score"               : round(f1, 4),
                "roc_auc"                : round(auc, 4) if auc is not None else None,
                "classification_report"  : rpt,
            }
            result["confusion_matrix"] = {
                "matrix" : cm.tolist(),
                "labels" : class_names,
                "image"  : _cm_to_base64(cm, class_names),
            }

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(500, traceback.format_exc())


# ── helpers ────────────────────────────────────────────────────────────────────

def _cm_to_base64(cm: np.ndarray, labels: list[str]) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
