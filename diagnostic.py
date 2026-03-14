import numpy as np
import pandas as pd
import torch
import joblib
import os
from utils import FEATURE_COLS, normalize_columns, get_label, clean_features
from app import _BiLSTMDetector

# Load model and scaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "saved_model"
model_path  = os.path.join(MODEL_DIR, "bilstm_ddos.pt")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
meta_path   = os.path.join(MODEL_DIR, "label_encoder.pkl")

scaler = joblib.load(scaler_path)
meta   = joblib.load(meta_path) if os.path.exists(meta_path) else {}
n_feat = meta.get("n_features", len(FEATURE_COLS))

model = _BiLSTMDetector(n_feat).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load and prep test data
print("[*] Loading test data...")
df = pd.read_csv(r'archive (4)\csv\data.csv', nrows=1000)

df = normalize_columns(df)
for col in FEATURE_COLS:
    if col not in df.columns:
        df[col] = 0.0
df = clean_features(df)

y_true = get_label(df).values
print(f"True labels: {np.bincount(y_true)}")

# Scale features
X = scaler.transform(df[FEATURE_COLS].values.astype(np.float32))

seq_len = meta.get("seq_len", 10)
n = (len(X) // seq_len) * seq_len
X_seq = X[:n].reshape(-1, seq_len, len(FEATURE_COLS))
y_seq = y_true[:n].reshape(-1, seq_len)[:, -1]

# Inference
all_preds, all_probs, all_logits = [], [], []
with torch.no_grad():
    for i in range(0, len(X_seq), 512):
        Xb = torch.from_numpy(X_seq[i:i+512]).to(device)
        logits = model(Xb)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(1).cpu().numpy()
        
        all_logits.extend(logits.cpu().numpy())
        all_preds.extend(preds)
        all_probs.extend(probs.cpu().numpy())

preds = np.array(all_preds)
probs = np.array(all_probs)
logits = np.array(all_logits)

print(f"\nPredicted labels: {np.bincount(preds)}")
print(f"True labels     : {np.bincount(y_seq)}")
print(f"\nTrue distribution: {np.bincount(y_seq) / len(y_seq) * 100}%")
print(f"Pred distribution: {np.bincount(preds) / len(preds) * 100}%")

# Check logit ranges
print(f"\nLogit ranges:")
print(f"  Class 0 (Benign) logits  : {logits[:, 0].min():.4f} to {logits[:, 0].max():.4f}")
print(f"  Class 1 (Attack) logits  : {logits[:, 1].min():.4f} to {logits[:, 1].max():.4f}")

# Check a few samples
print(f"\nSample predictions (first 10):")
for i in range(min(10, len(preds))):
    print(f"  Index {i}: True={y_seq[i]}, Pred={preds[i]}, Logits=[{logits[i,0]:.3f}, {logits[i,1]:.3f}], Probs=[{probs[i,0]:.3f}, {probs[i,1]:.3f}]")
