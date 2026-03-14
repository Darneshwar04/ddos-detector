"""
Shared preprocessing utilities for the BiLSTM DDoS detector.
Used by both train_bilstm.py and testing/test_bilstm.py
"""

import numpy as np
import pandas as pd

# ── 64 canonical feature names used by the model ─────────────────────────────
FEATURE_COLS = [
    "flow_duration",
    "tot_fwd_pkts", "tot_bwd_pkts",
    "totlen_fwd_pkts", "totlen_bwd_pkts",
    "fwd_pkt_len_max", "fwd_pkt_len_min", "fwd_pkt_len_mean", "fwd_pkt_len_std",
    "bwd_pkt_len_max", "bwd_pkt_len_min", "bwd_pkt_len_mean", "bwd_pkt_len_std",
    "flow_byts_s", "flow_pkts_s",
    "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min",
    "fwd_iat_tot", "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max", "fwd_iat_min",
    "bwd_iat_tot", "bwd_iat_mean", "bwd_iat_std", "bwd_iat_max", "bwd_iat_min",
    "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags",
    "fwd_header_len", "bwd_header_len",
    "fwd_pkts_s", "bwd_pkts_s",
    "pkt_len_min", "pkt_len_max", "pkt_len_mean", "pkt_len_std", "pkt_len_var",
    "fin_flag_cnt", "syn_flag_cnt", "rst_flag_cnt", "psh_flag_cnt",
    "ack_flag_cnt", "urg_flag_cnt", "cwe_flag_cnt", "ece_flag_cnt",
    "down_up_ratio", "pkt_size_avg",
    "fwd_seg_size_avg", "bwd_seg_size_avg",
    "init_fwd_win_byts", "init_bwd_win_byts",
    "fwd_act_data_pkts", "fwd_seg_size_min",
    "active_mean", "active_std", "active_max", "active_min",
    "idle_mean", "idle_std", "idle_max", "idle_min",
]

# ── column renaming maps for each archive format ──────────────────────────────

# archive (1)  — CIC-IDS-2018  (already clean names, no leading spaces)
ARCHIVE1_COL_MAP = {
    "Flow Duration"    : "flow_duration",
    "Tot Fwd Pkts"     : "tot_fwd_pkts",
    "Tot Bwd Pkts"     : "tot_bwd_pkts",
    "TotLen Fwd Pkts"  : "totlen_fwd_pkts",
    "TotLen Bwd Pkts"  : "totlen_bwd_pkts",
    "Fwd Pkt Len Max"  : "fwd_pkt_len_max",
    "Fwd Pkt Len Min"  : "fwd_pkt_len_min",
    "Fwd Pkt Len Mean" : "fwd_pkt_len_mean",
    "Fwd Pkt Len Std"  : "fwd_pkt_len_std",
    "Bwd Pkt Len Max"  : "bwd_pkt_len_max",
    "Bwd Pkt Len Min"  : "bwd_pkt_len_min",
    "Bwd Pkt Len Mean" : "bwd_pkt_len_mean",
    "Bwd Pkt Len Std"  : "bwd_pkt_len_std",
    "Flow Byts/s"      : "flow_byts_s",
    "Flow Pkts/s"      : "flow_pkts_s",
    "Flow IAT Mean"    : "flow_iat_mean",
    "Flow IAT Std"     : "flow_iat_std",
    "Flow IAT Max"     : "flow_iat_max",
    "Flow IAT Min"     : "flow_iat_min",
    "Fwd IAT Tot"      : "fwd_iat_tot",
    "Fwd IAT Mean"     : "fwd_iat_mean",
    "Fwd IAT Std"      : "fwd_iat_std",
    "Fwd IAT Max"      : "fwd_iat_max",
    "Fwd IAT Min"      : "fwd_iat_min",
    "Bwd IAT Tot"      : "bwd_iat_tot",
    "Bwd IAT Mean"     : "bwd_iat_mean",
    "Bwd IAT Std"      : "bwd_iat_std",
    "Bwd IAT Max"      : "bwd_iat_max",
    "Bwd IAT Min"      : "bwd_iat_min",
    "Fwd PSH Flags"    : "fwd_psh_flags",
    "Bwd PSH Flags"    : "bwd_psh_flags",
    "Fwd URG Flags"    : "fwd_urg_flags",
    "Bwd URG Flags"    : "bwd_urg_flags",
    "Fwd Header Len"   : "fwd_header_len",
    "Bwd Header Len"   : "bwd_header_len",
    "Fwd Pkts/s"       : "fwd_pkts_s",
    "Bwd Pkts/s"       : "bwd_pkts_s",
    "Pkt Len Min"      : "pkt_len_min",
    "Pkt Len Max"      : "pkt_len_max",
    "Pkt Len Mean"     : "pkt_len_mean",
    "Pkt Len Std"      : "pkt_len_std",
    "Pkt Len Var"      : "pkt_len_var",
    "FIN Flag Cnt"     : "fin_flag_cnt",
    "SYN Flag Cnt"     : "syn_flag_cnt",
    "RST Flag Cnt"     : "rst_flag_cnt",
    "PSH Flag Cnt"     : "psh_flag_cnt",
    "ACK Flag Cnt"     : "ack_flag_cnt",
    "URG Flag Cnt"     : "urg_flag_cnt",
    "CWE Flag Count"   : "cwe_flag_cnt",
    "ECE Flag Cnt"     : "ece_flag_cnt",
    "Down/Up Ratio"    : "down_up_ratio",
    "Pkt Size Avg"     : "pkt_size_avg",
    "Fwd Seg Size Avg" : "fwd_seg_size_avg",
    "Bwd Seg Size Avg" : "bwd_seg_size_avg",
    "Init Fwd Win Byts": "init_fwd_win_byts",
    "Init Bwd Win Byts": "init_bwd_win_byts",
    "Fwd Act Data Pkts": "fwd_act_data_pkts",
    "Fwd Seg Size Min" : "fwd_seg_size_min",
    "Active Mean"      : "active_mean",
    "Active Std"       : "active_std",
    "Active Max"       : "active_max",
    "Active Min"       : "active_min",
    "Idle Mean"        : "idle_mean",
    "Idle Std"         : "idle_std",
    "Idle Max"         : "idle_max",
    "Idle Min"         : "idle_min",
}

# archive (2)  — CIC-DDoS-2019  (leading spaces + verbose names)
ARCHIVE2_COL_MAP = {
    " Flow Duration"              : "flow_duration",
    " Total Fwd Packets"          : "tot_fwd_pkts",
    " Total Backward Packets"     : "tot_bwd_pkts",
    "Total Length of Fwd Packets" : "totlen_fwd_pkts",
    " Total Length of Bwd Packets": "totlen_bwd_pkts",
    " Fwd Packet Length Max"      : "fwd_pkt_len_max",
    " Fwd Packet Length Min"      : "fwd_pkt_len_min",
    " Fwd Packet Length Mean"     : "fwd_pkt_len_mean",
    " Fwd Packet Length Std"      : "fwd_pkt_len_std",
    "Bwd Packet Length Max"       : "bwd_pkt_len_max",
    " Bwd Packet Length Min"      : "bwd_pkt_len_min",
    " Bwd Packet Length Mean"     : "bwd_pkt_len_mean",
    " Bwd Packet Length Std"      : "bwd_pkt_len_std",
    "Flow Bytes/s"                : "flow_byts_s",
    " Flow Packets/s"             : "flow_pkts_s",
    " Flow IAT Mean"              : "flow_iat_mean",
    " Flow IAT Std"               : "flow_iat_std",
    " Flow IAT Max"               : "flow_iat_max",
    " Flow IAT Min"               : "flow_iat_min",
    "Fwd IAT Total"               : "fwd_iat_tot",
    " Fwd IAT Mean"               : "fwd_iat_mean",
    " Fwd IAT Std"                : "fwd_iat_std",
    " Fwd IAT Max"                : "fwd_iat_max",
    " Fwd IAT Min"                : "fwd_iat_min",
    "Bwd IAT Total"               : "bwd_iat_tot",
    " Bwd IAT Mean"               : "bwd_iat_mean",
    " Bwd IAT Std"                : "bwd_iat_std",
    " Bwd IAT Max"                : "bwd_iat_max",
    " Bwd IAT Min"                : "bwd_iat_min",
    "Fwd PSH Flags"               : "fwd_psh_flags",
    " Bwd PSH Flags"              : "bwd_psh_flags",
    " Fwd URG Flags"              : "fwd_urg_flags",
    " Bwd URG Flags"              : "bwd_urg_flags",
    " Fwd Header Length"          : "fwd_header_len",
    " Bwd Header Length"          : "bwd_header_len",
    "Fwd Packets/s"               : "fwd_pkts_s",
    " Bwd Packets/s"              : "bwd_pkts_s",
    " Min Packet Length"          : "pkt_len_min",
    " Max Packet Length"          : "pkt_len_max",
    " Packet Length Mean"         : "pkt_len_mean",
    " Packet Length Std"          : "pkt_len_std",
    " Packet Length Variance"     : "pkt_len_var",
    "FIN Flag Count"              : "fin_flag_cnt",
    " SYN Flag Count"             : "syn_flag_cnt",
    " RST Flag Count"             : "rst_flag_cnt",
    " PSH Flag Count"             : "psh_flag_cnt",
    " ACK Flag Count"             : "ack_flag_cnt",
    " URG Flag Count"             : "urg_flag_cnt",
    " CWE Flag Count"             : "cwe_flag_cnt",
    " ECE Flag Count"             : "ece_flag_cnt",
    " Down/Up Ratio"              : "down_up_ratio",
    " Average Packet Size"        : "pkt_size_avg",
    " Avg Fwd Segment Size"       : "fwd_seg_size_avg",
    " Avg Bwd Segment Size"       : "bwd_seg_size_avg",
    "Init_Win_bytes_forward"      : "init_fwd_win_byts",
    " Init_Win_bytes_backward"    : "init_bwd_win_byts",
    " act_data_pkt_fwd"           : "fwd_act_data_pkts",
    " min_seg_size_forward"       : "fwd_seg_size_min",
    "Active Mean"                 : "active_mean",
    " Active Std"                 : "active_std",
    " Active Max"                 : "active_max",
    " Active Min"                 : "active_min",
    "Idle Mean"                   : "idle_mean",
    " Idle Std"                   : "idle_std",
    " Idle Max"                   : "idle_max",
    " Idle Min"                   : "idle_min",
}

# archive (4) — has clean names but slightly different spellings
ARCHIVE4_COL_MAP = {
    "Flow Duration"             : "flow_duration",
    "Total Fwd Packet"          : "tot_fwd_pkts",
    "Total Bwd packets"         : "tot_bwd_pkts",
    "Total Length of Fwd Packet": "totlen_fwd_pkts",
    "Total Length of Bwd Packet": "totlen_bwd_pkts",
    "Fwd Packet Length Max"     : "fwd_pkt_len_max",
    "Fwd Packet Length Min"     : "fwd_pkt_len_min",
    "Fwd Packet Length Mean"    : "fwd_pkt_len_mean",
    "Fwd Packet Length Std"     : "fwd_pkt_len_std",
    "Bwd Packet Length Max"     : "bwd_pkt_len_max",
    "Bwd Packet Length Min"     : "bwd_pkt_len_min",
    "Bwd Packet Length Mean"    : "bwd_pkt_len_mean",
    "Bwd Packet Length Std"     : "bwd_pkt_len_std",
    "Flow Bytes/s"              : "flow_byts_s",
    "Flow Packets/s"            : "flow_pkts_s",
    "Flow IAT Mean"             : "flow_iat_mean",
    "Flow IAT Std"              : "flow_iat_std",
    "Flow IAT Max"              : "flow_iat_max",
    "Flow IAT Min"              : "flow_iat_min",
    "Fwd IAT Total"             : "fwd_iat_tot",
    "Fwd IAT Mean"              : "fwd_iat_mean",
    "Fwd IAT Std"               : "fwd_iat_std",
    "Fwd IAT Max"               : "fwd_iat_max",
    "Fwd IAT Min"               : "fwd_iat_min",
    "Bwd IAT Total"             : "bwd_iat_tot",
    "Bwd IAT Mean"              : "bwd_iat_mean",
    "Bwd IAT Std"               : "bwd_iat_std",
    "Bwd IAT Max"               : "bwd_iat_max",
    "Bwd IAT Min"               : "bwd_iat_min",
    "Fwd PSH Flags"             : "fwd_psh_flags",
    "Bwd PSH Flags"             : "bwd_psh_flags",
    "Fwd URG Flags"             : "fwd_urg_flags",
    "Bwd URG Flags"             : "bwd_urg_flags",
    "Fwd Header Length"         : "fwd_header_len",
    "Bwd Header Length"         : "bwd_header_len",
    "Fwd Packets/s"             : "fwd_pkts_s",
    "Bwd Packets/s"             : "bwd_pkts_s",
    "Packet Length Min"         : "pkt_len_min",
    "Packet Length Max"         : "pkt_len_max",
    "Packet Length Mean"        : "pkt_len_mean",
    "Packet Length Std"         : "pkt_len_std",
    "Packet Length Variance"    : "pkt_len_var",
    "FIN Flag Count"            : "fin_flag_cnt",
    "SYN Flag Count"            : "syn_flag_cnt",
    "RST Flag Count"            : "rst_flag_cnt",
    "PSH Flag Count"            : "psh_flag_cnt",
    "ACK Flag Count"            : "ack_flag_cnt",
    "URG Flag Count"            : "urg_flag_cnt",
    "CWR Flag Count"            : "cwe_flag_cnt",
    "ECE Flag Count"            : "ece_flag_cnt",
    "Down/Up Ratio"             : "down_up_ratio",
    "Average Packet Size"       : "pkt_size_avg",
    "Fwd Segment Size Avg"      : "fwd_seg_size_avg",
    "Bwd Segment Size Avg"      : "bwd_seg_size_avg",
    "FWD Init Win Bytes"        : "init_fwd_win_byts",
    "Bwd Init Win Bytes"        : "init_bwd_win_byts",
    "Fwd Act Data Pkts"         : "fwd_act_data_pkts",
    "Fwd Seg Size Min"          : "fwd_seg_size_min",
    "Active Mean"               : "active_mean",
    "Active Std"                : "active_std",
    "Active Max"                : "active_max",
    "Active Min"                : "active_min",
    "Idle Mean"                 : "idle_mean",
    "Idle Std"                  : "idle_std",
    "Idle Max"                  : "idle_max",
    "Idle Min"                  : "idle_min",
}


def detect_format(columns: list[str]) -> str:
    """Identify which archive format a file belongs to."""
    cols_stripped = [c.strip() for c in columns]
    if "Tot Fwd Pkts" in cols_stripped:
        return "archive1"
    if "Total Fwd Packets" in cols_stripped or "Total Fwd Packets" in columns:
        return "archive2"
    if "Total Fwd Packet" in cols_stripped:
        return "archive4"
    return "unknown"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from all column names and apply the correct rename map."""
    fmt = detect_format(list(df.columns))
    if fmt == "archive1":
        return df.rename(columns=ARCHIVE1_COL_MAP)
    if fmt == "archive2":
        return df.rename(columns=ARCHIVE2_COL_MAP)
    if fmt == "archive4":
        return df.rename(columns=ARCHIVE4_COL_MAP)
    # fallback: try stripping whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    return df


def get_label(df: pd.DataFrame) -> pd.Series:
    """Return binary label series: 0=benign, 1=attack."""
    label_col = None
    for c in df.columns:
        if c.strip().lower() == "label":
            label_col = c
            break
    if label_col is None:
        raise ValueError("No 'Label' column found in DataFrame")
    raw = df[label_col].astype(str).str.strip().str.lower()
    return (raw != "benign").astype(np.int8)


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/NaN with 0 and ensure float32."""
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df[FEATURE_COLS] = (
        df[FEATURE_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )
    return df


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """
    Reshape flat arrays into overlapping windows of length seq_len.
    Uses non-overlapping windows for speed; label = last step's label.
    """
    n = (len(X) // seq_len) * seq_len
    X_seq = X[:n].reshape(-1, seq_len, X.shape[1])
    y_seq = y[:n].reshape(-1, seq_len)[:, -1]
    return X_seq, y_seq
