import os
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from scipy.io import loadmat
from scipy.signal import butter, filtfilt, stft

try:
    from thop import profile as thop_profile
except Exception:
    thop_profile = None


# =========================
# Reproducibility
# =========================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Config
# =========================
@dataclass
class PreprocessConfig:
    fs: int = 128
    bp_low: float = 0.2
    bp_high: float = 43.0
    bp_order: int = 5

    stft_window_len: int = 256
    stft_overlap: float = 0.5
    stft_nfft: int = 1024
    stft_window: str = "blackman"

    n_freq_bins: int = 36
    smoothing_frames: int = 15
    eps_db: float = 1e-12


@dataclass
class ModelConfig:
    T: int = 8
    input_dim: int = 504

    embed_dim: int = 64
    num_heads: int = 4
    depth: int = 2
    ff_hidden_dim: int = 128

    pos_enc_type: str = "trainable"  
    norm_type: str = "post"          

    head_dropout: float = 0.2

    # Optional evaluated regularization
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    ff_dropout: float = 0.0


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 1e-3
    max_epochs: int = 84
    patience: int = 10

    optimizer_name: str = "adam"  
    weight_decay: float = 0.0      

    loss_name: str = "ce"          
    focal_gamma: float = 2.0

    grad_clip_norm: float = 1.0


# =========================
# Baseline correction
# =========================
def imodpoly_baseline(y: np.ndarray, degree: int = 3, n_iter: int = 20) -> np.ndarray:
    x = np.arange(y.shape[0], dtype=np.float64)
    y_work = y.astype(np.float64).copy()
    for _ in range(n_iter):
        coeff = np.polyfit(x, y_work, degree)
        base = np.polyval(coeff, x)
        y_work = np.minimum(y_work, base)
    coeff = np.polyfit(x, y_work, degree)
    base = np.polyval(coeff, x)
    return base.astype(y.dtype)


# =========================
# Filtering
# =========================
def butter_bandpass_filter(sig: np.ndarray, fs: int, low: float, high: float, order: int) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
    return filtfilt(b, a, sig, axis=0)


# =========================
# Feature extraction, STFT frames to 504
# =========================
def moving_average_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    pad = k // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float64) / float(k)
    out = np.convolve(x_pad, kernel, mode="valid")
    return out.astype(x.dtype)


def extract_stft_frames_504(eeg: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    Returns frame features:
    X_frames shape (n_frames, 504)
    eeg shape (n_samples, 14) or (14, n_samples)
    """
    if eeg.ndim != 2:
        raise ValueError("EEG must be a 2D array")

    if eeg.shape[0] == 14 and eeg.shape[1] != 14:
        eeg = eeg.T

    if eeg.shape[1] != 14:
        raise ValueError("EEG must have 14 channels")

    eeg = eeg.astype(np.float64)

    for ch in range(14):
        base = imodpoly_baseline(eeg[:, ch], degree=3, n_iter=20)
        eeg[:, ch] = eeg[:, ch] - base

    eeg = butter_bandpass_filter(
        eeg,
        fs=cfg.fs,
        low=cfg.bp_low,
        high=cfg.bp_high,
        order=cfg.bp_order,
    )

    noverlap = int(cfg.stft_window_len * cfg.stft_overlap)
    f, _t, Z = stft(
        eeg.T,
        fs=cfg.fs,
        window=cfg.stft_window,
        nperseg=cfg.stft_window_len,
        noverlap=noverlap,
        nfft=cfg.stft_nfft,
        boundary=None,
        padded=False,
    )
    # Z shape (14, n_freq, n_frames)

    target_freqs = np.linspace(cfg.bp_low, cfg.bp_high, cfg.n_freq_bins)
    idx = np.array([int(np.argmin(np.abs(f - tf))) for tf in target_freqs], dtype=int)

    Z_sel = Z[:, idx, :]  # (14, 36, n_frames)
    power = (np.abs(Z_sel) ** 2).astype(np.float64)
    power_db = 10.0 * np.log10(power + cfg.eps_db)

    if cfg.smoothing_frames > 1:
        smoothed = np.empty_like(power_db)
        for ch in range(14):
            for fb in range(cfg.n_freq_bins):
                smoothed[ch, fb, :] = moving_average_1d(power_db[ch, fb, :], cfg.smoothing_frames)
        power_db = smoothed

    n_frames = power_db.shape[2]
    X_frames = power_db.transpose(2, 0, 1).reshape(n_frames, 14 * cfg.n_freq_bins)
    return X_frames.astype(np.float32)


def frames_to_T_tokens(X_frames: np.ndarray, T: int) -> np.ndarray:
    """
    Converts STFT frame features into T temporal tokens by splitting frames into T segments
    and averaging within each segment.
    Returns X_tokens shape (T, 504)
    """
    n_frames = X_frames.shape[0]
    if n_frames <= 0:
        raise ValueError("No STFT frames produced")

    edges = np.linspace(0, n_frames, T + 1)
    edges = np.round(edges).astype(int)

    tokens = []
    for i in range(T):
        a = int(edges[i])
        b = int(edges[i + 1])
        if b <= a:
            idx = min(a, n_frames - 1)
            seg = X_frames[idx:idx + 1, :]
        else:
            seg = X_frames[a:b, :]
        tokens.append(seg.mean(axis=0))

    X_tokens = np.stack(tokens, axis=0).astype(np.float32)
    return X_tokens


# =========================
# MAT loader, one sample per file, session wise grouping
# =========================
def _pick_array_from_mat(mat: Dict, prefer_keys: List[str]) -> Optional[np.ndarray]:
    for k in prefer_keys:
        if k in mat and isinstance(mat[k], np.ndarray):
            return mat[k]
    candidates = []
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim in (1, 2):
            candidates.append((k, v))
    if not candidates:
        return None
    candidates.sort(key=lambda kv: np.prod(kv[1].shape), reverse=True)
    return candidates[0][1]


def load_sessions_from_mat_dir(
    mat_dir: str,
    preprocess_cfg: PreprocessConfig,
    model_cfg: ModelConfig,
    eeg_keys: Optional[List[str]] = None,
    rating_keys: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X_all float32 shape (N, T, 504)
      ratings float32 shape (N,) values expected around 1..9
      groups int64 shape (N,) session id per file
    """
    if eeg_keys is None:
        eeg_keys = ["eeg", "EEG", "data", "Data", "signal", "signals", "X"]
    if rating_keys is None:
        rating_keys = ["rating", "ratings", "y", "label", "labels", "attention", "score", "scores"]

    mat_files = [f for f in os.listdir(mat_dir) if f.lower().endswith(".mat")]
    mat_files.sort()
    if len(mat_files) == 0:
        raise FileNotFoundError("No MAT files found in the directory")

    X_list: List[np.ndarray] = []
    r_list: List[np.ndarray] = []
    g_list: List[np.ndarray] = []

    for sid, fname in enumerate(mat_files, start=1):
        path = os.path.join(mat_dir, fname)
        mat = loadmat(path)

        eeg = _pick_array_from_mat(mat, eeg_keys)
        rating = _pick_array_from_mat(mat, rating_keys)

        if eeg is None:
            raise ValueError(f"EEG not found in {fname}")
        if rating is None:
            raise ValueError(f"Rating not found in {fname}")

        eeg = np.squeeze(eeg)
        if eeg.ndim == 1:
            raise ValueError(f"EEG array must be 2D in {fname}")

        rating = np.squeeze(rating).astype(np.float64)
        if rating.ndim == 0:
            r_scalar = float(rating)
        else:
            rating_flat = rating.reshape(-1)
            r_scalar = float(np.median(rating_flat))

        X_frames = extract_stft_frames_504(eeg, preprocess_cfg)
        X_tokens = frames_to_T_tokens(X_frames, model_cfg.T)  # (T, 504)

        X_list.append(X_tokens[None, :, :])  # (1, T, 504)
        r_list.append(np.array([r_scalar], dtype=np.float32))
        g_list.append(np.array([sid], dtype=np.int64))

    X_all = np.concatenate(X_list, axis=0)
    ratings_all = np.concatenate(r_list, axis=0)
    groups = np.concatenate(g_list, axis=0)

    return X_all, ratings_all, groups


# =========================
# Train only tertile discretization
# =========================
def tertile_discretize_train_only(r_train: np.ndarray, r_any: np.ndarray) -> np.ndarray:
    q1, q2 = np.quantile(r_train, [1.0 / 3.0, 2.0 / 3.0])
    y = np.zeros_like(r_any, dtype=np.int64)
    y[r_any > q1] = 1
    y[r_any > q2] = 2
    return y


# =========================
# Metrics
# =========================
def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    preds = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)

    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (conf > lo) & (conf <= hi)
        prop = float(np.mean(in_bin))
        if prop > 0:
            acc = float(np.mean(y_true[in_bin] == preds[in_bin]))
            avg_conf = float(np.mean(conf[in_bin]))
            ece += abs(acc - avg_conf) * prop
    return float(ece)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    y_pred = np.argmax(probs, axis=1)

    out: Dict[str, float] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["kappa"] = float(cohen_kappa_score(y_true, y_pred))

    out["ece"] = float(expected_calibration_error(y_true, probs))
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, probs, multi_class="ovr"))
    except Exception:
        out["roc_auc"] = float("nan")
    return out


# =========================
# Loss variants
# =========================
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


# =========================
# Positional encoding
# =========================
def sinusoidal_positional_encoding(T: int, D: int) -> torch.Tensor:
    pe = torch.zeros(T, D, dtype=torch.float32)
    pos = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * (-np.log(10000.0) / D))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


# =========================
# Transformer block, optional dropout for evaluated regularization
# =========================
class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        norm_type: str,
        attn_dropout: float,
        resid_dropout: float,
        ff_dropout: float,
    ):
        super().__init__()
        self.norm_type = norm_type

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.drop_resid = nn.Dropout(resid_dropout)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(ff_dropout),
        )

    def forward(self, x: torch.Tensor, need_attn: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.norm_type == "pre":
            x1 = self.norm1(x)
            attn_out, attn_w = self.attn(x1, x1, x1, need_weights=need_attn, average_attn_weights=False)
            x = x + self.drop_resid(attn_out)

            x2 = self.norm2(x)
            ff_out = self.ff(x2)
            x = x + self.drop_resid(ff_out)
            return x, attn_w

        attn_out, attn_w = self.attn(x, x, x, need_weights=need_attn, average_attn_weights=False)
        x = self.norm1(x + self.drop_resid(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop_resid(ff_out))
        return x, attn_w


# =========================
# NT AA model, input is (B, T, 504) so tokens are temporal segments
# =========================
class NT_AA(nn.Module):
    def __init__(self, cfg: ModelConfig, num_classes: int = 3):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Linear(cfg.input_dim, cfg.embed_dim)

        if cfg.pos_enc_type == "trainable":
            self.pos_enc = nn.Parameter(torch.zeros(1, cfg.T, cfg.embed_dim))
            self.register_buffer("pos_enc_fixed", torch.zeros(1, cfg.T, cfg.embed_dim), persistent=False)
        elif cfg.pos_enc_type == "sinusoidal":
            pe = sinusoidal_positional_encoding(cfg.T, cfg.embed_dim)
            self.register_buffer("pos_enc_fixed", pe, persistent=False)
            self.pos_enc = None
        else:
            self.register_buffer("pos_enc_fixed", torch.zeros(1, cfg.T, cfg.embed_dim), persistent=False)
            self.pos_enc = None

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    ff_hidden_dim=cfg.ff_hidden_dim,
                    norm_type=cfg.norm_type,
                    attn_dropout=cfg.attn_dropout,
                    resid_dropout=cfg.resid_dropout,
                    ff_dropout=cfg.ff_dropout,
                )
                for _ in range(cfg.depth)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(cfg.embed_dim, 256),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        x_tok = self.embed(x)

        if self.cfg.pos_enc_type == "trainable" and self.pos_enc is not None:
            x_tok = x_tok + self.pos_enc
        else:
            x_tok = x_tok + self.pos_enc_fixed

        attn_list: Optional[List[torch.Tensor]] = [] if return_attn else None
        for blk in self.blocks:
            x_tok, attn_w = blk(x_tok, need_attn=return_attn)
            if return_attn and attn_w is not None:
                attn_list.append(attn_w)

        x_pool = x_tok.mean(dim=1)
        logits = self.classifier(x_pool)
        return logits, attn_list


# =========================
# Train one fold with early stopping on val Macro F1
# =========================
def train_one_fold(
    model: NT_AA,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: TrainConfig,
    class_weights: Optional[np.ndarray],
) -> NT_AA:
    model = model.to(device)

    w_t = None
    if class_weights is not None:
        w_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    if train_cfg.loss_name == "focal":
        loss_fn = FocalLoss(gamma=train_cfg.focal_gamma, weight=w_t)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=w_t)

    if train_cfg.optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    best_state = None
    best_f1 = -1.0
    wait = 0

    for _epoch in range(train_cfg.max_epochs):
        model.train()
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)

            optimizer.zero_grad()
            logits, _ = model(bx, return_attn=False)
            loss = loss_fn(logits, by)
            loss.backward()

            if train_cfg.grad_clip_norm is not None and train_cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)

            optimizer.step()

        model.eval()
        preds_list = []
        y_list = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                logits, _ = model(bx, return_attn=False)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds_list.append(preds)
                y_list.append(by.numpy())

        preds = np.concatenate(preds_list)
        y_true = np.concatenate(y_list)
        val_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= train_cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# =========================
# GroupKFold evaluation
# =========================
def run_groupkfold_evaluation(
    X: np.ndarray,
    ratings_1to9: np.ndarray,
    groups: np.ndarray,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> Dict[str, Tuple[float, float]]:
    gkf = GroupKFold(n_splits=5)
    fold_metrics: List[Dict[str, float]] = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, ratings_1to9, groups), start=1):
        X_tr = X[tr_idx]  # (Ntr, T, 504)
        X_va = X[va_idx]

        r_tr = ratings_1to9[tr_idx]
        r_va = ratings_1to9[va_idx]

        y_tr = tertile_discretize_train_only(r_tr, r_tr)
        y_va = tertile_discretize_train_only(r_tr, r_va)

        scaler = StandardScaler()
        X_tr_flat = X_tr.reshape(-1, model_cfg.input_dim)
        X_va_flat = X_va.reshape(-1, model_cfg.input_dim)

        X_tr_flat = scaler.fit_transform(X_tr_flat)
        X_va_flat = scaler.transform(X_va_flat)

        X_tr_s = X_tr_flat.reshape(X_tr.shape).astype(np.float32)
        X_va_s = X_va_flat.reshape(X_va.shape).astype(np.float32)

        if train_cfg.loss_name == "cw_ce":
            counts = np.bincount(y_tr, minlength=3).astype(np.float64)
            counts[counts == 0] = 1.0
            w = (counts.sum() / (3.0 * counts)).astype(np.float32)
        else:
            w = None

        X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        X_va_t = torch.tensor(X_va_s, dtype=torch.float32)
        y_va_t = torch.tensor(y_va, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=train_cfg.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_va_t, y_va_t), batch_size=train_cfg.batch_size, shuffle=False)

        model = NT_AA(cfg=model_cfg, num_classes=3)
        model = train_one_fold(model, train_loader, val_loader, train_cfg, class_weights=w)

        model.eval()
        probs_list = []
        y_true_list = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                logits, _ = model(bx, return_attn=False)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)
                y_true_list.append(by.numpy())

        probs = np.concatenate(probs_list)
        y_true = np.concatenate(y_true_list)

        m = compute_metrics(y_true, probs)
        fold_metrics.append(m)

        print(
            f"Fold {fold} | "
            f"Acc {m['accuracy']:.5f} | "
            f"F1 {m['f1']:.5f} | "
            f"Kappa {m['kappa']:.5f} | "
            f"ECE {m['ece']:.5f} | "
            f"ROC AUC {m['roc_auc']:.5f}"
        )

    def mean_std(key: str) -> Tuple[float, float]:
        vals = np.array([fm[key] for fm in fold_metrics], dtype=np.float64)
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    summary: Dict[str, Tuple[float, float]] = {}
    for k in ["accuracy", "precision", "recall", "f1", "kappa", "ece", "roc_auc"]:
        summary[k] = mean_std(k)

    print("\nMean and Std across folds")
    for k, (mu, sd) in summary.items():
        print(f"{k}: {mu:.5f} ± {sd:.5f}")

    return summary


# =========================
# Model stats
# =========================
class LogitsOnlyWrapper(nn.Module):
    def __init__(self, model: NT_AA):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        logits, _ = self.model(x, return_attn=False)
        return logits


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def bytes_to_mib(b: int) -> float:
    return float(b) / (1024.0 * 1024.0)


def parameter_memory_bytes(model: nn.Module, dtype_bytes: int) -> int:
    total = sum(p.numel() for p in model.parameters())
    return int(total * dtype_bytes)


def measure_inference_time_ms(model: NT_AA, T: int, input_dim: int, n_runs: int = 300, warmup: int = 50) -> float:
    model.eval()
    x = torch.randn(1, T, input_dim, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x, return_attn=False)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(x, return_attn=False)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t1 = time.time()
    return float(((t1 - t0) / n_runs) * 1000.0)


def compute_flops(model: NT_AA, T: int, input_dim: int) -> float:
    if thop_profile is None:
        return float("nan")
    wrapper = LogitsOnlyWrapper(model).to(device)
    dummy = torch.randn(1, T, input_dim, device=device)
    try:
        flops, _ = thop_profile(wrapper, inputs=(dummy,))
        return float(flops)
    except Exception:
        return float("nan")


# =========================
# Main
# =========================
if __name__ == "__main__":
    preprocess_cfg = PreprocessConfig()

    model_cfg = ModelConfig(
        T=8,
        input_dim=504,
        embed_dim=64,
        num_heads=4,
        depth=2,
        ff_hidden_dim=128,
        pos_enc_type="trainable",
        norm_type="post",
        head_dropout=0.2,
        attn_dropout=0.0,
        resid_dropout=0.0,
        ff_dropout=0.0,
    )

    train_cfg = TrainConfig(
        batch_size=64,
        lr=1e-3,
        max_epochs=84,
        patience=10,
        optimizer_name="adam",
        weight_decay=0.0,
        loss_name="ce",
        grad_clip_norm=1.0,
    )

    mat_dir = "PATH_TO_MAT_SESSION_DIR"
    X, ratings_1to9, groups = load_sessions_from_mat_dir(mat_dir, preprocess_cfg, model_cfg)

    _summary = run_groupkfold_evaluation(
        X=X,
        ratings_1to9=ratings_1to9,
        groups=groups,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )

    model = NT_AA(cfg=model_cfg, num_classes=3).to(device)
    total_p, train_p = count_params(model)

    mem_fp32_b = parameter_memory_bytes(model, 4)
    mem_fp16_b = parameter_memory_bytes(model, 2)
    mem_int8_b = parameter_memory_bytes(model, 1)

    flops = compute_flops(model, model_cfg.T, model_cfg.input_dim)
    inf_ms = measure_inference_time_ms(model, model_cfg.T, model_cfg.input_dim)

    print("\nModel stats")
    print(f"Total params: {total_p}")
    print(f"Trainable params: {train_p}")
    print(f"Param memory FP32 MiB: {bytes_to_mib(mem_fp32_b):.4f}")
    print(f"Param memory FP16 MiB: {bytes_to_mib(mem_fp16_b):.4f}")
    print(f"Param memory INT8 MiB: {bytes_to_mib(mem_int8_b):.4f}")
    print(f"FLOPs per forward: {flops}")
    print(f"Inference time ms per sample: {inf_ms:.6f}")
