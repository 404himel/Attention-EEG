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




def sinusoidal_positional_encoding(T: int, D: int) -> torch.Tensor:
    pe = torch.zeros(T, D, dtype=torch.float32)
    pos = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, D, 2, dtype=torch.float32) * (-np.log(10000.0) / D))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)

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

    # =========================
# Initialize
# =========================
model = NT_AAR_Model_Tokens(input_dim, embed_dim=embed_dim, num_heads=4, num_classes=num_classes, T=T).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

