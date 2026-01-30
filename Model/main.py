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
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt

try:
    from thop import profile as thop_profile
except Exception:
    thop_profile = None


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelConfig:
    T: int
    input_dim: int
    embed_dim: int
    num_heads: int
    depth: int
    ff_hidden_dim: int
    head_dropout: float = 0.2
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    ff_dropout: float = 0.0


@dataclass
class TrainConfig:
    batch_size: int
    lr: float
    max_epochs: int
    patience: int = 10
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0

class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        attn_dropout: float,
        resid_dropout: float,
        ff_dropout: float,
    ):
        super().__init__()

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

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_w = self.attn(
            x, x, x,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        x = self.norm1(x + self.drop_resid(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop_resid(ff_out))
        return x, attn_w


class NT_AA(nn.Module):
    def __init__(self, cfg: ModelConfig, num_classes: int):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Linear(cfg.input_dim, cfg.embed_dim)
        self.pos_enc = nn.Parameter(torch.zeros(1, cfg.T, cfg.embed_dim))

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    ff_hidden_dim=cfg.ff_hidden_dim,
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
        x_tok = x_tok + self.pos_enc

        attn_list: Optional[List[torch.Tensor]] = [] if return_attn else None
        for blk in self.blocks:
            x_tok, attn_w = blk(x_tok, return_attn=return_attn)
            if return_attn and attn_w is not None:
                attn_list.append(attn_w)

        x_pool = x_tok.mean(dim=1)
        logits = self.classifier(x_pool)
        return logits, attn_list


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    ece = 0.0
    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        in_bin = (confs > lo) & (confs <= hi)
        prop = float(np.mean(in_bin))
        if prop > 0.0:
            acc = float(np.mean(y_true[in_bin] == preds[in_bin]))
            avg_conf = float(np.mean(confs[in_bin]))
            ece += abs(acc - avg_conf) * prop
    return float(ece)


def compute_metrics_from_probs(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    y_pred = np.argmax(probs, axis=1)
    out: Dict[str, float] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["kappa"] = float(cohen_kappa_score(y_true, y_pred))
    return out



def train_one_fold_with_history(
    model: NT_AA,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: TrainConfig,
    seed_for_fold: int,
) -> Tuple[NT_AA, Dict[str, np.ndarray]]:
    set_seed(seed_for_fold)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_f1 = float("-inf")
    wait = 0

    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []

    start_time = time.time()

    for epoch in range(train_cfg.max_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)

            optimizer.zero_grad()
            logits, _ = model(bx, return_attn=False)
            loss = loss_fn(logits, by)
            loss.backward()

            if train_cfg.grad_clip_norm and train_cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)

            optimizer.step()

            running_loss += float(loss.item()) * bx.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == by).sum().item())
            total += int(by.size(0))

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        probs_list: List[np.ndarray] = []
        true_list: List[np.ndarray] = []

        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                by = by.to(device)

                logits, _ = model(bx, return_attn=False)
                loss = loss_fn(logits, by)

                val_running_loss += float(loss.item()) * bx.size(0)
                pred = torch.argmax(logits, dim=1)
                val_correct += int((pred == by).sum().item())
                val_total += int(by.size(0))

                probs = F.softmax(logits, dim=1).detach().cpu().numpy()
                probs_list.append(probs)
                true_list.append(by.detach().cpu().numpy())

        val_loss = val_running_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        probs_all = np.concatenate(probs_list, axis=0)
        true_all = np.concatenate(true_list, axis=0)
        val_f1 = float(f1_score(true_all, np.argmax(probs_all, axis=1), average="macro", zero_division=0))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val Macro F1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= train_cfg.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTotal Training Time: {training_time / 60:.2f} minutes")

    if best_state is not None:
        model.load_state_dict(best_state)

    history = {
        "train_losses": np.array(train_losses, dtype=np.float32),
        "val_losses": np.array(val_losses, dtype=np.float32),
        "train_accuracies": np.array(train_accs, dtype=np.float32),
        "val_accuracies": np.array(val_accs, dtype=np.float32),
    }
    return model, history


def evaluate_with_attention(model: NT_AA, val_loader: DataLoader, T: int) -> Dict[str, object]:
    model.eval()

    all_true: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    attn_maps_batches: List[np.ndarray] = []

    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            logits, attn_list = model(bx, return_attn=True)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()

            all_probs.append(probs)
            all_true.append(by.numpy())

            if attn_list is not None and len(attn_list) > 0:
                per_layer: List[np.ndarray] = []
                for a in attn_list:
                    a_np = a.detach().cpu().numpy()
                    if a_np.ndim == 4:
                        a_np = a_np.mean(axis=1)
                    per_layer.append(a_np)
                a_stack = np.stack(per_layer, axis=0)
                a_mean = a_stack.mean(axis=0)
                attn_maps_batches.append(a_mean)

    y_true = np.concatenate(all_true, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    preds = np.argmax(probs, axis=1)

    m = compute_metrics_from_probs(y_true, probs)
    m["ece"] = expected_calibration_error(y_true, probs)
    try:
        m["roc_auc"] = float(roc_auc_score(y_true, probs, multi_class="ovr"))
    except Exception:
        m["roc_auc"] = float("nan")

    if len(attn_maps_batches) > 0:
        attn_maps = np.concatenate(attn_maps_batches, axis=0)
    else:
        attn_maps = np.zeros((len(y_true), T, T), dtype=np.float32)

    avg_attn = attn_maps.mean(axis=0)

    return {
        "y_true": y_true,
        "preds": preds,
        "probs": probs,
        "metrics": m,
        "attn_maps": attn_maps,
        "avg_attn": avg_attn,
    }



@dataclass
class SearchSpace:
    embed_dims: Tuple[int, ...] = (32, 64, 128)
    num_heads: Tuple[int, ...] = (2, 4, 8)
    batch_sizes: Tuple[int, ...] = (32, 64, 128)
    epochs: Tuple[int, ...] = (60, 80, 100, 120)
    depth: Tuple[int, ...] = (1, 2, 3, 4)
    ff_hidden: Tuple[int, ...] = (64, 128, 256, 512)
    lr_low: float = 1e-4
    lr_high: float = 2e-3


def sample_config(rng: np.random.Generator, T: int, input_dim: int, space: SearchSpace):
    embed_dim = int(rng.choice(space.embed_dims))
    num_heads = int(rng.choice(space.num_heads))

    if embed_dim % num_heads != 0:
        return None

    model_cfg = ModelConfig(
        T=T,
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=int(rng.choice(space.depth)),
        ff_hidden_dim=int(rng.choice(space.ff_hidden)),
        head_dropout=0.2,
        attn_dropout=0.0,
        resid_dropout=0.0,
        ff_dropout=0.0,
    )

    train_cfg = TrainConfig(
        batch_size=int(rng.choice(space.batch_sizes)),
        lr=float(rng.uniform(space.lr_low, space.lr_high)),
        max_epochs=int(rng.choice(space.epochs)),
        patience=10,
        weight_decay=0.0,
        grad_clip_norm=1.0,
    )

    return model_cfg, train_cfg


def run_groupkfold_for_config(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    n_splits: int = 5,
    base_seed: int = 42,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics: List[Dict[str, float]] = []

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    num_classes = int(np.max(y) + 1)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), start=1):
        train_loader = DataLoader(
            TensorDataset(X_t[tr_idx], y_t[tr_idx]),
            batch_size=train_cfg.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_t[va_idx], y_t[va_idx]),
            batch_size=train_cfg.batch_size,
            shuffle=False,
        )

        model = NT_AA(cfg=model_cfg, num_classes=num_classes)
        model, _history = train_one_fold_with_history(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_cfg=train_cfg,
            seed_for_fold=base_seed + fold,
        )

        out = evaluate_with_attention(model, val_loader, T=model_cfg.T)
        m = out["metrics"]
        fold_metrics.append(
            {
                "accuracy": float(m["accuracy"]),
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "f1": float(m["f1"]),
                "kappa": float(m["kappa"]),
            }
        )

    def mean_std(key: str) -> Tuple[float, float]:
        vals = np.array([fm[key] for fm in fold_metrics], dtype=np.float64)
        return float(np.mean(vals)), float(np.std(vals))

    summary: Dict[str, float] = {}
    for k in ["accuracy", "precision", "recall", "f1", "kappa"]:
        mu, sd = mean_std(k)
        summary[k] = mu
        summary[k + "_std"] = sd

    return fold_metrics, summary


def random_search_tuning(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_trials: int,
    space: SearchSpace,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[Dict, List[Dict[str, float]], Dict[str, float]]:
    rng = np.random.default_rng(seed)

    T = int(X.shape[1])
    input_dim = int(X.shape[2])

    best_mean_f1 = float("-inf")
    best_pack: Optional[Dict] = None
    best_fold_metrics: List[Dict[str, float]] = []
    best_summary: Dict[str, float] = {}

    for trial in range(1, n_trials + 1):
        sampled = sample_config(rng, T, input_dim, space)
        if sampled is None:
            continue

        model_cfg, train_cfg = sampled

        fold_metrics, summary = run_groupkfold_for_config(
            X=X,
            y=y,
            groups=groups,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            n_splits=n_splits,
            base_seed=seed + trial * 100,
        )

        mean_f1 = summary["f1"]
        mean_acc = summary["accuracy"]
        mean_kappa = summary["kappa"]

        print(
            "Trial", trial,
            "Mean Macro F1", f"{mean_f1:.5f}",
            "Mean Acc", f"{mean_acc:.5f}",
            "Mean Kappa", f"{mean_kappa:.5f}",
            "Cfg",
            {
                "embed_dim": model_cfg.embed_dim,
                "num_heads": model_cfg.num_heads,
                "depth": model_cfg.depth,
                "ff_hidden_dim": model_cfg.ff_hidden_dim,
                "batch_size": train_cfg.batch_size,
                "lr": train_cfg.lr,
                "max_epochs": train_cfg.max_epochs,
                "patience": train_cfg.patience,
            },
        )

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_pack = {
                "model_cfg": model_cfg.__dict__,
                "train_cfg": train_cfg.__dict__,
            }
            best_fold_metrics = fold_metrics
            best_summary = summary

    if best_pack is None:
        raise RuntimeError("No valid configuration was sampled, check embed_dim and num_heads")

    return best_pack, best_fold_metrics, best_summary


def print_table4(fold_metrics: List[Dict[str, float]]) -> None:
    keys = ["accuracy", "precision", "recall", "f1", "kappa"]
    header = ["Metric"] + [f"Fold {i}" for i in range(1, len(fold_metrics) + 1)] + ["Mean ± Std"]
    print("\nTable 4  Performance metrics across five cross validation folds")
    print(" | ".join(header))

    for k in keys:
        vals = [fm[k] for fm in fold_metrics]
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        row = [k] + [f"{v:.5f}" for v in vals] + [f"{mu:.5f} ± {sd:.5f}"]
        print(" | ".join(row))


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


def compute_flops(model: NT_AA, T: int, input_dim: int) -> float:
    if thop_profile is None:
        return float("nan")
    wrapper = LogitsOnlyWrapper(model).to(device)
    dummy = torch.randn(1, T, input_dim, device=device)
    try:
        flops, _params = thop_profile(wrapper, inputs=(dummy,))
        return float(flops)
    except Exception:
        return float("nan")


def measure_inference_time_ms(model: NT_AA, T: int, input_dim: int, n_samples: int = 100) -> float:
    model.eval()
    dummy = torch.randn(1, T, input_dim, device=device)

    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy, return_attn=False)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_samples):
            _ = model(dummy, return_attn=False)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t1 = time.time()
    return float(((t1 - t0) / n_samples) * 1000.0)


if __name__ == "__main__":
    """
    Hyperparameter Selection Procedure
    Random search with GroupKFold subject agnostic protocol.
    For each sampled configuration, five fold GroupKFold is used on training subjects.
    Primary selection metric is mean validation Macro F1.
    Secondary metrics are mean Accuracy and mean Cohen kappa.
    Adam optimizer, CE loss, early stopping patience 10 based on validation Macro F1.
    """

    X = np.load("X.npy")        # shape (N, T, input_dim)
    y = np.load("y.npy")        # shape (N,)
    groups = np.load("groups.npy")

    space = SearchSpace()

    best_pack, best_fold_metrics, best_summary = random_search_tuning(
        X=X,
        y=y,
        groups=groups,
        n_trials=25,
        space=space,
        n_splits=5,
        seed=42,
    )

    print("\nSelected optimal hyperparameters")
    print(best_pack)
    print_table4(best_fold_metrics)

    model_cfg = ModelConfig(**best_pack["model_cfg"])
    train_cfg = TrainConfig(**best_pack["train_cfg"])

    num_classes = int(np.max(y) + 1)
    model = NT_AA(cfg=model_cfg, num_classes=num_classes).to(device)

    total_params, trainable_params = count_params(model)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")


    flops = compute_flops(model, T=model_cfg.T, input_dim=model_cfg.input_dim)
    if np.isnan(flops):
        print("FLOPs: nan")
    else:
        print(f"FLOPs: {int(flops):,}")


    gkf = GroupKFold(n_splits=5)
    tr_idx, va_idx = next(gkf.split(X, y, groups))

    X_tr = torch.tensor(X[tr_idx], dtype=torch.float32)
    y_tr = torch.tensor(y[tr_idx], dtype=torch.long)
    X_va = torch.tensor(X[va_idx], dtype=torch.float32)
    y_va = torch.tensor(y[va_idx], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=train_cfg.batch_size, shuffle=False)

    print(f"\nTrain samples: {len(tr_idx)}, Val samples: {len(va_idx)}\n")

    model, history = train_one_fold_with_history(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=train_cfg,
        seed_for_fold=123,
    )

    out = evaluate_with_attention(model, val_loader, T=model_cfg.T)
    metrics = out["metrics"]

    print("\nPerformance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.7f}")
    print(f"Precision: {metrics['precision']:.7f}")
    print(f"Recall: {metrics['recall']:.7f}")
    print(f"F1-score: {metrics['f1']:.7f}")
    print(f"Cohen's Kappa: {metrics['kappa']:.7f}")
    print(f"ECE: {metrics['ece']:.7f}")
    if not np.isnan(metrics["roc_auc"]):
        print(f"ROC-AUC: {metrics['roc_auc']:.7f}")


    avg_inference_time = measure_inference_time_ms(model, T=model_cfg.T, input_dim=model_cfg.input_dim, n_samples=100)
    print(f"\nAverage Inference Time: {avg_inference_time:.4f} ms/sample")


    train_accuracies = history["train_accuracies"]
    val_accuracies = history["val_accuracies"]
    train_losses = history["train_losses"]
    val_losses = history["val_losses"]

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label="Train Accuracy", color="blue")
    plt.plot(val_accuracies, label="Validation Accuracy", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy vs Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss vs Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    cm = confusion_matrix(out["y_true"], out["preds"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Validation Set)")
    plt.tight_layout()
    plt.show()

    avg_attn = out["avg_attn"]
    plt.figure(figsize=(8, 7))
    im = plt.imshow(avg_attn, cmap="viridis", aspect="auto")
    plt.colorbar(im, label="Attention Weight")
    plt.xlabel("Key Token Index")
    plt.ylabel("Query Token Index")
    plt.title(f"Average Attention Map (T={model_cfg.T} tokens)")
    plt.xticks(range(model_cfg.T))
    plt.yticks(range(model_cfg.T))
    for i in range(model_cfg.T):
        for j in range(model_cfg.T):
            _ = plt.text(
                j,
                i,
                f"{avg_attn[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if avg_attn[i, j] > avg_attn.max() / 2 else "black",
                fontsize=8,
            )
    plt.tight_layout()
    plt.show()


    save_dict = {
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "attention_maps": out["attn_maps"],
        "avg_attention": out["avg_attn"],
        "best_hparams": np.array([str(best_pack)], dtype=object),
    }
    np.savez("NT_AA_training_history.npz", **save_dict)
    print("\nTraining history saved as 'NT_AA_training_history.npz'")
