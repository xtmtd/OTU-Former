"""Training entry points for pretrain and finetune."""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from otuformer.embedding.evaluator import (
    compute_clustering_metrics,
    compute_knn_accuracy,
    compute_linear_probing,
    compute_map,
    compute_recall_at_k,
    run_umap,
)
from otuformer.training.dataset import (
    MetricDataset,
    MultiCropDataset,
    _build_recursive_index,
    _make_eval_transform,
    _resolve_image_path,
    _supports_recursive_lookup,
)
from otuformer.training.loss import (
    ArcFaceLoss,
    LOSS_REGISTRY,
)
from otuformer.training.model import OTUFormerEncoder
from otuformer.utils.checkpoint import load_checkpoint, save_checkpoint


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_cpus(cpus: int) -> None:
    if cpus > 0:
        torch.set_num_threads(cpus)


def _resolve_device(device: str) -> torch.device:
    requested = (device or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


@torch.no_grad()
def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)


def _cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    warmup_epochs: float = 0.0,
    start_warmup_value: float = 0.0,
) -> np.ndarray:
    total_iters = max(1, epochs * niter_per_ep)
    warmup_iters = int(max(0.0, warmup_epochs) * niter_per_ep)
    warmup_iters = min(warmup_iters, total_iters)
    schedule = np.ones(total_iters, dtype=np.float32) * float(final_value)

    if warmup_iters > 0:
        warmup_schedule = np.linspace(
            start_warmup_value,
            base_value,
            warmup_iters,
            dtype=np.float32,
        )
        schedule[:warmup_iters] = warmup_schedule

    iters = np.arange(total_iters, dtype=np.float32)
    after = iters[warmup_iters:]
    if len(after) > 0:
        schedule[warmup_iters:] = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * (after - warmup_iters) / max(1, len(after)))
        )
    return schedule


def _compute_global_loss(
    student_global: list[torch.Tensor],
    teacher_global: list[torch.Tensor],
    temp_student: float,
    temp_teacher: float,
    disable_cross_view_loss: bool = False,
) -> torch.Tensor:
    """Compute global distillation loss following ref/ibot20260115.py.

    When disable_cross_view_loss=False (default): full Cartesian product.
    When disable_cross_view_loss=True: matching-pair zip only.
    """
    device = student_global[0].device
    loss = torch.tensor(0.0, device=device)
    if disable_cross_view_loss:
        for s, t in zip(student_global, teacher_global):
            loss += _ssl_loss(s, t, temp_student, temp_teacher)
        loss /= max(1, len(student_global))
    else:
        for s in student_global:
            for t in teacher_global:
                loss += _ssl_loss(s, t, temp_student, temp_teacher)
        loss /= max(1, len(student_global) * len(teacher_global))
    return loss


def _compute_local_loss(
    local_views_student: list[torch.Tensor],
    teacher_global: list[torch.Tensor],
    temp_student: float,
    temp_teacher: float,
) -> torch.Tensor:
    """Compute local-to-global distillation loss following ref/ibot20260115.py.

    Averages over all local_student x teacher_global pairs.
    """
    if not local_views_student:
        return torch.tensor(0.0, device=teacher_global[0].device)
    device = local_views_student[0].device
    loss = torch.tensor(0.0, device=device)
    for lv in local_views_student:
        for t in teacher_global:
            loss += _ssl_loss(lv, t, temp_student, temp_teacher)
    loss /= max(1, len(local_views_student) * len(teacher_global))
    return loss


def _update_teacher_center(
    center: torch.Tensor,
    teacher_global: torch.Tensor,
) -> torch.Tensor:
    """Update teacher center buffer with EMA of current-iteration teacher global outputs.

    Uses fixed 0.9/0.1 EMA coefficients matching ref/ibot20260115.py.
    """
    batch_mean = teacher_global.mean(dim=0, keepdim=True)
    return center * 0.9 + batch_mean * 0.1


def _build_teacher_temp_schedule(
    total_iters: int,
    teacher_temp_start: float,
    teacher_temp_end: float,
) -> np.ndarray:
    warmup_iters = int(total_iters * 0.7)
    if warmup_iters > 0:
        return np.concatenate(
            [
                np.linspace(
                    teacher_temp_start,
                    teacher_temp_end,
                    warmup_iters,
                    dtype=np.float32,
                ),
                np.ones(total_iters - warmup_iters, dtype=np.float32)
                * float(teacher_temp_end),
            ]
        )
    return np.ones(total_iters, dtype=np.float32) * float(teacher_temp_end)


def _ssl_loss(
    student_out: torch.Tensor,
    teacher_out: torch.Tensor,
    temp_student: float,
    temp_teacher: float,
) -> torch.Tensor:
    student_sim = student_out @ teacher_out.T / temp_student
    with torch.no_grad():
        teacher_sim = teacher_out @ teacher_out.T / temp_teacher
        teacher_sim = teacher_sim - teacher_sim.mean(dim=1, keepdim=True)
        teacher_probs = F.softmax(teacher_sim, dim=1)
    student_log_probs = F.log_softmax(student_sim, dim=1)
    return -(teacher_probs * student_log_probs).sum(dim=1).mean()


def _masked_token_loss(
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    mask_ratio: float,
    model_name: str,
) -> torch.Tensor:
    b, n, c = student_tokens.shape
    num_mask = max(1, int(mask_ratio * n))
    mask_indices = torch.rand(b, n, device=student_tokens.device).argsort(dim=1)[
        :, :num_mask
    ]

    student_masked = torch.gather(
        student_tokens,
        dim=1,
        index=mask_indices.unsqueeze(-1).expand(-1, -1, c),
    )
    teacher_masked = torch.gather(
        teacher_tokens,
        dim=1,
        index=mask_indices.unsqueeze(-1).expand(-1, -1, c),
    ).detach()

    if "eva" in model_name.lower():
        return F.mse_loss(student_masked, teacher_masked)

    student_masked = F.normalize(student_masked, dim=-1)
    teacher_masked = F.normalize(teacher_masked, dim=-1)
    loss = 2.0 - 2.0 * (student_masked * teacher_masked).sum(dim=-1)
    return loss.mean()


@dataclass
class InstantMetricsLogger:
    path: Path
    mode: str = "pretrain"

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        base_fields = ["iteration", "epoch", "step"]
        if self.mode == "pretrain":
            metric_fields = [
                "loss",
                "global_loss",
                "local_loss",
                "mask_loss",
                "lr",
                "teacher_temp",
                "grad_norm",
                "feature_std",
                "embedding_norm_mean",
                "cls_token_norm_mean",
                "teacher_center_norm",
                "cosine_similarity",
            ]
        else:
            metric_fields = [
                "loss",
                "lr",
                "grad_norm",
                "feature_std",
                "embedding_norm_mean",
                "cls_token_norm_mean",
            ]
        self.fieldnames = base_fields + metric_fields
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, **kwargs: Any) -> None:
        row = {k: kwargs.get(k, "") for k in self.fieldnames}
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

    def plot(self, out_dir: Path, metrics_path: Path | None = None) -> None:
        if not self.path.exists():
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import pandas as pd
        except ImportError:
            print("[Warning] matplotlib/pandas missing -> skip training curves.")
            return

        df_instant = pd.read_csv(self.path)
        if df_instant.empty:
            return

        df_metrics = None
        if metrics_path is not None and metrics_path.exists():
            df_metrics = pd.read_csv(metrics_path)
            if "epoch" in df_metrics.columns:
                df_metrics = df_metrics[df_metrics["epoch"].notna()]

        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.flatten()
        plot_idx = 0

        ax = axes[plot_idx]
        if "loss" in df_instant.columns:
            ax.plot(
                df_instant["iteration"],
                df_instant["loss"],
                label="Total loss",
                linewidth=1.5,
            )
        if "global_loss" in df_instant.columns:
            ax.plot(
                df_instant["iteration"],
                df_instant["global_loss"],
                label="Global loss",
                alpha=0.8,
            )
        if "local_loss" in df_instant.columns:
            ax.plot(
                df_instant["iteration"],
                df_instant["local_loss"],
                label="Local loss",
                alpha=0.8,
            )
        if "mask_loss" in df_instant.columns:
            ax.plot(
                df_instant["iteration"],
                df_instant["mask_loss"],
                label="Mask loss",
                alpha=0.8,
            )
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title("Training Losses")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

        ax = axes[plot_idx]
        has_cos = "cosine_similarity" in df_instant.columns
        has_grad = "grad_norm" in df_instant.columns
        second_panel_has_data = False
        if has_cos and has_grad:
            ax_twin = ax.twinx()
            ax.plot(
                df_instant["iteration"],
                df_instant["cosine_similarity"],
                label="Cosine Similarity",
                color="tab:blue",
                linewidth=1.5,
            )
            ax.set_ylabel("Cosine Similarity", color="tab:blue")
            ax.tick_params(axis="y", labelcolor="tab:blue")
            ax_twin.plot(
                df_instant["iteration"],
                df_instant["grad_norm"],
                label="Grad Norm",
                color="tab:orange",
                linewidth=1.5,
                alpha=0.8,
            )
            ax_twin.set_ylabel("Gradient Norm", color="tab:orange")
            ax_twin.tick_params(axis="y", labelcolor="tab:orange")
            ax.set_title("Cosine Similarity & Grad Norm")
            second_panel_has_data = True
        elif has_cos:
            ax.plot(
                df_instant["iteration"],
                df_instant["cosine_similarity"],
                label="Cosine Similarity",
                color="tab:blue",
            )
            ax.set_title("Cosine Similarity")
            second_panel_has_data = True
        elif has_grad:
            ax.plot(
                df_instant["iteration"],
                df_instant["grad_norm"],
                label="Grad Norm",
                color="tab:orange",
            )
            ax.set_title("Gradient Norm")
            second_panel_has_data = True
        if second_panel_has_data:
            ax.set_xlabel("Iteration")
            if has_cos and has_grad:
                lines_l, labels_l = ax.get_legend_handles_labels()
                lines_r, labels_r = ax_twin.get_legend_handles_labels()
                ax.legend(
                    lines_l + lines_r,
                    labels_l + labels_r,
                    fontsize=8,
                    loc="best",
                )
            else:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        else:
            ax.axis("off")

        ax = axes[plot_idx]
        has_left = any(
            c in df_instant.columns for c in ["feature_std", "embedding_norm_mean"]
        )
        has_cls = "cls_token_norm_mean" in df_instant.columns
        if has_left:
            if "feature_std" in df_instant.columns:
                ax.plot(
                    df_instant["iteration"],
                    df_instant["feature_std"],
                    label="Feature Std",
                    color="tab:green",
                )
            if "embedding_norm_mean" in df_instant.columns:
                ax.plot(
                    df_instant["iteration"],
                    df_instant["embedding_norm_mean"],
                    label="Embedding Norm",
                    color="tab:cyan",
                )
        if has_cls:
            ax_twin = ax.twinx()
            ax_twin.plot(
                df_instant["iteration"],
                df_instant["cls_token_norm_mean"],
                label="CLS Token Norm",
                color="tab:purple",
                alpha=0.8,
            )
            ax_twin.set_ylabel("CLS Token Norm", color="tab:purple")
            ax_twin.tick_params(axis="y", labelcolor="tab:purple")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Feature Std / Embedding Norm")
        ax.set_title("Feature Statistics")
        if has_cls:
            lines_l, labels_l = ax.get_legend_handles_labels()
            lines_r, labels_r = ax_twin.get_legend_handles_labels()
            ax.legend(lines_l + lines_r, labels_l + labels_r, fontsize=8, loc="best")
        else:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

        ax = axes[plot_idx]
        fourth_panel_has_data = False
        if "teacher_center_norm" in df_instant.columns:
            ax.plot(
                df_instant["iteration"],
                df_instant["teacher_center_norm"],
                label="Teacher center",
            )
            fourth_panel_has_data = True
        if "lr" in df_instant.columns:
            ax_twin = ax.twinx()
            ax_twin.plot(
                df_instant["iteration"],
                df_instant["lr"],
                label="Learning Rate",
                color="tab:red",
            )
            ax_twin.set_yscale("log")
            ax_twin.set_ylabel("Learning Rate", color="tab:red")
            ax_twin.tick_params(axis="y", labelcolor="tab:red")
            fourth_panel_has_data = True
        if "teacher_center_norm" in df_instant.columns:
            ax.set_ylabel("Teacher Center Norm", color="tab:purple")
            ax.tick_params(axis="y", labelcolor="tab:purple")
        if fourth_panel_has_data:
            ax.set_xlabel("Iteration")
            if (
                "lr" in df_instant.columns
                and "teacher_center_norm" in df_instant.columns
            ):
                ax.set_title("Teacher Center & Learning Rate")
                lines_l, labels_l = ax.get_legend_handles_labels()
                lines_r, labels_r = ax_twin.get_legend_handles_labels()
                ax.legend(
                    lines_l + lines_r,
                    labels_l + labels_r,
                    fontsize=8,
                    loc="best",
                )
            elif "teacher_center_norm" in df_instant.columns:
                ax.set_title("Teacher Center Norm")
                ax.legend(fontsize=8)
            else:
                ax.set_title("Learning Rate")
                lines_r, labels_r = ax_twin.get_legend_handles_labels()
                ax.legend(lines_r, labels_r, fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        else:
            ax.axis("off")

        if df_metrics is not None and not df_metrics.empty:
            for cols, title in [
                (["Recall@1", "Recall@5", "Recall@10"], "Recall@K"),
                (["kNN_Acc_k1", "kNN_Acc_k5", "kNN_Acc_k20"], "kNN Accuracy"),
                (["NMI", "ARI"], "Clustering"),
                (["mAP", "Linear_Probing_Acc"], "Retrieval/Probe"),
                (["Silhouette_Score", "Purity"], "Structure Quality"),
            ]:
                if plot_idx >= len(axes):
                    break
                ax = axes[plot_idx]
                use_twin = title in {"Structure Quality", "Clustering"}
                ax_right = ax.twinx() if use_twin else None
                # Detect which splits are actually present to avoid empty split labels
                splits_present = [
                    s
                    for s in ["train", "test"]
                    if not df_metrics[df_metrics.get("split", "") == s].empty
                ]
                for split, marker in [("train", "o"), ("test", "s")]:
                    if split not in splits_present:
                        continue
                    sub = df_metrics[df_metrics.get("split", "") == split]
                    multi_split = len(splits_present) > 1
                    for col in cols:
                        if col in sub.columns:
                            numeric = pd.to_numeric(sub[col], errors="coerce")
                            if numeric.notna().any():
                                if use_twin and title == "Structure Quality":
                                    target_ax = (
                                        ax_right if col == "Silhouette_Score" else ax
                                    )
                                elif use_twin and title == "Clustering":
                                    target_ax = ax_right if col == "ARI" else ax
                                else:
                                    target_ax = ax
                                # Only append split suffix when multiple splits are present
                                lbl = f"{col} ({split})" if multi_split else col
                                color = None
                                if use_twin and title == "Structure Quality":
                                    color = (
                                        "tab:red"
                                        if col == "Silhouette_Score"
                                        else "tab:blue"
                                    )
                                elif use_twin and title == "Clustering":
                                    color = "tab:red" if col == "ARI" else "tab:blue"
                                target_ax.plot(
                                    sub["epoch"],
                                    numeric,
                                    marker=marker,
                                    label=lbl,
                                    color=color,
                                )
                ax.set_xlabel("Epoch")
                ax.set_title(title)
                if use_twin and title == "Structure Quality":
                    ax.set_ylabel("Purity", color="tab:blue")
                    ax.tick_params(axis="y", labelcolor="tab:blue")
                    ax_right.set_ylabel("Silhouette Score", color="tab:red")
                    ax_right.tick_params(axis="y", labelcolor="tab:red")
                    lines_l, labels_l = ax.get_legend_handles_labels()
                    lines_r, labels_r = ax_right.get_legend_handles_labels()
                    ax.legend(lines_l + lines_r, labels_l + labels_r, fontsize=7)
                elif use_twin and title == "Clustering":
                    ax.set_ylabel("NMI", color="tab:blue")
                    ax.tick_params(axis="y", labelcolor="tab:blue")
                    ax_right.set_ylabel("ARI", color="tab:red")
                    ax_right.tick_params(axis="y", labelcolor="tab:red")
                    lines_l, labels_l = ax.get_legend_handles_labels()
                    lines_r, labels_r = ax_right.get_legend_handles_labels()
                    ax.legend(lines_l + lines_r, labels_l + labels_r, fontsize=7)
                else:
                    ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
                plot_idx += 1

        for i in range(plot_idx, len(axes)):
            axes[i].axis("off")

        fig.tight_layout()
        out_path = out_dir / f"training_curves_{self.mode}.pdf"
        fig.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"[Info] Saved training curves to {out_path}")


@dataclass
class EnhancedMetricsLogger:
    path: Path
    mode: str = "pretrain"

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        base_fields = ["epoch", "split"]
        metric_fields = [
            "NMI",
            "ARI",
            "Recall@1",
            "Recall@5",
            "Recall@10",
            "kNN_Acc_k1",
            "kNN_Acc_k5",
            "kNN_Acc_k20",
            "Linear_Probing_Acc",
            "mAP",
            "Silhouette_Score",
            "Purity",
        ]
        self.fieldnames = base_fields + metric_fields
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, epoch: int, split: str, metrics: dict[str, Any]) -> None:
        row = {"epoch": epoch, "split": split}
        for k in self.fieldnames[2:]:
            row[k] = metrics.get(k, "")
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def _extract_tokens(model: OTUFormerEncoder, images: torch.Tensor) -> torch.Tensor:
    feats = model.backbone.forward_features(images)
    if isinstance(feats, dict):
        return feats["x"]
    return feats


def _infer_backbone_image_size(model: OTUFormerEncoder, fallback: int = 224) -> int:
    patch_embed = getattr(model.backbone, "patch_embed", None)
    if patch_embed is not None:
        img_size = getattr(patch_embed, "img_size", None)
        if isinstance(img_size, (tuple, list)) and len(img_size) >= 1:
            return int(img_size[0])
        if isinstance(img_size, int):
            return int(img_size)
    default_cfg = getattr(model.backbone, "default_cfg", None)
    if isinstance(default_cfg, dict):
        input_size = default_cfg.get("input_size")
        if isinstance(input_size, (tuple, list)) and len(input_size) >= 3:
            return int(input_size[1])
    return int(fallback)


def _compute_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            n = p.grad.data.norm(2)
            total += float(n.item() ** 2)
    return float(total**0.5)


@torch.no_grad()
def _compute_instant_metrics(
    model: OTUFormerEncoder,
    sample_view: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    tokens = _extract_tokens(model, sample_view.to(device))
    cls_token_raw = tokens[:, 0]
    proj = model.projector(cls_token_raw)
    return {
        "feature_std": float(cls_token_raw.std(dim=0).mean().item()),
        "embedding_norm_mean": float(proj.norm(dim=1).mean().item()),
        "cls_token_norm_mean": float(cls_token_raw.norm(dim=1).mean().item()),
    }


@torch.no_grad()
def _compute_embeddings_from_csv(
    model: OTUFormerEncoder,
    csv_path: Path,
    root_dir: Path,
    image_size: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    import pandas as pd
    from PIL import Image

    df = pd.read_csv(csv_path)
    if "image" not in df.columns:
        raise ValueError(f"CSV {csv_path} missing required 'image' column")

    refs = [str(v) for v in df["image"]]
    missing_direct = [
        ref
        for ref in refs
        if _supports_recursive_lookup(ref) and not (root_dir / Path(ref)).exists()
    ]
    by_relative, by_name = ({}, {})
    if missing_direct:
        by_relative, by_name = _build_recursive_index(root_dir)
    image_paths = [
        _resolve_image_path(root_dir, ref, by_relative, by_name) for ref in refs
    ]
    labels = df["label"].astype(str).to_numpy() if "label" in df.columns else None

    tf = _make_eval_transform(image_size)
    all_embs: list[np.ndarray] = []
    model.eval()
    _ = num_workers
    for start in range(0, len(image_paths), max(1, batch_size)):
        batch_paths = image_paths[start : start + max(1, batch_size)]
        imgs = []
        for p in batch_paths:
            with Image.open(p) as im:
                img = im.convert("RGB")
            imgs.append(tf(img))
        batch = torch.stack(imgs, dim=0).to(device)
        # Use raw CLS token (backbone output) for evaluation, matching ref default
        # (ref uses --use_projector_output flag which defaults to False)
        tokens = _extract_tokens(model, batch)
        emb = tokens[:, 0].cpu().numpy()
        all_embs.append(emb)
    embs = (
        np.concatenate(all_embs, axis=0)
        if all_embs
        else np.zeros((0, 1), dtype=np.float32)
    )
    return embs, labels


def _maybe_subsample_for_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray | None,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if len(embeddings) <= max_samples:
        return embeddings, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(embeddings), size=max_samples, replace=False)
    if labels is None:
        return embeddings[idx], None
    return embeddings[idx], labels[idx]


def _compute_all_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray | None,
    compute_linear_probe: bool,
) -> dict[str, Any]:
    fields = {
        "NMI": "",
        "ARI": "",
        "Recall@1": "",
        "Recall@5": "",
        "Recall@10": "",
        "kNN_Acc_k1": "",
        "kNN_Acc_k5": "",
        "kNN_Acc_k20": "",
        "Linear_Probing_Acc": "",
        "mAP": "",
        "Silhouette_Score": "",
        "Purity": "",
    }
    if labels is None or len(embeddings) == 0:
        return fields

    try:
        fields.update(compute_recall_at_k(embeddings, labels, k_values=[1, 5, 10]))
        fields.update(compute_knn_accuracy(embeddings, labels, k_values=[1, 5, 20]))
        fields["mAP"] = compute_map(embeddings, labels)
        clustering = compute_clustering_metrics(embeddings, labels)
        fields["NMI"] = clustering.get("NMI", "")
        fields["ARI"] = clustering.get("ARI", "")
        fields["Silhouette_Score"] = clustering.get("Silhouette", "")
        fields["Purity"] = clustering.get("Purity", "")
        if compute_linear_probe:
            fields["Linear_Probing_Acc"] = compute_linear_probing(embeddings, labels)
    except Exception as exc:
        print(f"[Warning] Error computing metrics: {exc}")
    return fields


def _compute_and_log_all_metrics(
    args: argparse.Namespace,
    model: OTUFormerEncoder,
    device: torch.device,
    epoch: int,
    logs_dir: Path,
    metrics_logger: EnhancedMetricsLogger,
    eval_image_size: int,
    force_linear_probe: bool = False,
) -> None:
    visualize_csv = getattr(args, "visualize_data", "") or getattr(
        args, "train_data", ""
    )
    compute_lp = force_linear_probe or (((epoch + 1) % 10) == 0)

    if visualize_csv:
        try:
            feats, labels = _compute_embeddings_from_csv(
                model=model,
                csv_path=Path(visualize_csv),
                root_dir=Path(args.input_images_dir),
                image_size=eval_image_size,
                device=device,
                batch_size=max(1, args.batch_size),
                num_workers=max(0, args.num_workers),
            )
            feats_eval, labels_eval = _maybe_subsample_for_metrics(
                feats,
                labels,
                max_samples=max(1, args.metrics_sample_size),
                seed=args.seed,
            )
            metrics = _compute_all_metrics(
                feats_eval, labels_eval, compute_linear_probe=compute_lp
            )
            metrics_logger.log(epoch + 1, "train", metrics)
            print(f"[Metrics] Epoch {epoch + 1}:")
            for k, v in metrics.items():
                if v != "":
                    print(f"  {k}: {float(v):.4f}")
            if labels is not None and len(feats) >= 10:
                out_path = logs_dir / f"umap.train.epoch_{epoch + 1:04d}.pdf"
                run_umap(
                    feats,
                    labels,
                    out_path,
                    n_components=2,
                    n_neighbors=args.umap_n_neighbors,
                    min_dist=args.umap_min_dist,
                    metric=args.umap_metric,
                    max_classes=args.visualize_class_number,
                    title=f"UMAP Train - Epoch {epoch + 1}",
                )
                print(f"[Info] Saved UMAP plot to {out_path}")
        except Exception as exc:
            print(
                f"[Warning] Failed to compute embedding metrics at epoch {epoch + 1}: {exc}"
            )


def run_pretrain(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    _set_cpus(args.cpus)
    device = _resolve_device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = MultiCropDataset(
        csv_path=Path(args.train_data) if getattr(args, "train_data", "") else None,
        images_dir=Path(args.input_images_dir),
        global_crop_size=args.global_crop_size,
        local_crop_size=args.local_crop_size,
        local_crops=args.local_crops,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    student = OTUFormerEncoder(
        model_name=args.model_name,
        out_dim=args.out_dim,
        return_patch_tokens=True,
        img_size=args.global_crop_size,
    ).to(device)

    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    _set_seed(args.seed + 1)
    teacher = OTUFormerEncoder(
        model_name=args.model_name,
        out_dim=args.out_dim,
        return_patch_tokens=True,
        img_size=args.global_crop_size,
    ).to(device)
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    instant_logger = InstantMetricsLogger(
        logs_dir / "instant_metrics.pretrain.csv", mode="pretrain"
    )
    compute_embedding_metrics = bool(getattr(args, "compute_embedding_metrics", True))
    epoch_logger = (
        EnhancedMetricsLogger(logs_dir / "metrics.pretrain.csv", mode="pretrain")
        if compute_embedding_metrics
        else None
    )

    start_epoch = 0
    global_step = 0
    if getattr(args, "resume", ""):
        resume_path = Path(args.resume)
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            if "student" in ckpt:
                student.load_state_dict(ckpt["student"], strict=False)
            elif "model_state_dict" in ckpt:
                student.load_state_dict(ckpt["model_state_dict"], strict=False)
            if "teacher" in ckpt:
                teacher.load_state_dict(ckpt["teacher"], strict=False)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "center" in ckpt:
                teacher.center.copy_(ckpt["center"].to(device))
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            global_step = int(ckpt.get("iteration", start_epoch * max(1, len(loader))))
            print(
                f"[Info] Resume from {resume_path} at epoch {start_epoch}, iteration {global_step}"
            )

    niter_per_epoch = max(1, len(loader))
    total_steps = max(1, args.max_epochs * niter_per_epoch)
    warmup_epochs = float(max(0, args.warmup_epochs))
    warmup_iters = int(warmup_epochs * niter_per_epoch)
    if getattr(args, "resume", "") and global_step >= warmup_iters:
        warmup_epochs = 0.0
        print(
            f"[Info] Already past warmup phase (iter {global_step} >= {warmup_iters}), setting warmup_epochs=0"
        )
    elif getattr(args, "resume", "") and global_step > 0:
        remaining_warmup_iters = max(0, warmup_iters - global_step)
        warmup_epochs = remaining_warmup_iters / max(1, niter_per_epoch)
        print(
            f"[Info] Resuming during warmup phase, adjusting warmup to {warmup_epochs:.2f} epochs"
        )

    lr_schedule = _cosine_scheduler(
        base_value=args.lr,
        final_value=args.lr * 0.01,
        epochs=args.max_epochs,
        niter_per_ep=niter_per_epoch,
        warmup_epochs=warmup_epochs,
        start_warmup_value=args.lr * 0.1,
    )
    momentum_schedule = _cosine_scheduler(
        base_value=args.teacher_momentum,
        final_value=args.teacher_momentum_end,
        epochs=args.max_epochs,
        niter_per_ep=niter_per_epoch,
    )
    temp_student_schedule = np.ones(total_steps, dtype=np.float32) * float(
        args.student_temp
    )
    teacher_temp_schedule = _build_teacher_temp_schedule(
        total_iters=total_steps,
        teacher_temp_start=args.teacher_temp_start,
        teacher_temp_end=args.teacher_temp_end,
    )

    cosine_sim = 0.0
    eval_image_size = (
        int(args.extract_size)
        if int(getattr(args, "extract_size", 0)) > 0
        else _infer_backbone_image_size(teacher, fallback=args.global_crop_size)
    )
    if int(getattr(args, "extract_size", 0)) <= 0:
        print(f"[Info] Auto eval crop size from backbone: {eval_image_size}")

    with torch.no_grad():
        sample_batch = next(iter(loader))
        sample_views = [v.to(device) for v in sample_batch[:2]]
        s_out, _ = student(sample_views[0])
        t_out, _ = teacher(sample_views[0])
        init_cosine = F.cosine_similarity(
            s_out.mean(dim=0, keepdim=True),
            t_out.mean(dim=0, keepdim=True),
        ).item()
        print(f"[Info] Initial cosine similarity: {init_cosine:.4f}")
        print(
            f"[Info] Starting/Resuming with learning rate: {float(lr_schedule[min(global_step, total_steps - 1)]):.6f}"
        )

    for epoch in range(start_epoch, args.max_epochs):
        cosine_sim = 0.0
        student.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.max_epochs}", ncols=150)
        for step, views in enumerate(pbar):
            views = [v.to(device) for v in views]
            global_views = views[:2]
            local_views = views[2:]

            iter_idx = min(global_step, total_steps - 1)
            for group in optimizer.param_groups:
                group["lr"] = float(lr_schedule[iter_idx])

            student_global: list[torch.Tensor] = []
            student_tokens: list[torch.Tensor] = []
            for gv in global_views:
                s_proj, s_tok = student(gv)
                student_global.append(s_proj)
                student_tokens.append(s_tok)

            teacher_global: list[torch.Tensor] = []
            teacher_tokens: list[torch.Tensor] = []
            with torch.no_grad():
                for gv in global_views:
                    t_proj, t_tok = teacher(gv)
                    t_proj = t_proj - teacher.center
                    t_proj = F.normalize(t_proj, dim=-1)
                    teacher_global.append(t_proj)
                    teacher_tokens.append(t_tok)
                all_teacher = torch.cat(teacher_global, dim=0)
                teacher.center = _update_teacher_center(teacher.center, all_teacher)

            temp_student = float(temp_student_schedule[iter_idx])
            temp_teacher = float(teacher_temp_schedule[iter_idx])

            loss_global = _compute_global_loss(
                student_global,
                teacher_global,
                temp_student,
                temp_teacher,
                disable_cross_view_loss=args.disable_cross_view_loss,
            )

            loss_mask = torch.tensor(0.0, device=device)
            for s_tok, t_tok in zip(student_tokens, teacher_tokens):
                loss_mask += _masked_token_loss(
                    s_tok, t_tok, args.mask_ratio, args.model_name
                )
            loss_mask /= max(1, len(student_tokens))

            local_student_projs: list[torch.Tensor] = []
            for lv in local_views:
                l_proj, _ = student(lv)
                local_student_projs.append(l_proj)
            loss_local = _compute_local_loss(
                local_student_projs, teacher_global, temp_student, temp_teacher
            )

            total_loss = (
                loss_global
                + args.lambda_local * loss_local
                + args.lambda_mask * loss_mask
            )

            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = _compute_grad_norm(student)
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
            optimizer.step()

            m = float(momentum_schedule[iter_idx])
            update_teacher(student, teacher, m)

            with torch.no_grad():
                cosine_sim = F.cosine_similarity(
                    student_global[0].mean(dim=0, keepdim=True),
                    teacher_global[0].mean(dim=0, keepdim=True),
                ).item()

            if global_step % max(1, args.log_every_n_steps) == 0:
                with torch.no_grad():
                    instant_metrics = _compute_instant_metrics(
                        student, global_views[0], device
                    )
                instant_logger.log(
                    iteration=global_step,
                    epoch=epoch,
                    step=step,
                    loss=float(total_loss.item()),
                    global_loss=float(loss_global.item()),
                    local_loss=float(loss_local.item()),
                    mask_loss=float(loss_mask.item()),
                    lr=float(optimizer.param_groups[0]["lr"]),
                    teacher_temp=float(temp_teacher),
                    grad_norm=float(grad_norm),
                    teacher_center_norm=float(teacher.center.norm().item()),
                    cosine_similarity=float(cosine_sim),
                    **instant_metrics,
                )

            if step % 10 == 0 or step == len(loader) - 1:
                cos_display = (
                    f"{cosine_sim:.3e}"
                    if abs(cosine_sim) < 1e-3
                    else f"{cosine_sim:.4f}"
                )
                pbar.set_postfix(
                    L=f"{total_loss.item():.2e}",
                    GL=f"{loss_global.item():.2e}",
                    LocL=f"{loss_local.item():.2e}",
                    ML=f"{loss_mask.item():.2e}",
                    COS=cos_display,
                    LR=f"{optimizer.param_groups[0]['lr']:.2e}",
                    MOM=f"{m:.4f}",
                )

            global_step += 1

        should_save = (epoch + 1) % max(1, args.save_every_epochs) == 0 or (
            epoch + 1
        ) == args.max_epochs
        if should_save:
            ckpt = {
                "epoch": epoch,
                "iteration": global_step,
                "model_state_dict": student.state_dict(),
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "optimizer": optimizer.state_dict(),
                "center": teacher.center.detach().cpu(),
                "args": vars(args),
                "config": {
                    "model_name": args.model_name,
                    "out_dim": args.out_dim,
                },
            }
            ckpt_path = out_dir / f"SSL_epoch_{epoch + 1:04d}.pth"
            save_checkpoint(ckpt, ckpt_path)
            shutil.copy(str(ckpt_path), out_dir / "SSL_latest.pth")
            print(f"[Info] Saved checkpoint {ckpt_path}")

            keep = max(0, args.keep_last_checkpoints)
            if keep > 0:
                ckpts = sorted(out_dir.glob("SSL_epoch_*.pth"))
                if len(ckpts) > keep:
                    for old_ckpt in ckpts[:-keep]:
                        old_ckpt.unlink(missing_ok=True)
                        print(f"[Info] Deleted old checkpoint {old_ckpt.name}")

            if compute_embedding_metrics and epoch_logger is not None:
                _compute_and_log_all_metrics(
                    args=args,
                    model=teacher,
                    device=device,
                    epoch=epoch,
                    logs_dir=logs_dir,
                    metrics_logger=epoch_logger,
                    eval_image_size=eval_image_size,
                    force_linear_probe=False,
                )

    instant_logger.plot(
        logs_dir, epoch_logger.path if epoch_logger is not None else None
    )
    print(f"[Info] SSL pretraining complete. Logs: {logs_dir}")


def _freeze_backbone_blocks(model: OTUFormerEncoder, freeze_ratio: float) -> None:
    blocks = getattr(model.backbone, "blocks", None)
    if blocks is None:
        return
    n_blocks = len(blocks)
    n_freeze = int(math.floor(n_blocks * max(0.0, min(1.0, freeze_ratio))))
    for i, block in enumerate(blocks):
        requires_grad = i >= n_freeze
        for p in block.parameters():
            p.requires_grad = requires_grad


def run_finetune(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    _set_cpus(args.cpus)
    device = _resolve_device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_path = (
        Path(getattr(args, "resume", "")) if getattr(args, "resume", "") else None
    )
    if resume_path is not None:
        ckpt_path = resume_path
    else:
        ckpt_path = Path(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path)
    cfg = ckpt.get("config", {})
    model_name = cfg.get("model_name", args.model_name)
    out_dim = cfg.get("out_dim", args.metric_embed_dim)

    model = OTUFormerEncoder(model_name=model_name, out_dim=out_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    _freeze_backbone_blocks(model, args.freeze_ratio)

    ds = MetricDataset(
        csv_path=Path(args.train_data),
        images_dir=Path(args.input_images_dir),
        image_size=224,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    n_classes = len(ds.class_to_idx)
    loss_cls = LOSS_REGISTRY.get(args.loss, ArcFaceLoss)
    loss_fn = loss_cls(embed_dim=out_dim, num_classes=n_classes).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params += list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.finetune_lr)

    start_epoch = 0
    if resume_path is not None:
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "loss_state_dict" in ckpt:
            loss_fn.load_state_dict(ckpt["loss_state_dict"], strict=False)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(
            f"[Info] Resume from {resume_path} at epoch {start_epoch}, iteration {int(ckpt.get('iteration', 0))}"
        )

    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    instant_logger = InstantMetricsLogger(
        logs_dir / "instant_metrics.finetune.csv", mode="finetune"
    )

    compute_embedding_metrics = bool(getattr(args, "compute_embedding_metrics", True))
    epoch_logger = (
        EnhancedMetricsLogger(logs_dir / "metrics.finetune.csv", mode="pretrain")
        if compute_embedding_metrics
        else None
    )

    eval_image_size = (
        int(args.extract_size)
        if int(getattr(args, "extract_size", 0)) > 0
        else _infer_backbone_image_size(model, fallback=224)
    )
    if int(getattr(args, "extract_size", 0)) <= 0:
        print(f"[Info] Auto eval crop size from backbone: {eval_image_size}")

    # Global iteration counter (mirrors ref finetune_arcface `it` variable)
    global_step = int(ckpt.get("iteration", 0))

    for epoch in range(start_epoch, args.finetune_epochs):
        model.train()
        loss_fn.train()
        running = 0.0
        batches = 0

        pbar = tqdm(
            loader,
            desc=f"Finetune Epoch {epoch + 1}/{args.finetune_epochs}",
            ncols=120,
        )
        for step, (imgs, labels) in enumerate(pbar):
            batches += 1
            imgs = imgs.to(device)
            labels = labels.to(device)

            emb = model(imgs)
            loss = loss_fn(emb, labels)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = _compute_grad_norm(model)
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            running += float(loss.item())

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            # Log instant metrics every N steps (mirrors ref per-iteration logging)
            if global_step % max(1, args.log_every_n_steps) == 0:
                with torch.no_grad():
                    instant_metrics = _compute_instant_metrics(model, imgs, device)
                instant_logger.log(
                    iteration=global_step,
                    epoch=epoch,
                    step=step,
                    loss=float(loss.item()),
                    lr=float(optimizer.param_groups[0]["lr"]),
                    grad_norm=float(grad_norm),
                    **instant_metrics,
                )

            global_step += 1

        avg_loss = running / max(batches, 1)
        print(
            f"[Finetune] Epoch {epoch + 1}/{args.finetune_epochs} - Avg Loss: {avg_loss:.4f}"
        )

        # Save checkpoint per --save-every-epochs (mirrors ref arcface_epoch_XXXX.pth)
        should_save = (epoch + 1) % max(1, args.save_every_epochs) == 0 or (
            epoch + 1
        ) == args.finetune_epochs
        if should_save:
            ckpt_payload = {
                "epoch": epoch,
                "iteration": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss_state_dict": loss_fn.state_dict(),
                "config": {
                    "model_name": model_name,
                    "metric_embed_dim": out_dim,
                    "out_dim": out_dim,
                },
            }
            save_path = out_dir / f"finetune_epoch_{epoch + 1:04d}.pth"
            save_checkpoint(ckpt_payload, save_path)
            shutil.copy(str(save_path), out_dir / "finetune_latest.pth")
            print(f"[Info] Saved checkpoint {save_path}")

            keep = max(0, args.keep_last_checkpoints)
            if keep > 0:
                ckpts = sorted(out_dir.glob("finetune_epoch_*.pth"))
                if len(ckpts) > keep:
                    for old_ckpt in ckpts[:-keep]:
                        old_ckpt.unlink(missing_ok=True)
                        print(f"[Info] Deleted old checkpoint {old_ckpt.name}")

        # Compute embedding metrics at save epochs (mirrors ref compute_and_log_all_metrics)
        if should_save and compute_embedding_metrics and epoch_logger is not None:
            _compute_and_log_all_metrics(
                args=args,
                model=model,
                device=device,
                epoch=epoch,
                logs_dir=logs_dir,
                metrics_logger=epoch_logger,
                eval_image_size=eval_image_size,
                force_linear_probe=True,
            )

    instant_logger.plot(
        logs_dir, epoch_logger.path if epoch_logger is not None else None
    )
    print(f"[Info] ArcFace fine-tuning complete. Logs: {logs_dir}")
