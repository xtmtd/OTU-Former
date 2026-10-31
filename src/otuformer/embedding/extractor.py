"""Extract embeddings from images using a trained OTU-Former checkpoint."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from otuformer.training.dataset import (
    _build_recursive_index,
    _resolve_image_path,
    _supports_recursive_lookup,
    build_eval_transform,
)
from otuformer.training.model import OTUFormerEncoder, native_register_count
from otuformer.utils.checkpoint import (
    load_checkpoint,
    load_checkpoint_encoder,
    resolve_checkpoint,
)
from otuformer.utils.device import resolve_device
from otuformer.constants import (
    ATTENTION_POOLING_TYPES,
    TOKEN_MODES,
)
from otuformer.utils.size import (
    resolve_onnx_input_size,
    resolve_training_image_size,
    validate_input_size,
)
from otuformer.data.provenance import sha256_bytes, snapshot_bytes
from otuformer.data.records import DataBundle, FileSnapshot
from otuformer.embedding.record_dataset import (
    RecordImageDataset,
    RecordLabelDataset,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class ExtractionResult:
    """Frame plus the settings and input bytes actually used."""

    frame: pd.DataFrame
    effective: dict[str, Any] = field(default_factory=dict)
    used_files: tuple[FileSnapshot, ...] = ()


def _transform_normalization(transform) -> dict[str, Any]:
    """Read mean/std from the executed eval transform instead of assuming."""
    for step in getattr(transform, "transforms", []):
        mean = getattr(step, "mean", None)
        std = getattr(step, "std", None)
        if mean is not None and std is not None:
            return {
                "mean": [float(value) for value in mean],
                "std": [float(value) for value in std],
            }
    return {"available": False, "reason": "no Normalize step in the eval transform"}


def _check_records_root(records: DataBundle, images_dir: Path) -> None:
    from otuformer.data.paths import require_contained

    root = Path(images_dir).expanduser().resolve()
    if records.images_dir != root:
        raise ValueError(
            "records were built from "
            f"{records.images_dir} but --input-images-dir is {root}"
        )
    for record in records.records:
        # Re-resolve now: validation pinned bytes, not the path a later swap can
        # redirect outside the root.
        require_contained(root, record.access_path, record.image_key)


class ImageFolderDataset(Dataset):
    """Load all images from a directory."""

    def __init__(
        self,
        images_dir: Path,
        extract_size: int = 224,
        eval_transform: str = "center-crop",
    ) -> None:
        self.paths = sorted(
            p
            for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        # Use the same eval transform as pretrain/finetune: direct BICUBIC resize
        self.transform = build_eval_transform(eval_transform, extract_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.paths[idx].name


class LabelCSVImageDataset(Dataset):
    """Load images and labels from a label CSV for attention-query training."""

    def __init__(
        self,
        images_dir: Path,
        label_csv: Path,
        extract_size: int = 224,
        eval_transform: str = "center-crop",
    ) -> None:
        self.transform = build_eval_transform(eval_transform, extract_size)
        label_df = pd.read_csv(label_csv)
        if "image" not in label_df.columns or "label" not in label_df.columns:
            raise ValueError("--label-csv must contain 'image' and 'label' columns")

        image_paths = [
            p
            for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        by_name: dict[str, Path] = {}
        for p in image_paths:
            by_name.setdefault(p.name, p)

        labels = sorted(label_df["label"].astype(str).unique().tolist())
        label_to_idx = {lab: i for i, lab in enumerate(labels)}

        samples: list[tuple[Path, int]] = []
        for _, row in label_df.iterrows():
            image_name = str(row["image"])
            label = str(row["label"])
            direct_path = (images_dir / image_name).expanduser()
            if direct_path.exists() and direct_path.is_file():
                p = direct_path
            else:
                p = by_name.get(Path(image_name).name)
            if p is None:
                continue
            samples.append((p, label_to_idx[label]))

        if not samples:
            raise ValueError(
                "No image rows from --label-csv matched files under --input-images-dir"
            )
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label_idx = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label_idx


class CSVImageDataset(Dataset):
    """Load extraction images in CSV order, preserving image id strings."""

    def __init__(
        self,
        images_dir: Path,
        csv_path: Path,
        extract_size: int = 224,
        eval_transform: str = "center-crop",
    ) -> None:
        self.transform = build_eval_transform(eval_transform, extract_size)
        df = pd.read_csv(csv_path)
        if "image" not in df.columns:
            raise ValueError("CSV must contain an 'image' column")

        refs = [str(x) for x in df["image"].tolist()]
        missing_direct = [
            ref
            for ref in refs
            if _supports_recursive_lookup(ref) and not (images_dir / Path(ref)).exists()
        ]
        by_relative, by_name = ({}, {})
        if missing_direct:
            by_relative, by_name = _build_recursive_index(images_dir)

        paths = [
            _resolve_image_path(images_dir, ref, by_relative, by_name) for ref in refs
        ]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            sample = ", ".join(missing[:3])
            raise FileNotFoundError(
                f"Some CSV image paths do not exist under {images_dir}: {sample}"
            )

        self.paths = paths
        self.ids = refs
        self.samples: list[str] | None = None

        if "sample" in df.columns:
            self.samples = df["sample"].astype(str).tolist()
        else:
            inferred: list[str | None] = []
            for path in self.paths:
                try:
                    rel = path.relative_to(images_dir)
                except ValueError:
                    inferred.append(None)
                    continue
                if len(rel.parts) >= 2:
                    inferred.append(rel.parts[0])
                else:
                    inferred.append(None)
            if any(v is not None for v in inferred):
                self.samples = [v if v is not None else "" for v in inferred]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.samples is None:
            return self.transform(img), self.ids[idx]
        return self.transform(img), self.ids[idx], self.samples[idx]


class LightweightAttentionPooling(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, dim))
        self.scale = dim**-0.5

    def forward(self, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = patch_tokens.shape[0]
        query = self.query.unsqueeze(0).expand(bsz, -1, -1)
        attn_scores = torch.bmm(query, patch_tokens.transpose(1, 2))
        attn_scores = attn_scores * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        pooled = torch.bmm(attn_weights, patch_tokens)
        return pooled.squeeze(1), attn_weights.squeeze(1)


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, freeze_attn: bool = True) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        if freeze_attn:
            for p in self.attn.parameters():
                p.requires_grad = False

    def forward(self, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = patch_tokens.shape[0]
        query = self.query.expand(bsz, -1, -1)
        out, weights = self.attn(query, patch_tokens, patch_tokens, need_weights=True)
        return out.squeeze(1), weights.squeeze(1)


class GatedAttentionPooling(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.attention_v = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a_v = self.attention_v(patch_tokens)
        a_u = self.attention_u(patch_tokens)
        attn_logits = self.attention_w(a_v * a_u)
        attn_weights = F.softmax(attn_logits, dim=1)
        pooled = torch.sum(attn_weights * patch_tokens, dim=1)
        return pooled, attn_weights.squeeze(-1)


def _build_attention_pooling(pooling_type: str, dim: int) -> nn.Module:
    if pooling_type == "lightweight":
        return LightweightAttentionPooling(dim)
    if pooling_type == "multihead":
        return MultiHeadAttentionPooling(dim, freeze_attn=True)
    if pooling_type == "gated":
        return GatedAttentionPooling(dim, hidden_dim=128)
    raise ValueError(
        f"Unsupported --attention-pooling-type '{pooling_type}', choose from: lightweight, multihead, gated"
    )


def _attention_pooling_checkpoint_path(
    checkpoint_path: Path, pooling_type: str
) -> Path:
    return checkpoint_path.parent / f"{checkpoint_path.stem}.{pooling_type}_pooling.pth"


# Patch policy recorded in every newly trained attention-pool checkpoint. An
# older pool that records none of the patch keys predates the policy, so its
# patch set is unknown and it is reusable only for a one-prefix, zero-register
# backbone (the legacy one-CLS/no-register pool).
_POOL_POLICY_KEYS = (
    "num_prefix_tokens",
    "register_tokens",
    "prefix_excluded",
    "attention_pooling_type",
)
_PATCH_POLICY_KEYS = ("num_prefix_tokens", "register_tokens", "prefix_excluded")


def _attention_pool_policy(model: OTUFormerEncoder, pooling_type: str) -> dict:
    """The patch set an attention pool was trained on."""
    backbone = model.backbone
    return {
        "num_prefix_tokens": int(getattr(backbone, "num_prefix_tokens", 1)),
        "register_tokens": native_register_count(backbone),
        "prefix_excluded": True,
        "attention_pooling_type": pooling_type,
    }


def _tagged_pooling_checkpoint_path(path: Path, policy: dict) -> Path:
    """Policy-specific sibling, e.g. ``<stem>.prefix5-reg4-lightweight_pooling.pth``.

    A pool path already ends in ``.<type>_pooling.pth``; that suffix is replaced
    so the tag stays directly attached to the checkpoint stem.
    """
    pooling_type = policy["attention_pooling_type"]
    suffix = f".{pooling_type}_pooling.pth"
    stem = path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem
    tag = f"prefix{policy['num_prefix_tokens']}-reg{policy['register_tokens']}"
    return path.parent / f"{stem}.{tag}-{pooling_type}_pooling.pth"


def _attention_pooling_checkpoint_candidates(
    checkpoint_path: Path,
    pooling_type: str,
    target_checkpoint_path: Path | None,
    policy: dict | None = None,
) -> list[Path]:
    """Tagged sibling first, then the requested/default legacy path."""
    legacy = _attention_pooling_checkpoint_path(checkpoint_path, pooling_type)
    if policy is None:
        if target_checkpoint_path is None:
            return [legacy]
        target = Path(target_checkpoint_path)
        if target == legacy:
            return [target]
        return [target, legacy]
    bases: list[Path] = []
    if target_checkpoint_path is not None:
        bases.append(Path(target_checkpoint_path))
    if legacy not in bases:
        bases.append(legacy)
    ordered: list[Path] = []
    for base in bases:
        for candidate in (_tagged_pooling_checkpoint_path(base, policy), base):
            if candidate not in ordered:
                ordered.append(candidate)
    return ordered


def _pool_policy_compatible(payload: object, policy: dict) -> tuple[bool, str]:
    """Whether a stored attention-pool checkpoint matches this patch set.

    A file that records none of the patch keys predates the policy, so its
    patch set is unknown and it is reusable only for the legacy one-prefix,
    zero-register backbone. Any policy field that *is* recorded must match --
    including ``attention_pooling_type`` and a half-written patch policy -- so a
    renamed or partially recorded file is never silently reused just because its
    filename matches.
    """
    args = payload.get("args") if isinstance(payload, dict) else None
    if not isinstance(args, dict):
        args = {}
    differs = [
        key
        for key in _POOL_POLICY_KEYS
        if key in args and args.get(key) != policy[key]
    ]
    if differs:
        return False, "records a different " + ", ".join(sorted(differs))
    missing = [key for key in _POOL_POLICY_KEYS if key not in args]
    if not missing:
        return True, ""
    if not any(key in args for key in _PATCH_POLICY_KEYS):
        # Pre-v0.10.0 format: no patch policy at all. Its patch set is unknown,
        # so it is reusable only for the one-prefix, zero-register backbone.
        if policy["num_prefix_tokens"] == 1 and policy["register_tokens"] == 0:
            return True, ""
        return False, "has no recorded patch policy"
    # A partially recorded policy cannot be verified -- e.g. the patch keys are
    # present but the pooling type is not -- so reject it instead of trusting a
    # matching filename. Otherwise a ``teacher``-format payload would load with
    # ``strict=False`` and leave the pool randomly initialized.
    return False, (
        "has an incomplete patch policy (missing " + ", ".join(sorted(missing)) + ")"
    )


def _patch_features(backbone: torch.nn.Module, feats: torch.Tensor) -> torch.Tensor:
    """Drop every prefix token (CLS and registers) from backbone features.

    Registers are never image patches, so every patch consumer starts at the
    backbone's own ``num_prefix_tokens``; a CLS-free backbone keeps its first
    patch.
    """
    prefix = int(getattr(backbone, "num_prefix_tokens", 1))
    return feats[:, prefix:, :]


def _iter_trainable_params(attention_pool: nn.Module) -> Iterable[nn.Parameter]:
    if hasattr(attention_pool, "query"):
        query = getattr(attention_pool, "query")
        if isinstance(query, nn.Parameter):
            return [query]
    return list(attention_pool.parameters())


def _finetune_attention_query(
    model: OTUFormerEncoder,
    attention_pool: nn.Module,
    images_dir: Path,
    label_csv: Path | None,
    extract_size: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    num_epochs: int,
    eval_transform: str = "center-crop",
    *,
    dataset: RecordLabelDataset | None = None,
) -> None:
    if dataset is None:
        if label_csv is None:
            raise ValueError(
                "attention-query training needs a labelled CSV or data records"
            )
        dataset = LabelCSVImageDataset(
            images_dir=images_dir,
            label_csv=label_csv,
            extract_size=extract_size,
            eval_transform=eval_transform,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("Need at least one full batch for attention query training")

    for p in model.parameters():
        p.requires_grad = False

    for p in attention_pool.parameters():
        p.requires_grad = False
    trainable_params = list(_iter_trainable_params(attention_pool))
    for p in trainable_params:
        p.requires_grad = True

    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
    temperature = 0.07

    model.train()
    attention_pool.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_knn = 0.0
        num_steps = 0
        pbar = tqdm(
            loader, desc=f"Query Finetune [{epoch + 1}/{num_epochs}]", ncols=120
        )
        for imgs, label_idx in pbar:
            imgs = imgs.to(device)
            label_idx = label_idx.to(device)

            with torch.no_grad():
                feats = model.backbone.forward_features(imgs)
                if isinstance(feats, dict):
                    feats = feats["x"]
                patch_tokens = _patch_features(model.backbone, feats)

            pooled, _ = attention_pool(patch_tokens)
            pooled = F.normalize(pooled, dim=-1)
            sim_matrix = pooled @ pooled.T / temperature

            pos_mask = (label_idx.unsqueeze(0) == label_idx.unsqueeze(1)).float()
            pos_mask.fill_diagonal_(0)
            neg_mask = 1.0 - pos_mask
            neg_mask.fill_diagonal_(0)

            exp_sim = torch.exp(sim_matrix)
            pos_sim = (sim_matrix * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
            neg_sim = torch.log((exp_sim * neg_mask).sum(dim=1) + 1e-8)
            loss = (-pos_sim + neg_sim).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

            with torch.no_grad():
                sim_np = sim_matrix.detach().cpu().numpy()
                np.fill_diagonal(sim_np, -np.inf)
                nearest_idx = sim_np.argmax(axis=1)
                knn_acc = (
                    label_idx[nearest_idx] == label_idx
                ).float().mean().item() * 100.0
            running_knn += float(knn_acc)
            num_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / max(num_steps, 1)
        avg_knn = running_knn / max(num_steps, 1)
        print(
            f"[Info] Attention query epoch {epoch + 1}/{num_epochs} loss: {avg_loss:.4f}, "
            f"kNN-Acc(k=1): {avg_knn:.2f}%"
        )

    model.eval()
    attention_pool.eval()


def _frame_dim(frame) -> int | None:
    """Feature width actually emitted, read from the frame itself."""
    dimensions = [column for column in frame.columns if column.startswith("dim_")]
    return len(dimensions) or None


def _record_effective_k(effective: dict[str, Any] | None, k: int) -> None:
    """Record the patch count the top-K branch actually used."""
    if effective is None:
        return
    chain = effective.setdefault("patch_topk_effective_k", [])
    if k not in chain:
        chain.append(int(k))


def _projector_normalizes(model: torch.nn.Module) -> bool:
    """Whether the loaded projector head ends in a normalization step."""
    projector = getattr(model, "projector", None)
    if projector is None:
        return False
    code = getattr(type(projector).forward, "__code__", None)
    return bool(code is not None and "normalize" in code.co_names)


def _probe_backbone_norm(backbone) -> dict:
    """Watch whether the backbone's own norm layers run during this extraction.

    Recorded from forward hooks on the modules themselves: "extraction adds no
    normalization" and "the backbone did not normalize" are different claims, and
    only the hook can tell them apart. No-op placeholders (``nn.Identity``) are not
    norm layers and are not reported as one.
    """
    state: dict[str, Any] = {"layers": [], "fired": []}
    if backbone is None:
        return state
    for name in ("fc_norm", "norm"):
        module = getattr(backbone, name, None)
        if not isinstance(module, torch.nn.Module) or isinstance(
            module, torch.nn.Identity
        ):
            continue
        label = f"backbone.{name} ({type(module).__name__})"
        state["layers"].append(label)

        def mark(*_args, label=label):
            if label not in state["fired"]:
                state["fired"].append(label)

        module.register_forward_hook(mark)
    return state


def _backbone_norm_record(state: dict | None) -> dict[str, Any]:
    if state is None:
        return {
            "status": "unavailable",
            "reason": "this backend does not expose a backbone normalization contract",
        }
    if not state["layers"]:
        return {
            "status": "not_applied",
            "detail": "the backbone exposes no final norm module",
        }
    if state["fired"]:
        return {
            "status": "applied",
            "stage": "inside backbone forward_features",
            "layers": list(state["fired"]),
            "detail": "extraction adds no further normalization after forward_features",
        }
    return {
        "status": "not_applied",
        "layers": list(state["layers"]),
        "detail": "the backbone's norm module did not run on this path",
    }


def _record_feature_chain(
    effective: dict[str, Any],
    *,
    token_mode: str,
    use_projector_output: bool,
    model: torch.nn.Module | None,
    output_dim: int | None,
    input_dim: int | None = None,
) -> None:
    """Record the feature path that actually executed, not the flags requested."""
    # patch-topk and attention-pool never consume the projector, whatever the flag.
    projector_used = token_mode == "cls" and use_projector_output
    if projector_used:
        ops = ["projector"]
    elif token_mode == "cls":
        ops = ["cls_token_index_0"]
    elif token_mode == "patch-topk":
        ops = ["topk_by_patch_l2_norm", "flatten"]
        policy = effective.get("patch_topk_pca") or {}
        # PCA is skipped for tiny selections, so its steps are only recorded when
        # they ran.
        if policy.get("mode") == "fit_transform":
            ops = [*ops, "standard_scaler", "pca"]
        elif policy.get("mode") == "truncate_first_dims":
            ops = [*ops, "truncate_first_dims"]
    else:
        ops = ["attention_pool_forward"]

    if projector_used and model is not None:
        normalizes = _projector_normalizes(model)
        effective["vector_normalization"] = (
            {
                "status": "applied",
                "stage": "projector_head_l2",
                "detail": "the projector head normalizes its own output",
            }
            if normalizes
            else {"status": "none", "detail": "the projector head output is unnormalized"}
        )
        if normalizes:
            ops = [*ops, "l2_normalize"]
    else:
        effective["vector_normalization"] = {
            "status": "none",
            "detail": "no post-feature normalization runs after this token mode",
        }

    chain: dict[str, Any] = {"token_mode": token_mode, "ops": ops}
    if input_dim is not None:
        chain["input_dim"] = int(input_dim)
    if output_dim is not None:
        chain["output_dim"] = int(output_dim)
    # ``post_feature_ops`` is the flat operation list the records export has always
    # carried; ``feature_chain`` is the same chain plus the dimensions.
    effective["post_feature_ops"] = list(ops)
    effective["feature_chain"] = chain


def _feature_input_dim(effective: dict[str, Any], model: torch.nn.Module | None) -> int | None:
    """Width of the tensor the recorded operations consumed."""
    policy = effective.get("patch_topk_pca") or {}
    if policy.get("input_dim") is not None:
        return int(policy["input_dim"])
    backbone = getattr(model, "backbone", None)
    width = getattr(backbone, "num_features", None)
    return int(width) if isinstance(width, int) else None


def _apply_patch_topk_pca(
    embeddings: np.ndarray,
    topk_patches: int,
    effective: dict[str, Any] | None = None,
) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n_samples, dim = embeddings.shape
    preferred_dim = {10: 256, 20: 512, 30: 1024}.get(topk_patches, 512)
    policy: dict[str, Any] = {
        "topk_patches": topk_patches,
        "preferred_dim": preferred_dim,
        "input_dim": int(dim),
        "samples": int(n_samples),
        "seed": 42,
    }
    if effective is not None:
        effective["patch_topk_pca"] = policy
    if n_samples < 3:
        target_dim = min(preferred_dim, dim)
        policy["mode"] = "truncate_first_dims" if 0 < target_dim < dim else "none"
        policy["reason"] = "fewer than three samples"
        policy["output_dim"] = int(min(target_dim, dim) if 0 < target_dim < dim else dim)
        if 0 < target_dim < dim:
            return embeddings[:, :target_dim]
        return embeddings

    max_possible_dim = min(n_samples - 1, dim)
    target_dim = min(preferred_dim, max_possible_dim)
    if target_dim <= 0 or target_dim >= dim:
        policy["mode"] = "none"
        policy["output_dim"] = int(dim)
        return embeddings
    policy.update(
        {
            "mode": "fit_transform",
            "fit_scope": "all extracted rows in this run",
            "scaler": "standard",
            "explained_variance": None,
        }
    )

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=target_dim, random_state=42)
    reduced = pca.fit_transform(embeddings_scaled)
    explained = float(pca.explained_variance_ratio_.sum())
    policy["output_dim"] = int(target_dim)
    policy["explained_variance"] = explained
    # sklearn resolves "auto" privately; record the solver that actually ran rather
    # than the request.
    policy["solver"] = str(
        getattr(pca, "_fit_svd_solver", None) or getattr(pca, "svd_solver", "unknown")
    )
    print(f"[Info] PCA: {dim}D -> {target_dim}D (explained variance: {explained:.2%})")
    return reduced


def _apply_patch_topk_pca_to_df(
    df: pd.DataFrame, topk_patches: int, effective: dict[str, Any] | None = None
) -> pd.DataFrame:
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    if not dim_cols:
        return df
    reduced = _apply_patch_topk_pca(
        df[dim_cols].to_numpy(), topk_patches=topk_patches, effective=effective
    )
    reduced_cols = [f"dim_{i}" for i in range(reduced.shape[1])]
    fixed_cols = [c for c in df.columns if not c.startswith("dim_")]
    out = df[fixed_cols].copy()
    reduced_df = pd.DataFrame(reduced, columns=reduced_cols, index=out.index)
    return pd.concat([out, reduced_df], axis=1)


def detect_batch_mode(images_dir: Path) -> bool:
    """True if images_dir contains subdirectories with images."""
    subdirs = [p for p in images_dir.iterdir() if p.is_dir()]
    for sub in subdirs:
        if any(
            p.suffix.lower() in IMAGE_EXTENSIONS for p in sub.iterdir() if p.is_file()
        ):
            return True
    return False


def _load_model(
    checkpoint_path: Path,
    model_name: str,
    device: torch.device,
    use_student: bool = False,
    *,
    payload: dict[str, Any] | None = None,
    effective: dict[str, Any] | None = None,
) -> tuple[OTUFormerEncoder, int]:
    """Load model from checkpoint and return ``(model, checkpoint_size)``.

    The encoder is constructed with ``img_size=checkpoint_size`` so the
    positional embeddings load exactly; the transform may run at a different
    ``requested_input_size`` via ``dynamic_img_size=True`` interpolation.

    For SSL pretrain checkpoints the teacher is the EMA-averaged model and
    produces better embeddings for downstream analysis (consistent with ref
    script and DINO/iBOT convention).  Pass ``use_student=True`` to load the
    student weights instead.

    Weight-key priority is owned by
    :func:`otuformer.utils.checkpoint.resolve_checkpoint_state_dict`.

    Ref-script checkpoints (``model``/``teacher``/``student`` weight keys,
    ``projector.<i>`` projector names, ``args`` instead of ``config``) are
    supported for reading; see :mod:`otuformer.utils.checkpoint`.
    """
    ckpt = payload if payload is not None else load_checkpoint(checkpoint_path)
    architecture = resolve_checkpoint(ckpt, model_name, prefer_student=use_student)
    checkpoint_size = resolve_training_image_size(ckpt)
    model, architecture = load_checkpoint_encoder(
        ckpt, architecture, image_size=checkpoint_size
    )
    validate_input_size(checkpoint_size, model, architecture.model_name)
    model.eval().to(device)
    print(f"[Info] Loaded weights from checkpoint key: '{architecture.source_key}'")
    if effective is not None:
        effective["checkpoint"] = {
            "source_key": architecture.source_key,
            "model_name": architecture.model_name,
            "model_name_from_metadata": architecture.model_name_from_metadata,
            "embedding_head": architecture.embedding_head,
            "embedding_dim": architecture.embedding_dim,
            "checkpoint_image_size": checkpoint_size,
            "register_tokens": {
                "actual": architecture.actual_register_tokens,
                "added": architecture.added_register_tokens,
            },
            "requested_student": bool(use_student),
        }
    return model, checkpoint_size


def _extract_one_dir(
    model: torch.nn.Module,
    images_dir: Path,
    extract_size: int,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
    use_projector_output: bool = False,
    token_mode: str = "cls",
    topk_patches: int = 20,
    apply_patch_topk_pca: bool = True,
    eval_transform: str = "center-crop",
    effective: dict[str, Any] | None = None,
) -> pd.DataFrame:
    from otuformer.training.model import OTUFormerEncoder

    ds = ImageFolderDataset(images_dir, extract_size, eval_transform=eval_transform)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    all_ids: list[str] = []
    all_embs: list[np.ndarray] = []
    desc = f"Extract {images_dir.name}" if images_dir.name else "Extract"
    with torch.no_grad():
        for imgs, names in tqdm(loader, desc=desc, ncols=120):
            imgs = imgs.to(device)
            if isinstance(model, OTUFormerEncoder):
                feats = model.backbone.forward_features(imgs)
                if isinstance(feats, dict):
                    feats = feats["x"]
            else:
                feats = None

            if token_mode == "patch-topk":
                if feats is None:
                    raise ValueError(
                        "token-mode 'patch-topk' requires OTUFormerEncoder backbone features"
                    )
                patch_tokens = _patch_features(model.backbone, feats)
                num_patches = patch_tokens.shape[1]
                k = min(topk_patches, num_patches)
                _record_effective_k(effective, k)
                if k <= 0:
                    raise ValueError("--topk-patches must be >= 1")
                patch_norms = patch_tokens.norm(dim=-1)
                _, topk_indices = torch.topk(patch_norms, k=k, dim=1)
                topk_indices = topk_indices.unsqueeze(-1).expand(
                    -1, -1, patch_tokens.shape[-1]
                )
                topk_tokens = torch.gather(patch_tokens, dim=1, index=topk_indices)
                embs = topk_tokens.reshape(topk_tokens.shape[0], -1)
            elif token_mode == "attention-pool":
                if feats is None:
                    raise ValueError(
                        "token-mode 'attention-pool' requires OTUFormerEncoder backbone features"
                    )
                patch_tokens = _patch_features(model.backbone, feats)
                if not hasattr(model, "attention_pool"):
                    raise ValueError(
                        "attention-pool is not initialized; provide --label-csv so query can be trained"
                    )
                pooled, _ = model.attention_pool(patch_tokens)
                embs = pooled
            elif use_projector_output:
                # Projector output: SSL output or the task embedding for fine-tuned checkpoints.
                embs = model(imgs)
                if isinstance(embs, tuple):
                    embs = embs[0]
            else:
                # default: raw CLS token from backbone (matches ref default behaviour)
                if feats is not None:
                    embs = feats[:, 0]
                else:
                    embs = model(imgs)
                    if isinstance(embs, tuple):
                        embs = embs[0]
            embs = embs.cpu().numpy()
            all_ids.extend(names)
            all_embs.append(embs)
    if len(all_embs) == 0:
        return pd.DataFrame(columns=["id"])
    embs_array = np.concatenate(all_embs, axis=0)
    if token_mode == "patch-topk" and apply_patch_topk_pca:
        print(f"[Info] Applying PCA reduction for Top-{topk_patches} patches...")
        embs_array = _apply_patch_topk_pca(embs_array, topk_patches=topk_patches)
    dim_cols = [f"dim_{i}" for i in range(embs_array.shape[1])]
    df = pd.DataFrame(embs_array, columns=dim_cols)
    df.insert(0, "id", all_ids)
    return df


def _extract_one_csv(
    model: torch.nn.Module,
    images_dir: Path,
    csv_path: Path | None,
    extract_size: int,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
    use_projector_output: bool = False,
    token_mode: str = "cls",
    topk_patches: int = 20,
    eval_transform: str = "center-crop",
    *,
    dataset: RecordImageDataset | None = None,
    effective: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if dataset is not None:
        if csv_path is not None:
            raise ValueError("pass either a CSV path or a prebuilt dataset, not both")
        ds = dataset
    else:
        if csv_path is None:
            raise ValueError("a CSV path or a prebuilt dataset is required")
        ds = CSVImageDataset(
            images_dir=images_dir,
            csv_path=csv_path,
            extract_size=extract_size,
            eval_transform=eval_transform,
        )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    all_ids: list[str] = []
    all_samples: list[str] | None = [] if ds.samples is not None else None
    all_embs: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extract CSV", ncols=120):
            if len(batch) == 3:
                imgs, names, samples = batch
            else:
                imgs, names = batch
                samples = None
            imgs = imgs.to(device)
            feats = (
                model.backbone.forward_features(imgs)
                if isinstance(model, OTUFormerEncoder)
                else None
            )
            if isinstance(feats, dict):
                feats = feats["x"]

            if token_mode == "patch-topk":
                if feats is None:
                    raise ValueError(
                        "token-mode 'patch-topk' requires OTUFormerEncoder backbone features"
                    )
                patch_tokens = _patch_features(model.backbone, feats)
                num_patches = patch_tokens.shape[1]
                k = min(topk_patches, num_patches)
                _record_effective_k(effective, k)
                patch_norms = patch_tokens.norm(dim=-1)
                _, topk_indices = torch.topk(patch_norms, k=k, dim=1)
                topk_indices = topk_indices.unsqueeze(-1).expand(
                    -1, -1, patch_tokens.shape[-1]
                )
                topk_tokens = torch.gather(patch_tokens, dim=1, index=topk_indices)
                embs = topk_tokens.reshape(topk_tokens.shape[0], -1)
            elif token_mode == "attention-pool":
                if feats is None or not hasattr(model, "attention_pool"):
                    raise ValueError("attention-pool is not initialized")
                patch_tokens = _patch_features(model.backbone, feats)
                pooled, _ = model.attention_pool(patch_tokens)
                embs = pooled
            elif use_projector_output:
                embs = model(imgs)
                if isinstance(embs, tuple):
                    embs = embs[0]
            else:
                if feats is not None:
                    embs = feats[:, 0]
                else:
                    embs = model(imgs)
                    if isinstance(embs, tuple):
                        embs = embs[0]

            all_ids.extend(list(names))
            if all_samples is not None and samples is not None:
                all_samples.extend([str(v) for v in list(samples)])
            all_embs.append(embs.cpu().numpy())

    if not all_embs:
        return pd.DataFrame(columns=["id"])
    arr = np.concatenate(all_embs, axis=0)
    if token_mode == "patch-topk":
        print(f"[Info] Applying PCA reduction for Top-{topk_patches} patches...")
        arr = _apply_patch_topk_pca(
            arr, topk_patches=topk_patches, effective=effective
        )
    cols = [f"dim_{i}" for i in range(arr.shape[1])]
    out = pd.DataFrame(arr, columns=cols)
    out.insert(0, "id", all_ids)
    if all_samples is not None:
        out.insert(1, "sample", all_samples)
    return out


def _iter_external_tensors(proto):
    """Every TensorProto in the model, including those held in node attributes.

    A ``Constant`` node keeps its payload in an attribute tensor, and attributes can
    nest subgraphs, so walking only ``graph.initializer`` misses legal external data.
    """
    import onnx

    def walk_graph(graph):
        for tensor in graph.initializer:
            yield tensor
        for sparse in getattr(graph, "sparse_initializer", ()):
            yield sparse.values
            yield sparse.indices
        for node in graph.node:
            for attribute in node.attribute:
                yield from walk_attribute(attribute)

    def walk_attribute(attribute):
        if attribute.type == onnx.AttributeProto.TENSOR:
            yield attribute.t
        elif attribute.type == onnx.AttributeProto.TENSORS:
            yield from attribute.tensors
        elif attribute.type == onnx.AttributeProto.GRAPH:
            yield from walk_graph(attribute.g)
        elif attribute.type == onnx.AttributeProto.GRAPHS:
            for graph in attribute.graphs:
                yield from walk_graph(graph)
        elif attribute.type in (
            onnx.AttributeProto.SPARSE_TENSOR,
            onnx.AttributeProto.SPARSE_TENSORS,
        ):
            # A sparse payload has two tensors of its own, each of which can be
            # stored externally. Covering sparse graph initializers says nothing
            # about sparse attributes.
            sparse = (
                [attribute.sparse_tensor]
                if attribute.type == onnx.AttributeProto.SPARSE_TENSOR
                else attribute.sparse_tensors
            )
            for entry in sparse:
                yield entry.values
                yield entry.indices

    yield from walk_graph(proto.graph)
    for function in getattr(proto, "functions", ()):
        # A FunctionProto carries default values for its own attributes, and a node
        # inside the body reaches them through ``ref_attr_name`` rather than holding
        # the payload itself. Walking only the body nodes misses that data entirely.
        for attribute in getattr(function, "attribute_proto", ()):
            yield from walk_attribute(attribute)
        for node in function.node:
            for attribute in node.attribute:
                yield from walk_attribute(attribute)


def _onnx_graph_path(onnx_path: Path) -> Path:
    """The graph path as it is handed to the runtime: absolute, *not* resolved.

    onnxruntime resolves an external-data ``location`` against the path it was
    given. Resolving first would make the file names we hash and record differ
    from the file names it opens, so the entry spelling is the one that governs
    every step (enumeration, session, hashing, records).
    """
    graph = Path(onnx_path).expanduser()
    return graph if graph.is_absolute() else Path.cwd() / graph


def _external_entry(root: Path, location: str) -> Path:
    """The access path the runtime uses for one external-data ``location``.

    The runtime opens ``root / location`` unchanged and lets the kernel resolve it,
    so that expression -- not a folded or pre-resolved spelling -- is what must be
    hashed, checked and recorded. Resolving it first would bind the hash to whatever
    the link pointed at then, and a later, permanent redirection of an intermediate
    link would silently load different bytes.

    Containment is still judged on the resolved target, so a link that escapes the
    graph directory is refused.
    """
    parts = PurePosixPath(location).parts
    if not parts:
        raise ValueError("external tensor location is empty")
    return root.joinpath(*parts)


def _onnx_external_files(onnx_path: Path) -> list[Path]:
    """External tensor files referenced anywhere in the model, next to its path.

    The same enumeration drives input protection, the session loading mode and the
    hash checks, so a tensor cannot be visible to one and invisible to another. The
    returned paths keep the graph's own directory spelling; only the containment
    check looks through symlinks, so a payload cannot be smuggled in from outside.

    A location that resolves outside the graph directory is refused here as well as
    by the runtime, so enumeration, loading and the protection list agree.
    """
    import onnx

    graph = _onnx_graph_path(onnx_path)
    proto = onnx.load(str(graph), load_external_data=False)
    root = graph.parent
    root_resolved = root.resolve()
    found: list[Path] = []
    for tensor in _iter_external_tensors(proto):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        locations = [
            entry.value for entry in tensor.external_data if entry.key == "location"
        ]
        if len(locations) != 1:
            # Readers disagree on which duplicate wins (onnx takes the first,
            # onnxruntime the last), so one of them would read a file the others
            # never hashed. Refuse rather than pick a side.
            raise ValueError(
                f"external tensor {tensor.name!r} declares {len(locations)} 'location' "
                "entries; exactly one is required so every step reads the same file"
            )
        location = locations[0]
        candidate = _external_entry(root, location)
        resolved = candidate.resolve()
        if resolved != root_resolved and root_resolved not in resolved.parents:
            raise ValueError(
                f"external tensor {tensor.name!r} resolves outside the graph directory"
            )
        if not candidate.is_file():
            raise FileNotFoundError(f"ONNX external tensor file is missing: {candidate}")
        found.append(candidate)
    return sorted(set(found))


def _require_unchanged(snapshots: list[FileSnapshot]) -> None:
    for snapshot in snapshots:
        data = snapshot.path.read_bytes()
        if len(data) != snapshot.size_bytes or sha256_bytes(data) != snapshot.sha256:
            raise RuntimeError(
                f"ONNX input changed while the session was loading: {snapshot.path}"
            )


def _extract_with_onnx(
    onnx_path: Path,
    images_dir: Path,
    extract_size: int,
    batch_size: int,
    num_workers: int,
    extract_csv: Path | None,
    eval_transform: str = "center-crop",
    *,
    records: DataBundle | None = None,
    dataset: RecordImageDataset | None = None,
    effective: dict[str, Any] | None = None,
    used_files: list[FileSnapshot] | None = None,
) -> pd.DataFrame:
    import onnxruntime as ort

    # One path governs loading, hashing and records: the entry spelling.
    graph = _onnx_graph_path(onnx_path)
    graph_bytes: bytes | None = None
    graph_snapshot: FileSnapshot | None = None
    external_inputs: list[FileSnapshot] = []
    external_paths: list[Path] = []
    if records is not None:
        graph_bytes, graph_snapshot = snapshot_bytes(graph, follow_symlink=False)
        if used_files is not None:
            used_files.append(graph_snapshot)
        external_paths = _onnx_external_files(graph)
        for external in external_paths:
            _data, external_snapshot = snapshot_bytes(external, follow_symlink=False)
            external_inputs.append(external_snapshot)
            if used_files is not None:
                used_files.append(external_snapshot)
    # An external-data graph can only be opened from its original path; the
    # hashes around session construction bound the window rather than closing it
    # (a concurrent change-and-restore is not detectable). The graph is part of
    # that check, not only its tensors.
    pinned = [*([graph_snapshot] if graph_snapshot is not None else []), *external_inputs]
    if pinned:
        _require_unchanged(pinned)
    if graph_bytes is not None and not external_inputs:
        session = ort.InferenceSession(graph_bytes, providers=["CPUExecutionProvider"])
    else:
        session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    if pinned:
        _require_unchanged(pinned)
    # The ONNX graph runs on its own provider, not on a torch device.
    print(f"Resolved device: cpu (onnxruntime {session.get_providers()[0]})")
    input_name = session.get_inputs()[0].name
    if effective is not None:
        output = session.get_outputs()[0]
        effective["onnx"] = {
            "provider": session.get_providers()[0],
            "input_name": input_name,
            "output_name": output.name,
            "output_shape": list(output.shape),
            "graph_file": str(graph),
            "external_tensor_files": [str(path) for path in external_paths],
            "hash_window": (
                "graph and external tensors are re-hashed before and after session "
                "construction"
                if pinned
                else "not_applicable: this call is not record-backed"
            ),
            "undetectable_change": (
                "a concurrent change-and-restore inside that window is not detectable"
                if pinned
                else None
            ),
            "internal_feature_contract": {
                "status": "unavailable",
                "reason": "an ONNX graph does not expose an internal layer/norm contract",
            },
        }

    if extract_csv is not None or dataset is not None:
        return _extract_one_csv_onnx(
            session=session,
            input_name=input_name,
            images_dir=images_dir,
            csv_path=extract_csv,
            extract_size=extract_size,
            batch_size=batch_size,
            num_workers=num_workers,
            eval_transform=eval_transform,
            dataset=dataset,
        )

    if detect_batch_mode(images_dir):
        subdirs = sorted(p for p in images_dir.iterdir() if p.is_dir())
        frames = []
        for sub in tqdm(subdirs, desc="Batch dirs", ncols=120):
            df = _extract_one_dir_onnx(
                session=session,
                input_name=input_name,
                images_dir=sub,
                extract_size=extract_size,
                batch_size=batch_size,
                num_workers=num_workers,
                eval_transform=eval_transform,
            )
            if len(df) == 0:
                continue
            df.insert(1, "sample", sub.name)
            frames.append(df)
        if len(frames) == 0:
            return pd.DataFrame(columns=["id", "sample"])
        return pd.concat(frames, ignore_index=True)

    return _extract_one_dir_onnx(
        session=session,
        input_name=input_name,
        images_dir=images_dir,
        extract_size=extract_size,
        batch_size=batch_size,
        num_workers=num_workers,
        eval_transform=eval_transform,
    )


def _extract_one_dir_onnx(
    session,
    input_name: str,
    images_dir: Path,
    extract_size: int,
    batch_size: int,
    num_workers: int,
    eval_transform: str = "center-crop",
) -> pd.DataFrame:
    ds = ImageFolderDataset(images_dir, extract_size, eval_transform=eval_transform)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    all_ids: list[str] = []
    all_embs: list[np.ndarray] = []
    desc = f"Extract {images_dir.name}" if images_dir.name else "Extract"
    for imgs, names in tqdm(loader, desc=desc, ncols=120):
        out = session.run(None, {input_name: imgs.numpy()})[0]
        all_ids.extend(names)
        all_embs.append(out)
    if len(all_embs) == 0:
        return pd.DataFrame(columns=["id"])
    embs_array = np.concatenate(all_embs, axis=0)
    dim_cols = [f"dim_{i}" for i in range(embs_array.shape[1])]
    df = pd.DataFrame(embs_array, columns=dim_cols)
    df.insert(0, "id", all_ids)
    return df


def _extract_one_csv_onnx(
    session,
    input_name: str,
    images_dir: Path,
    csv_path: Path | None,
    extract_size: int,
    batch_size: int,
    num_workers: int,
    eval_transform: str = "center-crop",
    *,
    dataset: RecordImageDataset | None = None,
) -> pd.DataFrame:
    if dataset is not None:
        if csv_path is not None:
            raise ValueError("pass either a CSV path or a prebuilt dataset, not both")
        ds = dataset
    else:
        if csv_path is None:
            raise ValueError("a CSV path or a prebuilt dataset is required")
        ds = CSVImageDataset(
            images_dir=images_dir,
            csv_path=csv_path,
            extract_size=extract_size,
            eval_transform=eval_transform,
        )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    all_ids: list[str] = []
    all_samples: list[str] | None = [] if ds.samples is not None else None
    all_embs: list[np.ndarray] = []

    for batch in tqdm(loader, desc="Extract CSV", ncols=120):
        if len(batch) == 3:
            imgs, names, samples = batch
        else:
            imgs, names = batch
            samples = None
        out = session.run(None, {input_name: imgs.numpy()})[0]
        all_ids.extend(list(names))
        if all_samples is not None and samples is not None:
            all_samples.extend([str(v) for v in list(samples)])
        all_embs.append(out)

    if not all_embs:
        return pd.DataFrame(columns=["id"])
    arr = np.concatenate(all_embs, axis=0)
    cols = [f"dim_{i}" for i in range(arr.shape[1])]
    out = pd.DataFrame(arr, columns=cols)
    out.insert(0, "id", all_ids)
    if all_samples is not None:
        out.insert(1, "sample", all_samples)
    return out


_EXTRACT_PARAMETERS = """
checkpoint_path, images_dir, model_name, extract_size, batch_size, device,
num_workers, use_projector_output, use_student, token_mode, topk_patches,
attention_pooling_type, attention_pooling_epochs, attention_train_csv, seed,
extract_csv, attention_pooling_checkpoint_path, onnx_path, eval_transform
"""


def extract_embeddings(
    checkpoint_path: Path | None,
    images_dir: Path,
    model_name: str = "vit_tiny_patch16_224",
    extract_size: int | None = None,
    batch_size: int = 32,
    device: str = "auto",
    num_workers: int = 0,
    use_projector_output: bool = False,
    use_student: bool = False,
    token_mode: str = "cls",
    topk_patches: int = 20,
    attention_pooling_type: str = "lightweight",
    attention_pooling_epochs: int = 20,
    attention_train_csv: Path | None = None,
    seed: int = 42,
    extract_csv: Path | None = None,
    attention_pooling_checkpoint_path: Path | None = None,
    onnx_path: Path | None = None,
    eval_transform: str = "center-crop",
    *,
    records: DataBundle | None = None,
) -> pd.DataFrame:
    """Extract embeddings and return DataFrame (see ``_extract_embeddings_result``)."""
    return _extract_embeddings_result(
        checkpoint_path=checkpoint_path,
        images_dir=images_dir,
        model_name=model_name,
        extract_size=extract_size,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        use_projector_output=use_projector_output,
        use_student=use_student,
        token_mode=token_mode,
        topk_patches=topk_patches,
        attention_pooling_type=attention_pooling_type,
        attention_pooling_epochs=attention_pooling_epochs,
        attention_train_csv=attention_train_csv,
        seed=seed,
        extract_csv=extract_csv,
        attention_pooling_checkpoint_path=attention_pooling_checkpoint_path,
        onnx_path=onnx_path,
        eval_transform=eval_transform,
        records=records,
    ).frame


def _extract_embeddings_result(
    checkpoint_path: Path | None,
    images_dir: Path,
    model_name: str = "vit_tiny_patch16_224",
    extract_size: int | None = None,
    batch_size: int = 32,
    device: str = "auto",
    num_workers: int = 0,
    use_projector_output: bool = False,
    use_student: bool = False,
    token_mode: str = "cls",
    topk_patches: int = 20,
    attention_pooling_type: str = "lightweight",
    attention_pooling_epochs: int = 20,
    attention_train_csv: Path | None = None,
    seed: int = 42,
    extract_csv: Path | None = None,
    attention_pooling_checkpoint_path: Path | None = None,
    onnx_path: Path | None = None,
    eval_transform: str = "center-crop",
    *,
    records: DataBundle | None = None,
) -> ExtractionResult:
    """Extract embeddings and return the frame plus what actually ran.

    For SSL pretrain checkpoints the teacher (EMA model) is loaded by default,
    consistent with the ref script and DINO/iBOT convention.  Pass
    ``use_student=True`` to load the student weights instead.

    By default extracts the raw CLS token from the backbone. Pass
    ``use_projector_output=True`` to use the checkpoint's projector or task embedding.

    Token modes:
      - ``cls``: CLS token (or projector output if enabled)
      - ``patch-topk``: flatten top-K patch tokens by L2 norm
      - ``attention-pool``: attention pool patch tokens (learned pool if available,
        otherwise CLS-query pooling)
    """
    if token_mode not in TOKEN_MODES:
        raise ValueError(
            f"Unsupported token_mode '{token_mode}', choose from: "
            f"{', '.join(TOKEN_MODES)}"
        )
    if topk_patches < 1:
        raise ValueError("--topk-patches must be >= 1")
    if attention_pooling_type not in ATTENTION_POOLING_TYPES:
        raise ValueError(
            "Unsupported --attention-pooling-type, choose from: lightweight, multihead, gated"
        )
    if attention_pooling_epochs < 1:
        raise ValueError("--attention-pooling-epochs must be >= 1")

    if records is not None:
        if extract_csv is not None or attention_train_csv is not None:
            raise ValueError(
                "records cannot be combined with extract_csv or attention_train_csv; "
                "the manifest already supplies the selection and labels"
            )
        records.raise_for_errors()
        _check_records_root(records, images_dir)

    effective: dict[str, Any] = {
        "requested": {
            "model_name": model_name,
            "extract_size": extract_size,
            "device": device,
            "token_mode": token_mode,
            "use_projector_output": bool(use_projector_output),
            "use_student": bool(use_student),
            "topk_patches": topk_patches,
            "attention_pooling_type": attention_pooling_type,
            "attention_pooling_epochs": attention_pooling_epochs,
            "eval_transform": eval_transform,
        },
        "data_path": "records" if records is not None else (
            "csv" if extract_csv is not None else "directory"
        ),
    }
    used_files: list[FileSnapshot] = []
    record_dataset = None
    if records is not None:
        effective["selection"] = {
            "selection_rule": records.selection_rule,
            "records": len(records.records),
            "selector_columns": list(records.selector_columns),
        }
        if records.selector is not None:
            used_files.append(records.selector)

    norm_probe: list[dict] = []
    # Set only by a run that trained a pool; written by ``_result`` so that a failure
    # anywhere before success leaves no unregistered weights behind.
    pending_pool: dict[str, Any] = {"path": None, "make_payload": None}

    def _persist_pool() -> None:
        """Publish the trained pool, owning the file outright or not at all.

        The entry is created with ``O_EXCL``, so the no-overwrite rule holds at the
        instant of writing rather than at some earlier check: a file that appeared in
        between is never clobbered. The protected scope opens *before* the descriptor
        exists, so an interruption arriving right after a successful create -- not
        only a failure during the write -- still closes and removes it.
        """
        path = pending_pool["path"]
        if path is None or pending_pool["make_payload"] is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_fd: int | None = None
        handle = None
        try:
            try:
                raw_fd = os.open(
                    path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
                )
            except FileExistsError:
                raise ValueError(
                    "Refusing to overwrite the existing attention pooling "
                    f"checkpoint: {path}"
                ) from None
            handle = os.fdopen(raw_fd, "wb")
            raw_fd = None
            with handle:
                torch.save(pending_pool["make_payload"](), handle)
                handle.flush()
                os.fsync(handle.fileno())
            effective["attention_pooling"]["written_file"] = str(path)
            print(f"[Info] Saved finetuned attention pooling checkpoint: {path}")
        except BaseException:
            # A descriptor or a wrapped handle is proof this call created the entry;
            # the FileExistsError path has neither and must never touch the file.
            owned = handle is not None or raw_fd is not None
            if handle is not None and not handle.closed:
                try:
                    handle.close()
                except OSError:  # pragma: no cover - already closed
                    pass
            elif raw_fd is not None:
                try:
                    os.close(raw_fd)
                except OSError:  # pragma: no cover - already closed
                    pass
            if owned:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError as cleanup_error:
                    # Best effort: never let a failed removal replace the real error.
                    print(
                        f"[Warning] could not remove the partial attention pooling "
                        f"checkpoint {path}: {cleanup_error}"
                    )
            raise

    def _result(frame: pd.DataFrame) -> ExtractionResult:
        if norm_probe:
            effective["backbone_output_norm"] = _backbone_norm_record(norm_probe[0])
        _persist_pool()
        return ExtractionResult(
            frame=frame, effective=dict(effective), used_files=tuple(used_files)
        )

    if onnx_path is not None:
        if token_mode != "cls":
            raise ValueError(
                f"ONNX inference only supports token-mode 'cls', got '{token_mode}'. "
                "Use PyTorch checkpoint for patch-topk or attention-pool modes."
            )
        onnx_size = resolve_onnx_input_size(Path(onnx_path))
        if extract_size is not None and extract_size != onnx_size:
            raise ValueError(
                f"--extract-size {extract_size} does not match ONNX input size "
                f"{onnx_size} (the ONNX graph governs the resolution)."
            )
        if records is not None:
            record_dataset = RecordImageDataset(records, onnx_size, eval_transform)
            effective["input_normalization"] = _transform_normalization(
                record_dataset.transform
            )
        effective["input_size"] = onnx_size
        effective["device"] = "cpu (onnxruntime)"
        effective["feature_output"] = "onnx_graph_output"
        effective["backbone_output_norm"] = {
            "status": "unavailable",
            "reason": "ONNX graph does not expose backbone normalization",
        }
        effective["post_feature_ops"] = []
        effective["vector_normalization"] = {
            "status": "unavailable",
            "reason": "the graph output contract is not introspectable",
        }
        frame = _extract_with_onnx(
            onnx_path=Path(onnx_path),
            images_dir=images_dir,
            extract_size=onnx_size,
            batch_size=batch_size,
            num_workers=num_workers,
            extract_csv=extract_csv,
            eval_transform=eval_transform,
            records=records,
            dataset=record_dataset,
            effective=effective,
            used_files=used_files,
        )
        dim = _frame_dim(frame)
        effective["feature_chain"] = {
            "token_mode": token_mode,
            "ops": ["onnx_graph_output"],
            "output_dim": dim,
        }
        return _result(frame)

    if checkpoint_path is None:
        raise ValueError("--checkpoint is required when --onnx-path is not provided.")

    dev = resolve_device(device)
    effective["device"] = str(dev)
    print(f"Resolved device: {dev}")
    checkpoint_payload = None
    if records is not None and checkpoint_path is not None:
        from otuformer.data.provenance import read_verified_bytes

        checkpoint_bytes, checkpoint_snapshot = snapshot_bytes(checkpoint_path)
        used_files.append(checkpoint_snapshot)
        checkpoint_payload = torch.load(
            io.BytesIO(read_verified_bytes(checkpoint_snapshot)),
            map_location="cpu",
            weights_only=False,
        )
        del checkpoint_bytes
    model, checkpoint_size = _load_model(
        checkpoint_path,
        model_name,
        dev,
        use_student=use_student,
        payload=checkpoint_payload,
        effective=effective,
    )
    requested_size = checkpoint_size if extract_size is None else extract_size
    validate_input_size(
        requested_size, model, getattr(model, "model_name", model_name)
    )
    effective["input_size"] = requested_size
    effective["preprocessing"] = {
        "eval_transform": eval_transform,
        "input_size": requested_size,
    }
    if records is not None:
        # Build the record view only once the effective resolution is known.
        record_dataset = RecordImageDataset(records, requested_size, eval_transform)
        effective["input_normalization"] = _transform_normalization(
            record_dataset.transform
        )
    if extract_size is None:
        print(f"[Info] Auto extract size from checkpoint: {requested_size}")
    print(
        f"[Info] Extraction mode: token_mode={token_mode}, "
        f"use_projector_output={use_projector_output}, topk_patches={topk_patches}"
    )

    if token_mode == "attention-pool":
        policy = _attention_pool_policy(model, attention_pooling_type)
        ckpt_candidates = _attention_pooling_checkpoint_candidates(
            checkpoint_path=checkpoint_path,
            pooling_type=attention_pooling_type,
            target_checkpoint_path=attention_pooling_checkpoint_path,
            policy=policy,
        )
        save_ckpt_path = ckpt_candidates[0]
        attention_pool = _build_attention_pooling(
            pooling_type=attention_pooling_type,
            dim=model.backbone.num_features,
        ).to(dev)

        ckpt_path: Path | None = None
        pool_source: dict[str, Any] = {"status": "missing"}
        for candidate in ckpt_candidates:
            if not candidate.exists():
                continue
            if records is not None:
                _pool_bytes, pool_snapshot = snapshot_bytes(candidate)
                used_files.append(pool_snapshot)
                payload = torch.load(
                    io.BytesIO(read_verified_bytes(pool_snapshot)),
                    map_location="cpu",
                    weights_only=False,
                )
            else:
                payload = torch.load(candidate, map_location="cpu", weights_only=False)
            compatible, reason = _pool_policy_compatible(payload, policy)
            if not compatible:
                if candidate == save_ckpt_path:
                    # The policy-specific sibling is the canonical file for this
                    # request. Fail with its path instead of silently loading a
                    # different artifact for the same request.
                    raise ValueError(
                        f"Attention pooling checkpoint {candidate} {reason}; it "
                        "was trained on a different patch set than this backbone. "
                        "Remove it or point --attention-pooling-checkpoint at a "
                        "compatible file."
                    )
                # An incompatible legacy fallback is simply not reusable.
                continue
            if isinstance(payload, dict) and "teacher" in payload:
                model.attention_pool = attention_pool
                msg = model.load_state_dict(payload["teacher"], strict=False)
                missing_pool = [
                    key for key in msg.missing_keys if key.startswith("attention_pool.")
                ]
                if missing_pool:
                    # Older files saved the model before the trained pool was
                    # attached; loading them would silently use random weights.
                    raise ValueError(
                        f"Attention pooling checkpoint {candidate} contains no trained "
                        f"pool parameters ({len(missing_pool)} missing, e.g. "
                        f"{missing_pool[0]}); retrain it instead of reusing it."
                    )
                if msg.missing_keys or msg.unexpected_keys:
                    print(
                        f"[Warning] Finetuned checkpoint loaded with missing={len(msg.missing_keys)}, "
                        f"unexpected={len(msg.unexpected_keys)}"
                    )
            else:
                state = (
                    payload.get("attention_pool_state_dict")
                    if isinstance(payload, dict)
                    else None
                )
                if state is None:
                    raise ValueError(
                        f"Invalid attention pooling checkpoint: {candidate}, expected 'teacher' or 'attention_pool_state_dict'"
                    )
                attention_pool.load_state_dict(state, strict=True)
                model.attention_pool = attention_pool
            ckpt_path = candidate
            pool_source = {"status": "cached", "path": str(candidate)}
            print(f"[Info] Loaded finetuned attention pooling from: {ckpt_path}")
            break
        if ckpt_path is None:
            if attention_train_csv is None and records is None:
                raise ValueError(
                    "attention-pool needs a compatible attention-pool checkpoint. "
                    "None was found for num_prefix_tokens="
                    f"{policy['num_prefix_tokens']}, register_tokens="
                    f"{policy['register_tokens']}, attention_pooling_type="
                    f"{policy['attention_pooling_type']} at {save_ckpt_path}; "
                    "provide --label-csv or a labelled manifest to train one."
                )
            if records is None:
                columns = set(pd.read_csv(attention_train_csv, nrows=0).columns)
                if "image" not in columns or "label" not in columns:
                    raise ValueError(
                        "attention-pool query training requires --label-csv with 'image' and 'label' columns"
                    )
            print(
                f"[Info] No finetuned attention pooling checkpoint found: {save_ckpt_path}. "
                f"Starting query finetuning for {attention_pooling_epochs} epochs..."
            )
            _finetune_attention_query(
                model=model,
                attention_pool=attention_pool,
                images_dir=images_dir,
                label_csv=attention_train_csv,
                extract_size=requested_size,
                device=dev,
                batch_size=batch_size,
                num_workers=num_workers,
                num_epochs=attention_pooling_epochs,
                eval_transform=eval_transform,
                dataset=(
                    RecordLabelDataset(records, requested_size, eval_transform)
                    if records is not None
                    else None
                ),
            )
            pool_source = {"status": "trained_in_this_run"}
            if save_ckpt_path.exists():
                raise ValueError(
                    "Refusing to overwrite the existing attention pooling "
                    f"checkpoint: {save_ckpt_path}"
                )
            # Attach before serializing: ``model.state_dict()`` is what gets saved,
            # and it only contains attention_pool.* once the trained module is on the
            # model. Saving first produced a checkpoint with no trained parameters.
            model.attention_pool = attention_pool
            pending_pool["path"] = save_ckpt_path

            def _pool_payload() -> dict[str, Any]:
                return {
                    "teacher": model.state_dict(),
                    "args": {
                        "model_name": model_name,
                        "token_mode": token_mode,
                        **policy,
                        "attention_pooling_epochs": attention_pooling_epochs,
                        "source_checkpoint": str(checkpoint_path),
                        "seed": seed,
                    },
                }

            # Serialized only once the run is about to succeed, so the saved weights
            # are the final ones and a failure never leaves an unregistered file.
            pending_pool["make_payload"] = _pool_payload
        if not hasattr(model, "attention_pool"):
            model.attention_pool = attention_pool
        model.attention_pool.eval()
        effective["attention_pooling"] = {
            "type": attention_pooling_type,
            "source": pool_source,
            # Only set when this run wrote the file: the name carries a policy tag
            # the caller cannot derive, and provenance must name what was produced.
            "written_file": None,
            "policy": {key: policy[key] for key in ("num_prefix_tokens", "register_tokens", "attention_pooling_type") if key in policy},
        }

    effective["feature_output"] = {
        "cls": "projector_output" if use_projector_output else "backbone_cls_token",
        "patch-topk": "topk_patch_tokens_flattened",
        "attention-pool": "attention_pooled_patch_tokens",
    }[token_mode]
    # Whether the backbone normalized its own output is observed by a hook, not
    # assumed from the fact that extraction adds nothing.
    norm_probe.append(_probe_backbone_norm(getattr(model, "backbone", None)))

    if extract_csv is not None or record_dataset is not None:
        out_df = _extract_one_csv(
            model=model,
            images_dir=images_dir,
            csv_path=extract_csv,
            extract_size=requested_size,
            batch_size=batch_size,
            device=dev,
            num_workers=num_workers,
            use_projector_output=use_projector_output,
            token_mode=token_mode,
            topk_patches=topk_patches,
            eval_transform=eval_transform,
            dataset=record_dataset,
            effective=effective,
        )
        _record_feature_chain(
            effective,
            token_mode=token_mode,
            use_projector_output=use_projector_output,
            model=model,
            input_dim=_feature_input_dim(effective, model),
            output_dim=_frame_dim(out_df),
        )
        return _result(out_df)

    if detect_batch_mode(images_dir):
        subdirs = sorted(p for p in images_dir.iterdir() if p.is_dir())
        frames = []
        for sub in tqdm(subdirs, desc="Batch dirs", ncols=120):
            df = _extract_one_dir(
                model,
                sub,
                requested_size,
                batch_size,
                dev,
                num_workers,
                use_projector_output=use_projector_output,
                token_mode=token_mode,
                topk_patches=topk_patches,
                apply_patch_topk_pca=False,
                eval_transform=eval_transform,
                effective=effective,
            )
            if len(df) == 0:
                continue
            df.insert(1, "sample", sub.name)
            frames.append(df)
        if len(frames) == 0:
            return _result(pd.DataFrame(columns=["id", "sample"]))
        out_df = pd.concat(frames, ignore_index=True)
        if token_mode == "patch-topk":
            print(f"[Info] Applying PCA reduction for Top-{topk_patches} patches...")
            out_df = _apply_patch_topk_pca_to_df(
                out_df, topk_patches=topk_patches, effective=effective
            )
        _record_feature_chain(
            effective,
            token_mode=token_mode,
            use_projector_output=use_projector_output,
            model=model,
            input_dim=_feature_input_dim(effective, model),
            output_dim=_frame_dim(out_df),
        )
        return _result(out_df)
    out_df = _extract_one_dir(
        model,
        images_dir,
        requested_size,
        batch_size,
        dev,
        num_workers,
        use_projector_output=use_projector_output,
        token_mode=token_mode,
        topk_patches=topk_patches,
        apply_patch_topk_pca=False,
        eval_transform=eval_transform,
        effective=effective,
    )
    if token_mode == "patch-topk":
        print(f"[Info] Applying PCA reduction for Top-{topk_patches} patches...")
        out_df = _apply_patch_topk_pca_to_df(
            out_df, topk_patches=topk_patches, effective=effective
        )
    _record_feature_chain(
        effective,
        token_mode=token_mode,
        use_projector_output=use_projector_output,
        model=model,
        input_dim=_feature_input_dim(effective, model),
        output_dim=_frame_dim(out_df),
    )
    return _result(out_df)
