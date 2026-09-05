"""Extract embeddings from images using a trained OTU-Former checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

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
    _make_eval_transform,
    _resolve_image_path,
    _supports_recursive_lookup,
)
from otuformer.training.model import OTUFormerEncoder
from otuformer.utils.checkpoint import load_checkpoint
from otuformer.utils.device import resolve_device

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ATTENTION_POOLING_TYPES = {"lightweight", "multihead", "gated"}


class ImageFolderDataset(Dataset):
    """Load all images from a directory."""

    def __init__(self, images_dir: Path, extract_size: int = 224) -> None:
        self.paths = sorted(
            p
            for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        # Use the same eval transform as pretrain/finetune: direct BICUBIC resize
        self.transform = _make_eval_transform(extract_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.paths[idx].name


class LabelCSVImageDataset(Dataset):
    """Load images and labels from a label CSV for attention-query training."""

    def __init__(
        self, images_dir: Path, label_csv: Path, extract_size: int = 224
    ) -> None:
        self.transform = _make_eval_transform(extract_size)
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
        self, images_dir: Path, csv_path: Path, extract_size: int = 224
    ) -> None:
        self.transform = _make_eval_transform(extract_size)
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


def _attention_pooling_checkpoint_candidates(
    checkpoint_path: Path,
    pooling_type: str,
    target_checkpoint_path: Path | None,
) -> list[Path]:
    legacy = _attention_pooling_checkpoint_path(checkpoint_path, pooling_type)
    if target_checkpoint_path is None:
        return [legacy]
    target = Path(target_checkpoint_path)
    if target == legacy:
        return [target]
    return [target, legacy]


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
    label_csv: Path,
    extract_size: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    num_epochs: int,
) -> None:
    dataset = LabelCSVImageDataset(
        images_dir=images_dir, label_csv=label_csv, extract_size=extract_size
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
                patch_tokens = feats[:, 1:, :]

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


def _apply_patch_topk_pca(embeddings: np.ndarray, topk_patches: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n_samples, dim = embeddings.shape
    preferred_dim = {10: 256, 20: 512, 30: 1024}.get(topk_patches, 512)
    if n_samples < 3:
        target_dim = min(preferred_dim, dim)
        if 0 < target_dim < dim:
            return embeddings[:, :target_dim]
        return embeddings

    max_possible_dim = min(n_samples - 1, dim)
    target_dim = min(preferred_dim, max_possible_dim)
    if target_dim <= 0 or target_dim >= dim:
        return embeddings

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=target_dim, random_state=42)
    reduced = pca.fit_transform(embeddings_scaled)
    explained = float(pca.explained_variance_ratio_.sum())
    print(f"[Info] PCA: {dim}D -> {target_dim}D (explained variance: {explained:.2%})")
    return reduced


def _apply_patch_topk_pca_to_df(df: pd.DataFrame, topk_patches: int) -> pd.DataFrame:
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    if not dim_cols:
        return df
    reduced = _apply_patch_topk_pca(df[dim_cols].to_numpy(), topk_patches=topk_patches)
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
) -> OTUFormerEncoder:
    """Load model from checkpoint.

    For SSL pretrain checkpoints the teacher is the EMA-averaged model and
    produces better embeddings for downstream analysis (consistent with ref
    script and DINO/iBOT convention).  Pass ``use_student=True`` to load the
    student weights instead.

    Priority for weight selection:
      - ``use_student=True``  → ``ckpt["student"]`` → ``ckpt["model_state_dict"]``
      - ``use_student=False`` → ``ckpt["teacher"]`` → ``ckpt["model_state_dict"]``
    """
    ckpt = load_checkpoint(checkpoint_path)
    cfg = ckpt.get("config", {})
    out_dim = cfg.get("out_dim") or cfg.get("metric_embed_dim", 256)
    resolved_model = cfg.get("model_name", model_name)
    model = OTUFormerEncoder(
        model_name=resolved_model, out_dim=out_dim, pretrained=False
    )

    if use_student:
        state_dict = ckpt.get("student") or ckpt.get("model_state_dict")
        source = "student" if "student" in ckpt else "model_state_dict"
    else:
        state_dict = ckpt.get("teacher") or ckpt.get("model_state_dict")
        source = "teacher" if "teacher" in ckpt else "model_state_dict"

    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    print(f"[Info] Loaded weights from checkpoint key: '{source}'")
    return model


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
) -> pd.DataFrame:
    from otuformer.training.model import OTUFormerEncoder

    ds = ImageFolderDataset(images_dir, extract_size)
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
                patch_tokens = feats[:, 1:, :]
                num_patches = patch_tokens.shape[1]
                k = min(topk_patches, num_patches)
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
                patch_tokens = feats[:, 1:, :]
                if not hasattr(model, "attention_pool"):
                    raise ValueError(
                        "attention-pool is not initialized; provide --label-csv so query can be trained"
                    )
                pooled, _ = model.attention_pool(patch_tokens)
                embs = pooled
            elif use_projector_output:
                # projector output: L2-normalised embedding
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
    csv_path: Path,
    extract_size: int,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
    use_projector_output: bool = False,
    token_mode: str = "cls",
    topk_patches: int = 20,
) -> pd.DataFrame:
    ds = CSVImageDataset(
        images_dir=images_dir, csv_path=csv_path, extract_size=extract_size
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
                patch_tokens = feats[:, 1:, :]
                num_patches = patch_tokens.shape[1]
                k = min(topk_patches, num_patches)
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
                patch_tokens = feats[:, 1:, :]
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
        arr = _apply_patch_topk_pca(arr, topk_patches=topk_patches)
    cols = [f"dim_{i}" for i in range(arr.shape[1])]
    out = pd.DataFrame(arr, columns=cols)
    out.insert(0, "id", all_ids)
    if all_samples is not None:
        out.insert(1, "sample", all_samples)
    return out


def _extract_with_onnx(
    onnx_path: Path,
    images_dir: Path,
    extract_size: int,
    batch_size: int,
    num_workers: int,
    extract_csv: Path | None,
) -> pd.DataFrame:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    if extract_csv is not None:
        return _extract_one_csv_onnx(
            session=session,
            input_name=input_name,
            images_dir=images_dir,
            csv_path=extract_csv,
            extract_size=extract_size,
            batch_size=batch_size,
            num_workers=num_workers,
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
    )


def _extract_one_dir_onnx(
    session,
    input_name: str,
    images_dir: Path,
    extract_size: int,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    ds = ImageFolderDataset(images_dir, extract_size)
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
    csv_path: Path,
    extract_size: int,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    ds = CSVImageDataset(
        images_dir=images_dir, csv_path=csv_path, extract_size=extract_size
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


def extract_embeddings(
    checkpoint_path: Path | None,
    images_dir: Path,
    model_name: str = "vit_tiny_patch16_224",
    extract_size: int = 224,
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
) -> pd.DataFrame:
    """Extract embeddings and return DataFrame.

    For SSL pretrain checkpoints the teacher (EMA model) is loaded by default,
    consistent with the ref script and DINO/iBOT convention.  Pass
    ``use_student=True`` to load the student weights instead.

    By default extracts the raw CLS token from the backbone.  Pass
    ``use_projector_output=True`` to use the L2-normalised projector output.

    Token modes:
      - ``cls``: CLS token (or projector output if enabled)
      - ``patch-topk``: flatten top-K patch tokens by L2 norm
      - ``attention-pool``: attention pool patch tokens (learned pool if available,
        otherwise CLS-query pooling)
    """
    if token_mode not in {"cls", "patch-topk", "attention-pool"}:
        raise ValueError(
            f"Unsupported token_mode '{token_mode}', choose from: cls, patch-topk, attention-pool"
        )
    if topk_patches < 1:
        raise ValueError("--topk-patches must be >= 1")
    if attention_pooling_type not in ATTENTION_POOLING_TYPES:
        raise ValueError(
            "Unsupported --attention-pooling-type, choose from: lightweight, multihead, gated"
        )
    if attention_pooling_epochs < 1:
        raise ValueError("--attention-pooling-epochs must be >= 1")

    if onnx_path is not None:
        if token_mode != "cls":
            raise ValueError(
                f"ONNX inference only supports token-mode 'cls', got '{token_mode}'. "
                "Use PyTorch checkpoint for patch-topk or attention-pool modes."
            )
        return _extract_with_onnx(
            onnx_path=Path(onnx_path),
            images_dir=images_dir,
            extract_size=extract_size,
            batch_size=batch_size,
            num_workers=num_workers,
            extract_csv=extract_csv,
        )

    if checkpoint_path is None:
        raise ValueError("--checkpoint is required when --onnx-path is not provided.")

    dev = resolve_device(device)
    model = _load_model(checkpoint_path, model_name, dev, use_student=use_student)
    print(
        f"[Info] Extraction mode: token_mode={token_mode}, "
        f"use_projector_output={use_projector_output}, topk_patches={topk_patches}"
    )

    if token_mode == "attention-pool":
        ckpt_candidates = _attention_pooling_checkpoint_candidates(
            checkpoint_path=checkpoint_path,
            pooling_type=attention_pooling_type,
            target_checkpoint_path=attention_pooling_checkpoint_path,
        )
        ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)
        save_ckpt_path = (
            Path(attention_pooling_checkpoint_path)
            if attention_pooling_checkpoint_path is not None
            else _attention_pooling_checkpoint_path(
                checkpoint_path, attention_pooling_type
            )
        )
        attention_pool = _build_attention_pooling(
            pooling_type=attention_pooling_type,
            dim=model.backbone.num_features,
        ).to(dev)

        if ckpt_path is not None:
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and "teacher" in payload:
                model.attention_pool = attention_pool
                msg = model.load_state_dict(payload["teacher"], strict=False)
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
                        f"Invalid attention pooling checkpoint: {ckpt_path}, expected 'teacher' or 'attention_pool_state_dict'"
                    )
                attention_pool.load_state_dict(state, strict=True)
                model.attention_pool = attention_pool
            print(f"[Info] Loaded finetuned attention pooling from: {ckpt_path}")
        else:
            if attention_train_csv is None:
                raise ValueError(
                    "attention-pool requires training a query vector. Please provide --label-csv for training data."
                )
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
                extract_size=extract_size,
                device=dev,
                batch_size=batch_size,
                num_workers=num_workers,
                num_epochs=attention_pooling_epochs,
            )
            save_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "teacher": model.state_dict(),
                    "args": {
                        "model_name": model_name,
                        "token_mode": token_mode,
                        "attention_pooling_type": attention_pooling_type,
                        "attention_pooling_epochs": attention_pooling_epochs,
                        "source_checkpoint": str(checkpoint_path),
                        "seed": seed,
                    },
                },
                save_ckpt_path,
            )
            print(
                f"[Info] Saved finetuned attention pooling checkpoint: {save_ckpt_path}"
            )
        if not hasattr(model, "attention_pool"):
            model.attention_pool = attention_pool
        model.attention_pool.eval()

    if extract_csv is not None:
        out_df = _extract_one_csv(
            model=model,
            images_dir=images_dir,
            csv_path=extract_csv,
            extract_size=extract_size,
            batch_size=batch_size,
            device=dev,
            num_workers=num_workers,
            use_projector_output=use_projector_output,
            token_mode=token_mode,
            topk_patches=topk_patches,
        )
        return out_df

    if detect_batch_mode(images_dir):
        subdirs = sorted(p for p in images_dir.iterdir() if p.is_dir())
        frames = []
        for sub in tqdm(subdirs, desc="Batch dirs", ncols=120):
            df = _extract_one_dir(
                model,
                sub,
                extract_size,
                batch_size,
                dev,
                num_workers,
                use_projector_output=use_projector_output,
                token_mode=token_mode,
                topk_patches=topk_patches,
                apply_patch_topk_pca=False,
            )
            if len(df) == 0:
                continue
            df.insert(1, "sample", sub.name)
            frames.append(df)
        if len(frames) == 0:
            return pd.DataFrame(columns=["id", "sample"])
        out_df = pd.concat(frames, ignore_index=True)
        if token_mode == "patch-topk":
            print(f"[Info] Applying PCA reduction for Top-{topk_patches} patches...")
            out_df = _apply_patch_topk_pca_to_df(out_df, topk_patches=topk_patches)
        return out_df
    out_df = _extract_one_dir(
        model,
        images_dir,
        extract_size,
        batch_size,
        dev,
        num_workers,
        use_projector_output=use_projector_output,
        token_mode=token_mode,
        topk_patches=topk_patches,
        apply_patch_topk_pca=False,
    )
    if token_mode == "patch-topk":
        print(f"[Info] Applying PCA reduction for Top-{topk_patches} patches...")
        out_df = _apply_patch_topk_pca_to_df(out_df, topk_patches=topk_patches)
    return out_df
