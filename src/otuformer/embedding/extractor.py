"""Extract embeddings from images using a trained OTU-Former checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from otuformer.training.dataset import _make_eval_transform
from otuformer.training.model import OTUFormerEncoder
from otuformer.utils.checkpoint import load_checkpoint
from otuformer.utils.device import resolve_device

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


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
    model = OTUFormerEncoder(model_name=resolved_model, out_dim=out_dim)

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
            if use_projector_output:
                # projector output: L2-normalised 256-d embedding
                embs = model(imgs)
                if isinstance(embs, tuple):
                    embs = embs[0]
            else:
                # default: raw CLS token from backbone (matches ref default behaviour)
                if isinstance(model, OTUFormerEncoder):
                    feats = model.backbone.forward_features(imgs)
                    if isinstance(feats, dict):
                        feats = feats["x"]
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
    dim_cols = [f"dim_{i}" for i in range(embs_array.shape[1])]
    df = pd.DataFrame(embs_array, columns=dim_cols)
    df.insert(0, "id", all_ids)
    return df


def extract_embeddings(
    checkpoint_path: Path,
    images_dir: Path,
    model_name: str = "vit_tiny_patch16_224",
    extract_size: int = 224,
    batch_size: int = 32,
    device: str = "auto",
    num_workers: int = 0,
    use_projector_output: bool = False,
    use_student: bool = False,
) -> pd.DataFrame:
    """Extract embeddings and return DataFrame.

    For SSL pretrain checkpoints the teacher (EMA model) is loaded by default,
    consistent with the ref script and DINO/iBOT convention.  Pass
    ``use_student=True`` to load the student weights instead.

    By default extracts the raw CLS token from the backbone.  Pass
    ``use_projector_output=True`` to use the L2-normalised projector output.
    """
    dev = resolve_device(device)
    model = _load_model(checkpoint_path, model_name, dev, use_student=use_student)

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
            )
            if len(df) == 0:
                continue
            df.insert(1, "sample", sub.name)
            frames.append(df)
        if len(frames) == 0:
            return pd.DataFrame(columns=["id", "sample"])
        return pd.concat(frames, ignore_index=True)
    return _extract_one_dir(
        model,
        images_dir,
        extract_size,
        batch_size,
        dev,
        num_workers,
        use_projector_output=use_projector_output,
    )
