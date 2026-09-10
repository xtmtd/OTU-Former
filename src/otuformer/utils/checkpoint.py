"""Checkpoint save/load utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import torch

from otuformer.training.model import ArcFaceEmbeddingHead

ARCFACE_EMBEDDING_HEAD = "arcface_mlp_512"
PROJECTION_EMBEDDING_HEAD = "projection_mlp_2048"
EMBEDDING_HEADS = (ARCFACE_EMBEDDING_HEAD, PROJECTION_EMBEDDING_HEAD)

# The projector's first linear width identifies which head a checkpoint holds:
# ArcFaceEmbeddingHead is 512-wide, ProjectionHead is 2048-wide.
_HEAD_BY_HIDDEN_DIM = {512: ARCFACE_EMBEDDING_HEAD, 2048: PROJECTION_EMBEDDING_HEAD}

# ``projector.<i>.*`` is the ref-script naming, ``projector.net.<i>.*`` is ours.
_FIRST_LINEAR_KEYS = ("projector.net.0.weight", "projector.0.weight")
_LAST_LINEAR_KEYS = (
    "projector.net.4.weight",
    "projector.4.weight",
    "projector.net.2.weight",
    "projector.2.weight",
)


class CheckpointArchitecture(NamedTuple):
    """Everything a read-only consumer needs to rebuild an encoder."""

    model_name: str
    embedding_dim: int
    embedding_head: str
    state_dict: dict[str, Any]
    source_key: str
    # True when ``model_name`` came from checkpoint metadata, which outranks
    # the caller's fallback and therefore cannot be corrected by a CLI flag.
    model_name_from_metadata: bool = False


def save_checkpoint(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def _linear_weight(state_dict: dict, keys: tuple[str, ...]) -> torch.Tensor | None:
    for key in keys:
        value = state_dict.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            return value
    return None


def _config(checkpoint: dict) -> dict:
    cfg = checkpoint.get("config")
    return cfg if isinstance(cfg, dict) else {}


def _legacy_args(checkpoint: dict) -> dict:
    args = checkpoint.get("args")
    return args if isinstance(args, dict) else {}


_WEIGHT_KEYS = ("teacher", "model_state_dict", "model", "student")


def resolve_checkpoint_state_dict(
    checkpoint: dict, prefer_student: bool = False
) -> tuple[dict, str]:
    """Return ``(state_dict, source_key)``, tolerating legacy script layouts.

    ``teacher`` (EMA, better embeddings) is preferred for SSL checkpoints.
    ``prefer_student`` puts ``student`` first but still falls back, so it has no
    effect on fine-tune checkpoints (which store neither). The ref script
    writes ``model`` for its ArcFace fine-tune checkpoints.
    """
    keys = ("student", *_WEIGHT_KEYS) if prefer_student else _WEIGHT_KEYS
    for key in keys:
        if key in checkpoint:
            return checkpoint[key], key
    raise KeyError(
        "Checkpoint has no model weights; expected one of "
        + ", ".join(repr(key) for key in keys)
        + "."
    )


def resolve_checkpoint_embedding_head(checkpoint: dict, state_dict: dict) -> str:
    """Return the embedding head a checkpoint holds.

    An explicit ``config.embedding_head`` wins; otherwise the head is inferred
    from the projector shapes, so checkpoints whose metadata is missing or
    wrong still load the module the weights actually fit.
    """
    declared = _config(checkpoint).get("embedding_head")
    if declared:
        if declared not in EMBEDDING_HEADS:
            raise ValueError(f"Unsupported embedding head: {declared}")
        return declared
    first = _linear_weight(state_dict, _FIRST_LINEAR_KEYS)
    if first is None:
        # No projector weights: historical OTU/SSL checkpoints use ProjectionHead.
        return PROJECTION_EMBEDDING_HEAD
    head = _HEAD_BY_HIDDEN_DIM.get(int(first.shape[0]))
    if head is None:
        raise ValueError(
            f"Unrecognised projector width {int(first.shape[0])}; expected 512 "
            f"({ARCFACE_EMBEDDING_HEAD}) or 2048 ({PROJECTION_EMBEDDING_HEAD})."
        )
    return head


def resolve_projector_out_dim(state_dict: dict) -> int | None:
    """Return the output width of the projector's last linear layer, if present."""
    last = _linear_weight(state_dict, _LAST_LINEAR_KEYS)
    return int(last.shape[0]) if last is not None else None


def normalize_legacy_projector_keys(state_dict: dict) -> dict:
    """Map ref-script ``projector.<i>.*`` names onto ``projector.net.<i>.*``.

    Returns ``state_dict`` unchanged (same object) when no key needs renaming.
    """
    renamed: dict[str, Any] = {}
    changed = False
    for key, value in state_dict.items():
        parts = key.split(".")
        if len(parts) > 2 and parts[0] == "projector" and parts[1].isdigit():
            key = ".".join((parts[0], "net", *parts[1:]))
            changed = True
        renamed[key] = value
    return renamed if changed else state_dict


def resolve_checkpoint(
    checkpoint: dict,
    default_model_name: str,
    prefer_student: bool = False,
) -> CheckpointArchitecture:
    """Resolve how to rebuild an encoder from a checkpoint.

    ``config`` wins over a ref-script ``args`` dict, which wins over the
    projector shapes in the weights, which win over the caller's defaults.
    """
    state_dict, source_key = resolve_checkpoint_state_dict(checkpoint, prefer_student)
    cfg = _config(checkpoint)
    args = _legacy_args(checkpoint)
    recorded_name = cfg.get("model_name") or args.get("model_name")
    model_name = recorded_name or default_model_name
    embedding_dim = (
        cfg.get("out_dim")
        or cfg.get("metric_embed_dim")
        or args.get("out_dim")
        or args.get("metric_embed_dim")
    )
    if embedding_dim is None:
        embedding_dim = resolve_projector_out_dim(state_dict)
    if embedding_dim is None:
        embedding_dim = 256
    return CheckpointArchitecture(
        model_name=str(model_name),
        embedding_dim=int(embedding_dim),
        embedding_head=resolve_checkpoint_embedding_head(checkpoint, state_dict),
        state_dict=state_dict,
        source_key=source_key,
        model_name_from_metadata=bool(recorded_name),
    )


def _drop_incompatible_buffers(model, state_dict: dict) -> dict:
    """Drop buffers whose shape differs from the rebuilt model.

    Read-only consumers never use ``center`` (SSL-only training state), and a
    fine-tune checkpoint may have been written while resizing the embedding
    head. Returns ``state_dict`` unchanged (same object) when nothing is
    dropped.
    """
    buffers = dict(model.named_buffers())
    kept: dict[str, Any] = {}
    dropped = False
    for key, value in state_dict.items():
        current = buffers.get(key)
        if (
            current is not None
            and isinstance(value, torch.Tensor)
            and current.shape != value.shape
        ):
            dropped = True
            continue
        kept[key] = value
    return kept if dropped else state_dict


def _reject_backbone_mismatch(
    model, state_dict: dict, architecture: CheckpointArchitecture
) -> None:
    """Fail loudly when the checkpoint's backbone does not fit the model.

    ``strict=False`` hides missing/unexpected keys, so a wrong model name (for
    example a ref-script checkpoint that records no config) would otherwise load
    a partly random backbone or die on an opaque size-mismatch error. Stateless
    backbones (test doubles) are skipped.
    """
    expected = {
        key: value
        for key, value in model.state_dict().items()
        if key.startswith("backbone.")
    }
    if not expected:
        return
    provided = {
        key: value for key, value in state_dict.items() if key.startswith("backbone.")
    }
    missing = sorted(set(expected) - set(provided))
    unexpected = sorted(set(provided) - set(expected))
    mismatched = sorted(
        key
        for key in set(expected) & set(provided)
        if expected[key].shape != provided[key].shape
    )
    if not (missing or unexpected or mismatched):
        return
    if architecture.model_name_from_metadata:
        hint = (
            "the checkpoint records that model name and it takes precedence over "
            "--model-name, so the metadata and the weights disagree"
        )
    else:
        hint = "the checkpoint records no model name; pass the correct --model-name"
    raise ValueError(
        f"Checkpoint backbone does not fit '{architecture.model_name}': "
        f"{len(missing)} missing, {len(unexpected)} unexpected, "
        f"{len(mismatched)} shape-mismatched backbone tensor(s) "
        f"(e.g. {(missing + unexpected + mismatched)[0]}); {hint}."
    )


def apply_checkpoint_weights(model, architecture: CheckpointArchitecture) -> None:
    """Install the checkpoint's embedding head on ``model`` and load its weights."""
    if architecture.embedding_head == ARCFACE_EMBEDDING_HEAD:
        model.projector = ArcFaceEmbeddingHead(
            model.backbone.num_features, architecture.embedding_dim
        )
    state_dict = _drop_incompatible_buffers(
        model, normalize_legacy_projector_keys(architecture.state_dict)
    )
    _reject_backbone_mismatch(model, state_dict, architecture)
    model.load_state_dict(state_dict, strict=False)
