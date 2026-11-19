"""Checkpoint save/load utilities."""

from __future__ import annotations

import contextlib
import hashlib
import io
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np
import torch

from otuformer.constants import SSL_HEAD_BY_METHOD, SSL_HEADS, SSL_METHODS
from otuformer.training.model import (
    OTUFormerEncoder,
    ArcFaceEmbeddingHead,
    has_class_token,
    native_register_count,
)

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

# Top-level keys that make a payload a metric/head-only/legacy candidate even
# when it also stores SSL-shaped weights, and the fixed first-projector width
# that likewise marks a metric layout.
_METRIC_STATE_KEYS = ("loss_state_dict", "class_labels", "loss_func", "model")
_METRIC_PROJECTOR_WIDTH = 512
_SSL_PROJECTOR_WIDTH = 2048
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


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
    # Register layout resolved against the selected weights; ``None`` means
    # "not resolved yet" and is never treated as a count of zero.
    actual_register_tokens: int | None = None
    added_register_tokens: int | None = None
    # Active SSL training identity (method, head), separate from the inference
    # ``embedding_head`` layout and from any recorded source lineage.
    ssl_identity: tuple[str, str] | None = None
    # Validated SSL ancestor identity of a metric-head descendant.
    ssl_source_identity: tuple[str, str] | None = None


def save_checkpoint(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def load_checkpoint_bytes(data: bytes, map_location: str = "cpu") -> dict[str, Any]:
    """Deserialize exactly the bytes a caller already read and verified.

    Keeps the deserialized model and the recorded digest bound to one read: a
    second open could see a replaced file.
    """
    return torch.load(io.BytesIO(data), map_location=map_location, weights_only=False)


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

    A payload carrying *both* spellings of the same tensor is rejected instead of
    silently keeping whichever came last: picking one by insertion order would let
    a corrupt canonical tensor hide behind a healthy legacy alias (or the reverse),
    which contradicts "the complete selected branch was validated".
    """
    renamed: dict[str, Any] = {}
    conflicts: list[str] = []
    changed = False
    for key, value in state_dict.items():
        parts = key.split(".")
        target = key
        if len(parts) > 2 and parts[0] == "projector" and parts[1].isdigit():
            target = ".".join((parts[0], "net", *parts[1:]))
            changed = True
        if target in renamed:
            conflicts.append(target)
        renamed[target] = value
    if conflicts:
        raise ValueError(
            "Checkpoint projector state is ambiguous: "
            f"{sorted(set(conflicts))[:3]} appear under both the legacy "
            "projector.<i> and the canonical projector.net.<i> spelling."
        )
    return renamed if changed else state_dict


def _validate_ssl_pair(
    method: Any, head: Any, *, origin: str
) -> tuple[str, str]:
    """Return a validated ``(method, head)`` SSL identity pair."""
    if method not in SSL_METHODS:
        raise ValueError(
            f"Unsupported SSL method {method!r} in {origin}; expected one of "
            f"{SSL_METHODS}."
        )
    if head not in SSL_HEADS:
        raise ValueError(
            f"Unsupported SSL head {head!r} in {origin}; expected one of "
            f"{SSL_HEADS}."
        )
    expected = SSL_HEAD_BY_METHOD[method]
    if head != expected:
        raise ValueError(
            f"SSL identity in {origin} is inconsistent: method {method!r} pairs "
            f"with {expected!r}, not {head!r}."
        )
    return (method, head)


def _has_metric_markers(checkpoint: dict) -> bool:
    """True when the payload belongs to the metric/head-only/legacy family.

    Key presence decides, not truthiness: an empty ``loss`` state still marks
    the payload, so it cannot be relabelled as SSL from its weight shapes.
    """
    cfg = _config(checkpoint)
    if "embedding_head" in cfg or "loss" in cfg:
        return True
    return any(key in checkpoint for key in _METRIC_STATE_KEYS)


def _reject_contradictory_ssl(checkpoint: dict, reason: str) -> None:
    cfg = _config(checkpoint)
    if cfg.get("ssl_method") is not None or cfg.get("ssl_head") is not None:
        raise ValueError(
            f"Checkpoint mixes {reason} with active ssl_method/ssl_head metadata; "
            "SSL identity and metric layout are separate contracts."
        )
    if "dino_objective" in checkpoint:
        raise ValueError(
            f"Checkpoint mixes {reason} with DINO training state; SSL identity and "
            "metric layout are separate contracts."
        )


# The recorded identity's MLP: (layer index, parameter, expected rows, expected
# last dimension). ``None`` means "not fixed here" (input width / D).
_SSL_PROJECTOR_LAYOUT = (
    (0, "weight", _SSL_PROJECTOR_WIDTH, None),
    (0, "bias", _SSL_PROJECTOR_WIDTH, None),
    (2, "weight", _SSL_PROJECTOR_WIDTH, _SSL_PROJECTOR_WIDTH),
    (2, "bias", _SSL_PROJECTOR_WIDTH, None),
    (4, "weight", None, _SSL_PROJECTOR_WIDTH),
    (4, "bias", None, None),
)

# Training-only relational queue metadata. It belongs to relational payloads
# only; a read-only consumer never reconstructs the queue, but the shared
# identity validator still rejects it on a DINO identity.
_QUEUE_CONFIG_KEY = "relational_queue_config"
_QUEUE_STATE_KEY = "relational_queue"

# Config keys that describe an active DINO recipe. A relational identity that
# still records them (or DINO training state) has contradictory metadata.
# KoLeo is not one of them: it is a shared pretrain regularizer, so a relational
# payload may record the four keys when its weight is positive.
_DINO_ONLY_CONFIG_KEYS = (
    "dino_prototypes",
    "dino_centering",
    "dino_hidden_dim",
    "dino_mlp_init_std",
    "dino_mlp_init_bias",
    "dino_weight_norm_api",
    "dino_prototype_scale",
    "dino_freeze_last_layer_epochs",
    "dino_target_dtype",
    "dino_center_momentum",
    "dino_sinkhorn_iterations",
)


def _projector_tensor(
    state_dict: dict, index: int, kind: str
) -> tuple[str, Any] | tuple[None, None]:
    """Find one projector tensor under our naming or the ref-script naming."""
    for prefix in ("projector.net", "projector"):
        key = f"{prefix}.{index}.{kind}"
        value = state_dict.get(key)
        if isinstance(value, torch.Tensor):
            return key, value
    return None, None


def _validate_ssl_projector(
    cfg: dict, state_dict: dict, pair: tuple[str, str]
) -> None:
    """Require the recorded identity's complete D-dimensional MLP layout.

    Every layer, bias and shape of the three-linear ``ProjectionHead`` must be
    present: the shared readers load with ``strict=False``, so a missing tensor
    would otherwise stay at its random initialization while the payload still
    claims the identity.
    """
    last = None
    for index, kind, rows, columns in _SSL_PROJECTOR_LAYOUT:
        key, value = _projector_tensor(state_dict, index, kind)
        if value is None:
            raise ValueError(
                f"Checkpoint records SSL identity {pair} but its projector is "
                f"incomplete: missing projector.{index}.{kind}. A complete "
                f"{_SSL_PROJECTOR_WIDTH}-hidden MLP is required."
            )
        expected_ndim = 2 if kind == "weight" else 1
        if value.ndim != expected_ndim:
            raise ValueError(
                f"Checkpoint records SSL identity {pair} but {key} has "
                f"{value.ndim} dimensions instead of {expected_ndim}."
            )
        if rows is not None and int(value.shape[0]) != rows:
            raise ValueError(
                f"Checkpoint records SSL identity {pair} but {key} has shape "
                f"{tuple(value.shape)}; the recorded MLP needs {rows} rows."
            )
        if columns is not None and int(value.shape[-1]) != columns:
            raise ValueError(
                f"Checkpoint records SSL identity {pair} but {key} has shape "
                f"{tuple(value.shape)}; the recorded MLP needs last dimension "
                f"{columns}."
            )
        if index == 4 and kind == "weight":
            last = value
    assert last is not None  # guarded by the layout loop
    out_dim = cfg.get("out_dim")
    bias_key, bias = _projector_tensor(state_dict, 4, "bias")
    if int(bias.shape[0]) != int(last.shape[0]):
        raise ValueError(
            f"Checkpoint records SSL identity {pair} but {bias_key} has "
            f"{int(bias.shape[0])} entries while the output layer has "
            f"{int(last.shape[0])} rows."
        )
    if out_dim is None:
        return
    if isinstance(out_dim, bool) or not isinstance(out_dim, int) or out_dim <= 0:
        raise ValueError(
            f"Checkpoint records an invalid out_dim {out_dim!r} for SSL identity "
            f"{pair}; expected a positive integer."
        )
    if int(last.shape[0]) != out_dim:
        raise ValueError(
            f"Checkpoint SSL identity {pair} records out_dim={out_dim} but the "
            f"projector outputs {int(last.shape[0])} dimensions."
        )


def resolve_checkpoint_ssl_identity(
    checkpoint: dict, state_dict: dict
) -> tuple[str, str] | None:
    """Return the payload's active SSL identity, or ``None`` when it has none.

    Metric/head-only/legacy payloads own no SSL identity; a historical
    relational payload is inferred from its recognized MLP layout so old files
    keep loading, but that inference is never emitted as recorded provenance.
    """
    cfg = _config(checkpoint)
    method = cfg.get("ssl_method")
    head = cfg.get("ssl_head")
    declared = method is not None or head is not None

    if _has_metric_markers(checkpoint):
        _reject_contradictory_ssl(checkpoint, "a metric/legacy embedding payload")
        return None

    first = _linear_weight(state_dict, _FIRST_LINEAR_KEYS)
    width = int(first.shape[0]) if first is not None else None
    if width == _METRIC_PROJECTOR_WIDTH:
        _reject_contradictory_ssl(checkpoint, "a 512-wide metric projector")
        return None

    if declared:
        pair = _validate_ssl_pair(method, head, origin="checkpoint config")
        _validate_ssl_projector(cfg, state_dict, pair)
        if pair[0] == "relational":
            # Relational owns no DINO-exclusive config or objective state; its
            # presence means the payload is mislabelled or inconsistent.
            recorded = sorted(
                name for name in _DINO_ONLY_CONFIG_KEYS if name in cfg
            )
            if "dino_objective" in checkpoint or recorded:
                raise ValueError(
                    "Checkpoint records SSL identity "
                    f"{pair} but also carries DINO-only state "
                    f"({', '.join(recorded) or 'dino_objective'}); a relational "
                    "payload must not record DINO config or objective tensors."
                )
        else:
            # The relational history queue is relational-only state. Readers
            # ignore its tensors, but a DINO identity claiming them is
            # contradictory metadata, not an additive field.
            recorded = [
                name
                for name in (_QUEUE_CONFIG_KEY, _QUEUE_STATE_KEY)
                if (name in cfg if name == _QUEUE_CONFIG_KEY else name in checkpoint)
            ]
            if recorded:
                raise ValueError(
                    f"Checkpoint records SSL identity {pair} but also carries "
                    f"relational queue metadata ({', '.join(recorded)}); the "
                    "queue is relational-only and cannot belong to a DINO run."
                )
        return pair

    if "dino_objective" in checkpoint:
        raise ValueError(
            "Checkpoint stores DINO training state without a complete "
            "ssl_method/ssl_head identity pair."
        )

    if width == _SSL_PROJECTOR_WIDTH and (
        _linear_weight(state_dict, _LAST_LINEAR_KEYS) is not None
    ):
        return ("relational", PROJECTION_EMBEDDING_HEAD)
    return None


def resolve_checkpoint_ssl_source(checkpoint: dict) -> tuple[str, str] | None:
    """Return a metric payload's validated SSL source pair, if it records one.

    The pair is provenance only: it names the SSL ancestor and is anchored by a
    valid ``ssl_initialization_checkpoint_sha256``. A hash or path alone cannot
    establish method/head, and a missing pair stays valid for historical files.
    """
    cfg = _config(checkpoint)
    method = cfg.get("ssl_source_method")
    head = cfg.get("ssl_source_head")
    if method is None and head is None:
        return None
    if method is None or head is None:
        raise ValueError(
            "Checkpoint records an incomplete SSL source identity: "
            "ssl_source_method and ssl_source_head must both be present."
        )
    pair = _validate_ssl_pair(method, head, origin="ssl_source_* provenance")
    sha = cfg.get("ssl_initialization_checkpoint_sha256")
    if not isinstance(sha, str) or not _SHA256_HEX.match(sha.strip().lower()):
        raise ValueError(
            f"Checkpoint records SSL source identity {pair} without a valid "
            "ssl_initialization_checkpoint_sha256 (64 hex characters); a path or "
            "method name alone cannot anchor the source."
        )
    return pair


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
        ssl_identity=resolve_checkpoint_ssl_identity(checkpoint, state_dict),
        ssl_source_identity=resolve_checkpoint_ssl_source(checkpoint),
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


def reject_noncanonical_register_keys(state_dict: dict) -> None:
    """Reject any register key that is not exactly ``backbone.reg_token``.

    Wrapper prefixes (``module.``, ``_orig_mod.``, ``student.``) are never
    normalised, so a prefixed register tensor must fail here instead of falling
    through to the legacy zero-register path.
    """
    for key in state_dict:
        if key != "backbone.reg_token" and key.endswith("reg_token"):
            raise ValueError(
                f"Checkpoint register key {key!r} is not canonical; the only "
                "supported location is 'backbone.reg_token'."
            )


def recorded_register_count(checkpoint: dict) -> int | None:
    """Read ``config.register_tokens``, rejecting a malformed value.

    ``args.register_tokens`` is never consulted: it records the invoking option
    and can be ``None``/``0`` even when the saved architecture has registers.
    Every checkpoint entry point validates through this function so an
    unsupported recorded value (negative, boolean, string) is rejected by the
    design's decision table instead of silently passing as a legacy zero.
    """
    recorded = _config(checkpoint).get("register_tokens")
    if recorded is None:
        return None
    if isinstance(recorded, bool) or not isinstance(recorded, int) or recorded < 0:
        raise ValueError(
            "Checkpoint config.register_tokens must be a non-negative integer "
            f"count, got {recorded!r}."
        )
    return recorded


@contextlib.contextmanager
def preserved_rng():
    """Run a construction probe without advancing the caller's RNG streams."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    with torch.random.fork_rng(devices=devices, enabled=True):
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def resolve_register_layout(
    checkpoint: dict, state_dict: dict, native_backbone: Any
) -> tuple[int, int]:
    """Return ``(actual_count, added_count)`` for the selected encoder state.

    The recorded count is read only from ``config.register_tokens``: ``args``
    records the invoking option and may be ``None``/``0`` even when the saved
    architecture has registers. The native probe has already been validated by
    the encoder constructor, so this function reads its attributes instead of
    duplicating the unsupported-backbone/no-CLS error.
    """
    reject_noncanonical_register_keys(state_dict)
    native = native_register_count(native_backbone)
    hidden = int(getattr(native_backbone, "num_features", 0))
    cfg = _config(checkpoint)
    label = cfg.get("model_name") or _legacy_args(checkpoint).get("model_name") or "the checkpoint"
    recorded = recorded_register_count(checkpoint)
    tensor = state_dict.get("backbone.reg_token")
    shape = tuple(tensor.shape) if hasattr(tensor, "shape") else None

    def reject(reason: str) -> None:
        raise ValueError(f"Checkpoint register layout is invalid: {reason}.")

    if not has_class_token(native_backbone):
        if native > 0:  # pragma: no cover - the constructor rejects this first
            reject(f"{label} has native registers but no CLS token")
        if recorded not in (None, 0):
            reject(f"{label} has no CLS token but records register_tokens={recorded}")
        if tensor is not None:
            reject(f"{label} has no CLS token but stores a register tensor")
        return (0, 0)

    if native > 0:
        native_tensor = getattr(native_backbone, "reg_token", None)
        expected = (
            tuple(native_tensor.shape)
            if hasattr(native_tensor, "shape")
            else (1, native, hidden)
        )
        if recorded is not None and recorded != native:
            reject(
                f"{label} natively has {native} register tokens but records "
                f"register_tokens={recorded}"
            )
        if tensor is None or shape != expected:
            reject(
                f"{label} natively has {native} register tokens but the selected "
                f"state holds reg_token {shape} instead of {expected}"
            )
        return (native, 0)

    if recorded in (None, 0):
        if tensor is not None:
            reject(
                f"{label} natively has no register tokens and records "
                f"register_tokens={recorded} but stores reg_token {shape}"
            )
        return (0, 0)
    if recorded == 4:
        expected = (1, 4, hidden)
        if tensor is None or shape != expected:
            reject(
                f"{label} records register_tokens=4 but the selected state holds "
                f"reg_token {shape} instead of {expected}"
            )
        return (4, 4)
    reject(f"{label} records unsupported register_tokens={recorded}")
    raise AssertionError  # pragma: no cover - reject() always raises


def load_checkpoint_encoder(
    checkpoint: dict,
    architecture: CheckpointArchitecture,
    *,
    image_size: int,
) -> tuple[Any, CheckpointArchitecture]:
    """Rebuild a read-only encoder from a resolved checkpoint architecture.

    The selected state is pre-scanned for non-canonical register keys before
    any model exists, an RNG-isolated ``pretrained=False`` native encoder
    resolves the register layout, and that encoder is reused unless four
    registers must be added. The returned architecture carries both counts so
    every read-only consumer loads the same verified layout.
    """
    state_dict = normalize_legacy_projector_keys(architecture.state_dict)
    reject_noncanonical_register_keys(state_dict)
    with preserved_rng():
        native = OTUFormerEncoder(
            model_name=architecture.model_name,
            out_dim=architecture.embedding_dim,
            return_patch_tokens=True,
            img_size=image_size,
            pretrained=False,
        )
        actual, added = resolve_register_layout(
            checkpoint, state_dict, native.backbone
        )
        if added:
            encoder = OTUFormerEncoder(
                model_name=architecture.model_name,
                out_dim=architecture.embedding_dim,
                return_patch_tokens=True,
                img_size=image_size,
                pretrained=False,
                reg_tokens=4,
            )
        else:
            encoder = native
        resolved = architecture._replace(
            actual_register_tokens=actual, added_register_tokens=added
        )
        apply_checkpoint_weights(encoder, resolved)
    return encoder, resolved


def assert_complete_projector(state_dict: dict, projector: torch.nn.Module) -> None:
    """Reject partial saved heads before permissive encoder loading.

    Historical backbone-only files keep their existing fallback. Once a head
    has saved tensors, every parameter must fit the selected inference layout.
    """
    state = normalize_legacy_projector_keys(state_dict)
    provided = {key: value for key, value in state.items() if key.startswith("projector.")}
    if not provided:
        return
    expected = {f"projector.{key}": value for key, value in projector.state_dict().items()}
    missing = sorted(set(expected) - set(provided))
    unexpected = sorted(set(provided) - set(expected))
    mismatched = sorted(
        key for key in set(expected) & set(provided)
        if not isinstance(provided[key], torch.Tensor)
        or provided[key].shape != expected[key].shape
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "Checkpoint projector is incomplete or incompatible with the selected head: "
            f"{len(missing)} missing, {len(unexpected)} unexpected, "
            f"{len(mismatched)} shape-mismatched tensor(s) "
            f"(e.g. {(missing + unexpected + mismatched)[0]})."
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
    if architecture.ssl_identity is None and any(
        key.startswith("projector.") for key in state_dict
    ):
        assert_complete_projector(state_dict, model.projector)
    model.load_state_dict(state_dict, strict=False)


@dataclass(frozen=True)
class PretrainInitialization:
    """One validated backbone-only source for a new pretraining run.

    This is not a resume: only ``backbone.*`` transfers, heads and optimizer
    state start fresh, and the source file is never mutated.
    """

    path: str
    sha256: str
    source_key: str
    source_identity: tuple[str, str] | None
    architecture: "CheckpointArchitecture"
    image_size: int
    register_tokens: int
    backbone_state: dict[str, Any]

    def config_bundle(self) -> dict[str, Any]:
        """The closed ``config.pretrain_initialization`` bundle (version 1)."""
        identity = self.source_identity or (None, None)
        return {
            "version": 1,
            "path": self.path,
            "sha256": self.sha256,
            "source_key": self.source_key,
            "ssl_method": identity[0],
            "ssl_head": identity[1],
            "model_name": self.architecture.model_name,
            "image_size": int(self.image_size),
            "register_tokens": int(self.register_tokens),
            "transfer": "backbone-only",
        }


_PRETRAIN_INIT_KEYS = frozenset(
    {
        "version",
        "path",
        "sha256",
        "source_key",
        "ssl_method",
        "ssl_head",
        "model_name",
        "image_size",
        "register_tokens",
        "transfer",
    }
)


def validate_pretrain_initialization_bundle(bundle: Mapping[str, Any]) -> None:
    """Reject a malformed or contradicting initialization provenance bundle."""
    if not isinstance(bundle, Mapping):
        raise ValueError("config.pretrain_initialization must be a mapping.")
    missing = sorted(_PRETRAIN_INIT_KEYS - set(bundle))
    unexpected = sorted(set(bundle) - _PRETRAIN_INIT_KEYS)
    if missing or unexpected:
        raise ValueError(
            "config.pretrain_initialization fields do not match version 1 "
            f"(missing={missing}, unexpected={unexpected})."
        )
    if bundle["version"] != 1:
        raise ValueError(
            f"Unsupported pretrain_initialization version {bundle['version']!r}."
        )
    if bundle["transfer"] != "backbone-only":
        raise ValueError(
            "config.pretrain_initialization.transfer must be 'backbone-only'."
        )
    expected_sha = str(bundle["sha256"]).strip().lower()
    if not _SHA256_HEX.match(expected_sha):
        raise ValueError(
            "config.pretrain_initialization.sha256 must be 64 hex characters."
        )
    if bundle["source_key"] not in ("teacher", "student"):
        raise ValueError(
            "config.pretrain_initialization.source_key must be teacher or student."
        )


def prepare_pretrain_initialization(
    path: Path, *, source_key: str, default_model_name: str = ""
) -> "PretrainInitialization":
    """Load and validate one backbone-only source snapshot.

    The digest is taken from the exact bytes that were read, so a concurrently
    replaced file is detected before anything is copied into a new run. The
    caller's ``default_model_name`` is the existing fallback for recognized
    historical payloads that record no model metadata; recorded metadata always
    wins and a selected-branch state is never substituted.
    """
    if source_key not in ("teacher", "student"):
        raise ValueError(
            f"init source must be 'teacher' or 'student', got {source_key!r}."
        )
    resolved = Path(path)
    if not resolved.is_file():
        raise ValueError(f"Initialization checkpoint {resolved} does not exist.")
    raw = resolved.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Initialization checkpoint {resolved} is not a mapping payload."
        )

    # The selected branch must exist exactly as requested: the shared reader's
    # permissive fallback would silently initialize from the wrong weights.
    if source_key not in checkpoint:
        raise KeyError(
            f"Initialization checkpoint has no '{source_key}' branch; the other "
            "branch is never substituted."
        )
    state_dict = checkpoint[source_key]
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Initialization checkpoint '{source_key}' branch is not a state dict."
        )
    backbone = {
        key: value
        for key, value in state_dict.items()
        if key.startswith("backbone.")
    }
    if not backbone:
        raise ValueError(
            "Initialization checkpoint carries no 'backbone.*' tensors; raw and "
            "third-party backbone-only payloads are outside this contract."
        )
    for key, value in backbone.items():
        if not torch.is_tensor(value):
            raise ValueError(f"Initialization tensor {key!r} is not a tensor.")
        _require_finite_training_dtype(value, key)

    identity = resolve_checkpoint_ssl_identity(checkpoint, state_dict)
    if identity is None:
        raise ValueError(
            "Initialization checkpoint has no recognized SSL identity; metric, "
            "ONNX and raw backbone payloads are not supported here."
        )
    (
        architecture,
        image_size,
        register_tokens,
        target_backbone,
    ) = _pretrain_initialization_architecture(
        checkpoint,
        state_dict,
        source_key=source_key,
        default_model_name=default_model_name,
    )
    _require_backbone_fits(backbone, target_backbone, path=str(resolved))
    # Only ``backbone.*`` is transferred, but a source with a corrupt projector
    # is not a valid initialization identity: validate the complete selected
    # branch instead of laundering it because the head is discarded.
    # Historical ``projector.<i>.*`` names are a recognized layout the shared
    # reader already normalizes; normalizing a *copy* keeps that compatibility
    # without touching the payload the caller loaded.
    _require_complete_source_branch(
        normalize_legacy_projector_keys(dict(state_dict)),
        target_state=target_backbone,
        path=str(resolved),
    )

    return PretrainInitialization(
        path=str(resolved),
        sha256=digest,
        source_key=source_key,
        source_identity=identity,
        architecture=architecture,
        image_size=int(image_size),
        register_tokens=int(register_tokens),
        backbone_state=backbone,
    )


def _pretrain_initialization_architecture(
    checkpoint: dict,
    state_dict: dict,
    *,
    source_key: str,
    default_model_name: str,
) -> tuple["CheckpointArchitecture", int, int, dict[str, Any]]:
    """Resolve the source architecture, size and register layout.

    Recorded metadata outranks the caller's historical fallback, and the
    selected branch is the only state ever inspected. The register layout is
    resolved against an RNG-isolated ``pretrained=False`` native probe so a
    native-versus-added distinction survives into the new run.
    """
    from otuformer.utils.size import resolve_training_image_size

    cfg = _config(checkpoint)
    model_name = cfg.get("model_name")
    from_metadata = isinstance(model_name, str) and bool(model_name.strip())
    if not from_metadata:
        fallback = str(default_model_name or "").strip()
        if not fallback:
            raise ValueError(
                "Initialization checkpoint records no model name and no caller "
                "fallback model name was supplied; the source architecture cannot "
                "be established from weights alone."
            )
        model_name = fallback
    resolved_name = str(model_name).strip()
    image_size = int(resolve_training_image_size(checkpoint))
    embedding_dim = int(cfg.get("out_dim", 256))
    head = resolve_checkpoint_embedding_head(checkpoint, state_dict)
    with preserved_rng():
        native = OTUFormerEncoder(
            model_name=resolved_name,
            out_dim=embedding_dim,
            return_patch_tokens=True,
            img_size=image_size,
            pretrained=False,
        )
        actual, added = resolve_register_layout(checkpoint, state_dict, native.backbone)
    target = native
    if added:
        target = OTUFormerEncoder(
            model_name=resolved_name,
            out_dim=embedding_dim,
            return_patch_tokens=True,
            img_size=image_size,
            pretrained=False,
            reg_tokens=4,
        )
    architecture = CheckpointArchitecture(
        model_name=resolved_name,
        embedding_dim=embedding_dim,
        embedding_head=head,
        state_dict=state_dict,
        source_key=source_key,
        model_name_from_metadata=from_metadata,
        actual_register_tokens=actual,
        added_register_tokens=added,
        ssl_identity=resolve_checkpoint_ssl_identity(checkpoint, state_dict),
    )
    return architecture, image_size, actual, target.state_dict()


def _require_finite_training_dtype(tensor: Any, label: str) -> None:
    """Reject a tensor that is finite at its recorded precision but not after cast.

    The trainer runs in float32, so a float64 value above the float32 range
    becomes ``Inf`` on load. Checking only the saved dtype would accept it and
    fail later, mid-training.
    """
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} contains non-finite values.")
    try:
        cast = tensor.to(torch.float32)
    except (RuntimeError, TypeError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{label} cannot be loaded as the training dtype: {exc}") from exc
    if not bool(torch.isfinite(cast).all()):
        raise ValueError(
            f"{label} is not finite at the training float32 dtype; a value that is "
            "finite at its recorded precision can still overflow on load."
        )


def _require_complete_source_branch(
    state_dict: Mapping[str, Any], *, target_state: Mapping[str, Any], path: str
) -> None:
    """Validate the complete selected source encoder, not only ``backbone.*``.

    A NaN projector bias, a wrong projector input width or a missing projector
    layer means the payload is not a healthy training checkpoint, even though the
    transfer itself copies only the encoder.
    """
    expected = {
        key: value
        for key, value in target_state.items()
        if key.startswith("projector.")
    }
    missing = sorted(set(expected) - set(state_dict))
    mismatched = sorted(
        key
        for key in set(expected) & set(state_dict)
        if tuple(getattr(state_dict[key], "shape", ()))
        != tuple(getattr(expected[key], "shape", ()))
    )
    unexpected = sorted(
        key
        for key in set(state_dict) - set(expected)
        if key.startswith("projector.")
    )
    if missing or mismatched or unexpected:
        raise ValueError(
            f"Initialization checkpoint {path} does not carry a complete, "
            f"compatible projector: {len(missing)} missing (e.g. {missing[:2]}), "
            f"{len(mismatched)} shape mismatch(es) (e.g. {mismatched[:2]}) and "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:2]}). A corrupt source "
            "head is not a valid initialization source."
        )
    for key in expected:
        value = state_dict[key]
        if not torch.is_tensor(value):
            raise ValueError(f"Initialization tensor {key!r} is not a tensor.")
        _require_finite_training_dtype(value, f"Initialization tensor {key!r}")


def _require_backbone_fits(
    backbone: Mapping[str, Any], target_state: Mapping[str, Any], *, path: str
) -> None:
    """Reject an incomplete or mismatched source before any output changes.

    This runs at preparation time (the CLI does it before ``--overwrite`` can
    clear a directory) instead of only at load time, so a source with missing or
    differently shaped backbone tensors cannot destroy existing results first.
    """
    target = {
        key: value
        for key, value in target_state.items()
        if key.startswith("backbone.")
    }
    missing = sorted(set(target) - set(backbone))
    unexpected = sorted(set(backbone) - set(target))
    mismatched = sorted(
        key
        for key in set(target) & set(backbone)
        if tuple(getattr(target[key], "shape", ()))
        != tuple(getattr(backbone[key], "shape", ()))
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Initialization checkpoint {path} does not carry a complete backbone "
            f"for its recorded architecture: {len(missing)} missing, "
            f"{len(unexpected)} unexpected, {len(mismatched)} shape mismatch(es) "
            f"(e.g. {(missing + unexpected + mismatched)[0]})."
        )


def initialize_pretrain_backbone(
    student: Any, prepared: "PretrainInitialization"
) -> None:
    """Copy only ``backbone.*`` from the prepared source into a fresh encoder."""
    target = {
        key: value
        for key, value in student.state_dict().items()
        if key.startswith("backbone.")
    }
    missing = sorted(set(target) - set(prepared.backbone_state))
    unexpected = sorted(set(prepared.backbone_state) - set(target))
    mismatched = sorted(
        key
        for key in set(target) & set(prepared.backbone_state)
        if tuple(target[key].shape) != tuple(prepared.backbone_state[key].shape)
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "Initialization checkpoint backbone does not fit the new run: "
            f"{len(missing)} missing, {len(unexpected)} unexpected, "
            f"{len(mismatched)} shape mismatch(es) "
            f"(e.g. {(missing + unexpected + mismatched)[0]})."
        )
    state = student.state_dict()
    for key, tensor in prepared.backbone_state.items():
        state[key] = tensor
    student.load_state_dict(state, strict=True)
