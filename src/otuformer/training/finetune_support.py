"""Finetune losses, provenance, pseudo rounds and optimizer layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from otuformer.constants import (
    ARCFACE_FAMILY_LOSSES,
    DINO_CENTERING_MODES,
    DINO_PROTOTYPES_MIN,
    IBOT_PROTOTYPES_MIN,
    LEGACY_RELATIONAL_PAIRING,
    MASKING_STRATEGIES,
    MAX_SUBCENTERS,
    MIN_SUBCENTERS,
    PATCH_LOSS_MODES,
    RELATIONAL_CENTER_SOURCE_CHOICES,
    RELATIONAL_PAIRING_CHOICES,
    RELATIONAL_TEACHER_VIEW_CHOICES,
    SSL_HEAD_BY_METHOD,
    SSL_METHODS,
)
from otuformer.training.dataset import (
    FINETUNE_AUGMENTATIONS,
    ORIENTATION_POLICIES,
    PRETRAIN_AUGMENTATIONS,
    IndexedDataset,
    MetricDataset,
    MultiCropDataset,
    _build_recursive_index,
    _resolve_image_path,
    _supports_recursive_lookup,
    build_finetune_augmentation_config,
    build_pretrain_augmentation_config,
    build_weak_teacher_augmentation_config,
    center_crop_eval_transform,
)
from otuformer.training.loss import (
    ArcFaceLoss,
    LOSS_REGISTRY,
    SubCenterArcFaceLoss,
    SupConLoss,
    dino_distillation_loss,
    ibot_patch_loss,
    koleo_loss,
    masked_patch_cosine_loss,
)
from otuformer.training.model import (
    PATCH_TARGET_LAYERS,
    ArcFaceEmbeddingHead,
    OTUFormerEncoder,
    PatchObjective,
    native_register_count,
)
from otuformer.utils.checkpoint import (
    ARCFACE_EMBEDDING_HEAD,
    PROJECTION_EMBEDDING_HEAD,
    load_checkpoint,
    recorded_register_count,
    resolve_checkpoint,
    resolve_checkpoint_embedding_head,
    resolve_projector_out_dim,
    save_checkpoint,
)
from otuformer.utils.paths import canonical_image_ref, path_identity
from otuformer.utils.size import (
    _positive_int,
    resolve_backbone_native_size,
    resolve_training_image_size,
    validate_input_size,
)
from .augmentation import (
    _finetune_register_plan,
    _select_augmentation_profile,
    _select_orientation_policy,
)
from .pretrain_recipe import (
    _assert_complete_ssl_projector,
)

def _resolve_finetune_ssl_source(
    checkpoint: dict[str, Any],
) -> tuple[str, str] | None:
    """SSL identity of a fine-tune source.

    An active SSL payload yields its own pair (including a historical relational
    inference); a metric-head descendant yields only its validated, SHA-anchored
    source pair. A hash or path alone never produces a pair.
    """
    from otuformer.utils.checkpoint import (
        resolve_checkpoint_ssl_identity,
        resolve_checkpoint_ssl_source,
    )

    state = checkpoint.get("model_state_dict")
    if isinstance(state, dict) and state:
        identity = resolve_checkpoint_ssl_identity(checkpoint, state)
        if identity is not None:
            return identity
    return resolve_checkpoint_ssl_source(checkpoint)


def preflight_finetune_source(
    checkpoint: dict[str, Any], model_name: str
) -> tuple[str, int]:
    """Validate a fine-tune source before the CLI creates or clears output.

    Active/source SSL identity is validated against the payload's actual
    ``model_state_dict``, so a valid teacher selection, a zero-register fallback
    or a later metric-head replacement cannot hide a conflicting alias.
    """
    from otuformer.utils.checkpoint import (
        preserved_rng,
        resolve_checkpoint_ssl_identity,
        resolve_checkpoint_ssl_source,
        resolve_projector_out_dim,
    )

    cfg = checkpoint.get("config") or {}
    resolved = cfg.get("model_name") or model_name
    image_size = resolve_training_image_size(checkpoint)
    state = checkpoint.get("model_state_dict")
    declared_dim = cfg.get("out_dim") or cfg.get("metric_embed_dim")
    state_dim = resolve_projector_out_dim(state) if isinstance(state, dict) else None
    out_dim = int(declared_dim) if declared_dim else (state_dim or 256)
    if isinstance(state, dict) and state:
        identity = resolve_checkpoint_ssl_identity(checkpoint, state)
        # A recorded source pair is provenance only, but it must be anchored and
        # consistent even on a source that is not itself an SSL payload.
        resolve_checkpoint_ssl_source(checkpoint)
        if identity is not None:
            with preserved_rng():
                probe = OTUFormerEncoder(
                    model_name=resolved,
                    out_dim=out_dim,
                    return_patch_tokens=True,
                    img_size=image_size,
                    pretrained=False,
                )
            _assert_complete_ssl_projector(
                state, probe, out_dim, "model_state_dict"
            )
        else:
            from otuformer.utils.checkpoint import (
                assert_complete_projector,
                resolve_checkpoint_embedding_head,
            )

            head = resolve_checkpoint_embedding_head(checkpoint, state)
            with preserved_rng():
                probe = OTUFormerEncoder(
                    model_name=resolved, out_dim=out_dim, img_size=image_size,
                    pretrained=False,
                )
                if head == ARCFACE_EMBEDDING_HEAD:
                    probe.projector = ArcFaceEmbeddingHead(probe.backbone.num_features, out_dim)
            assert_complete_projector(state, probe.projector)
    return _finetune_register_plan(checkpoint, resolved, image_size, out_dim)


def _validate_finetune_resume(
    checkpoint: dict[str, Any], current_class_labels: list[str], finetune_epochs: int
) -> None:
    completed_epochs = int(checkpoint.get("epoch", -1)) + 1
    if finetune_epochs <= completed_epochs:
        raise ValueError("--finetune-epochs must exceed the completed checkpoint epoch.")
    saved_labels = checkpoint.get("class_labels")
    if saved_labels is not None and list(saved_labels) != current_class_labels:
        raise ValueError("Cannot resume: training class labels differ from checkpoint.")
    if saved_labels is None:
        print("[Warning] Resume checkpoint lacks class labels; validating class count only.")


_MISSING = object()


_PROJECTOR_FIRST_LINEAR_KEYS = ("projector.net.0.weight", "projector.0.weight")


_ENCODER_STATE_PREFIXES = ("backbone.", "projector.")


def _has_encoder_weights(state: object) -> bool:
    """True when ``state`` carries at least one recognizable encoder parameter."""
    return isinstance(state, dict) and any(
        str(key).startswith(_ENCODER_STATE_PREFIXES) for key in state
    )


_LOSS_DIAGNOSTIC_FIELDS = (
    "epoch",
    "mode",
    "usable_batches",
    "skipped_batches",
    "valid_anchors",
    "margin_satisfied_fraction",
    "compact_hinge_fraction",
    "compact_mean_penalty",
    "local_assignments",
    "argmax_hits",
    "center_direction_cosine",
)


class LossDiagnosticsLogger:
    """Append-only per-epoch v0.8.0 loss diagnostics with a fixed schema."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle), None)
            if header != list(_LOSS_DIAGNOSTIC_FIELDS):
                raise ValueError(
                    f"Unrecognized loss-diagnostics schema in {self.path}: "
                    f"{header}. Refusing to append. Move or remove the file to "
                    "start a new log."
                )
            return
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=_LOSS_DIAGNOSTIC_FIELDS).writeheader()

    def log(self, **fields: object) -> None:
        row = {name: fields.get(name, "") for name in _LOSS_DIAGNOSTIC_FIELDS}
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=_LOSS_DIAGNOSTIC_FIELDS).writerow(row)


class BatchIdTraceLogger:
    """Opt-in JSON-lines trace of each training batch's indices and image refs."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            # Appending on resume matches LossDiagnosticsLogger: an existing
            # trace is extended, never truncated.
            self.path.write_text("", encoding="utf-8")

    def log(self, *, epoch: int, step: int, indices, refs) -> None:
        record = {
            "epoch": int(epoch),
            "step": int(step),
            "indices": [int(index) for index in indices],
            "refs": [str(ref) for ref in refs],
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _prototype_centers(loss_fn: nn.Module) -> torch.Tensor | None:
    """Detached L2-normalized ``(C, K, D)`` prototypes, or ``None`` if none.

    Ordinary ArcFace stores one center per class, so its ``(C, D)`` weight is
    returned as ``(C, 1, D)`` to keep every consumer on one shape.
    """
    weight = getattr(getattr(loss_fn, "head", None), "weight", None)
    if weight is None:
        return None
    centers = F.normalize(weight.detach(), dim=-1).clone()
    return centers.unsqueeze(1) if centers.ndim == 2 else centers


def _prototype_k(loss_fn: nn.Module) -> int:
    """Number of centers per class, for sizing the diagnostic counters."""
    centers = _prototype_centers(loss_fn)
    return int(centers.shape[1]) if centers is not None else 1


def _empty_loss_diagnostic_counts(num_classes: int, k: int) -> dict[str, object]:
    return {
        "local": [[0] * k for _ in range(num_classes)],
        "global": [[0] * k for _ in range(num_classes)],
        "margin_hits": 0,
        "samples": 0,
    }


@torch.no_grad()
def _accumulate_loss_diagnostics(
    counts: dict[str, object],
    loss_fn: nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    """Accumulate detached assignment and margin counts for one batch.

    A local winner is the closest center of the sample's own class; a global
    winner is the closest center of any class. The two counts are separate
    states, and the margin fraction compares the margin-adjusted target logit
    against the best rival logit rather than the ArcFace threshold branch.
    """
    centers = _prototype_centers(loss_fn)
    if centers is None:
        return
    num_classes, k, _ = centers.shape
    features = F.normalize(embeddings.detach(), dim=-1)
    cosine = (features @ centers.reshape(-1, centers.shape[-1]).T).reshape(
        features.shape[0], num_classes, k
    )
    rows = torch.arange(features.shape[0], device=features.device)
    global_winners = cosine.reshape(features.shape[0], -1).argmax(dim=1)
    local_winners = cosine[rows, labels].argmax(dim=1)
    for row in range(features.shape[0]):
        winner = int(global_winners[row])
        counts["global"][winner // k][winner % k] += 1
        counts["local"][int(labels[row])][int(local_winners[row])] += 1
    if num_classes >= 2:
        logits = loss_fn.head(embeddings.detach(), labels)
        target = logits.gather(1, labels[:, None]).squeeze(1)
        one_hot = F.one_hot(labels, num_classes).bool()
        rival = logits.masked_fill(one_hot, float("-inf")).max(dim=1).values
        counts["margin_hits"] += int((target > rival).sum())
    counts["samples"] += int(features.shape[0])


@torch.no_grad()
def _compact_hinge_stats(loss_fn: nn.Module) -> tuple[float | None, float | None]:
    """Same-class center-pair hinge activation and mean weighted penalty."""
    centers = _prototype_centers(loss_fn)
    weight = float(getattr(loss_fn, "compact_weight", 0.0))
    cap = getattr(loss_fn, "cap", None)
    if centers is None or weight == 0.0 or cap is None or centers.shape[1] < 2:
        return (None, None)
    penalties: list[float] = []
    for i in range(centers.shape[1]):
        for j in range(i + 1, centers.shape[1]):
            distance = 1.0 - (centers[:, i] * centers[:, j]).sum(dim=-1)
            penalties.extend(F.relu(distance - float(cap)).tolist())
    if not penalties:
        return (None, None)
    fraction = sum(1 for value in penalties if value > 0.0) / len(penalties)
    return (fraction, weight * (sum(penalties) / len(penalties)))


def _center_direction_cosine(
    previous: torch.Tensor | None, current: torch.Tensor | None
) -> list[list[float]]:
    """Per-center cosine between two post-epoch normalized center snapshots."""
    if previous is None or current is None:
        return []
    cosine = (previous * current).sum(dim=-1).clamp(-1.0, 1.0)
    return [[float(value) for value in row] for row in cosine]


def _sha256_file(path: Path) -> str:
    """SHA-256 of a checkpoint's bytes, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_image_ref(ref: object, images_root: Path) -> str:
    """Canonical POSIX reference relative to ``input_images_dir``.

    Shared with pseudo-label discovery so both use one definition; see
    :func:`otuformer.utils.paths.canonical_image_ref`.
    """
    return canonical_image_ref(ref, images_root)


def _train_manifest_sha256(
    image_refs: object, label_names: object, images_root: Path
) -> str:
    """SHA-256 of the canonical ``[image_ref, label]`` training manifest.

    Duplicate rows are preserved and rows are sorted, so the hash covers which
    images and labels were listed rather than their CSV order.
    """
    rows = sorted(
        [_canonical_image_ref(ref, images_root), str(label)]
        for ref, label in zip(image_refs, label_names)
    )
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _path_identity(resolved: Path) -> object:
    """Per-scan file identity; see :func:`otuformer.utils.paths.path_identity`."""
    return path_identity(resolved)


def _expert_path_identity(
    image_refs: object, images_root: Path
) -> tuple[bool, str | None]:
    """Expert-only path-identity check used for pseudo-source eligibility.

    Ordinary manifest validation runs first and keeps its hard failures; this
    check only answers whether the expert rows can seed one pseudo round. A
    duplicate resolved identity (alias) or a reference escaping the image root
    makes the checkpoint ineligible rather than failing ordinary training.

    References are resolved with the same rule :class:`MetricDataset` uses:
    direct path first, then the recursive relative/name index. Otherwise a CSV
    that names bare filenames for a species-subdirectory tree reads fine during
    training yet would be reported missing here.
    """
    from otuformer.training.dataset import (
        _build_recursive_index,
        _resolve_image_path,
        _supports_recursive_lookup,
    )

    root = Path(images_root).resolve()
    refs = [str(ref) for ref in image_refs]
    by_relative: dict[str, Path] = {}
    by_name: dict[str, list[Path]] = {}
    if any(
        _supports_recursive_lookup(ref) and not (root / Path(ref)).exists()
        for ref in refs
    ):
        by_relative, by_name = _build_recursive_index(root)
    seen: set[object] = set()
    for text in refs:
        try:
            target = _resolve_image_path(root, text, by_relative, by_name)
        except FileNotFoundError as exc:
            # Several files share the bare name, so no single identity exists.
            return False, f"ambiguous expert reference {text!r}: {exc}"
        try:
            resolved = target.resolve()
        except OSError as exc:
            return False, f"unresolvable expert reference {text!r}: {exc}"
        if resolved != root and root not in resolved.parents:
            return False, f"expert reference escapes the image root: {text!r}"
        try:
            identity = _path_identity(resolved)
        except OSError as exc:
            return False, f"missing expert image {text!r}: {exc}"
        if identity in seen:
            return False, f"duplicate expert path identity: {text!r}"
        seen.add(identity)
    return True, None


def _pseudo_source_status(
    *, loss: str, ssl_initialization_sha256: str | None, expert_reason: str | None
) -> tuple[bool, str | None]:
    """Decide whether a completed finetune#1 checkpoint may seed a pseudo round."""
    if loss not in ARCFACE_FAMILY_LOSSES:
        return False, f"loss {loss!r} is not ArcFace-family"
    if not ssl_initialization_sha256:
        return False, "missing SSL initialization file SHA-256"
    if expert_reason:
        return False, expert_reason
    return True, None


def _resolved_finetune_identity(
    *,
    optimizer_layout: str,
    finetune_lr: float,
    metric_head_lr: float | None,
    weight_decay: float,
    orientation_policy: str,
    batch_size: int,
    finetune_epochs: int,
    long_tail: str,
) -> dict[str, object]:
    """Resolved experiment-identity keys the existing config block lacks.

    The other identity keys (loss settings, dimensions, freeze ratio, seed,
    augmentation, image size) are already written by the caller; this returns
    only the additive ones so the checkpoint keeps a single identity record.
    """
    return {
        "finetune_lr": float(finetune_lr),
        "effective_metric_head_lr": float(
            finetune_lr if metric_head_lr is None else metric_head_lr
        ),
        "weight_decay": float(weight_decay),
        "orientation_policy": orientation_policy,
        "batch_size": int(batch_size),
        "finetune_epochs": int(finetune_epochs),
        "optimizer_name": "adamw",
        "optimizer_groups": optimizer_layout,
        "arcface_scale": 64.0,
        "arcface_margin": 0.5,
        "long_tail": long_tail,
    }


CB_DRW_BETA = 0.99


CB_DRW_MAX_CLASS_WEIGHT = 3.0


CB_DRW_START_FRACTION = 0.5


CB_DRW_RAMP_FRACTION = 0.1


def _effective_number_class_weights(
    expert_counts, *, beta: float = CB_DRW_BETA, cap: float = CB_DRW_MAX_CLASS_WEIGHT
) -> torch.Tensor:
    """Class-Balanced effective-number weights, sample-mean normalized then capped.

    Normalization uses expert training rows, and the cap applies to the target
    class weight before the manifest-wide denominator, so the final effective
    coefficient can exceed ``cap``.
    """
    # Compute in CPU float64: MPS has no float64, and the tiny class-weight
    # vector never needs to live on the accelerator.
    counts = torch.as_tensor(expert_counts).cpu().to(torch.float64)
    raw = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
    row_mean = float((raw * counts).sum() / counts.sum())
    return torch.clamp(raw / row_mean, max=cap)


def _drw_lambda(completed_steps: int, total_steps: int) -> float:
    """Deferred-reweighting interpolation: 0 before start, 1 after the ramp."""
    start = int(total_steps * CB_DRW_START_FRACTION)
    ramp = int(total_steps * CB_DRW_RAMP_FRACTION)
    if ramp <= 0:
        return 1.0 if completed_steps >= start else 0.0
    return min(max((completed_steps - start) / ramp, 0.0), 1.0)


def _cb_drw_class_weights(
    target: torch.Tensor, completed_steps: int, total_steps: int
) -> torch.Tensor:
    lam = _drw_lambda(completed_steps, total_steps)
    return 1.0 + lam * (target - 1.0)


def _cb_drw_r_bar(class_weights: torch.Tensor, training_counts) -> float:
    """Manifest-wide mean row weight; never depends on the current batch."""
    counts = torch.as_tensor(
        training_counts, dtype=class_weights.dtype, device=class_weights.device
    )
    return float((class_weights * counts).sum() / counts.sum())


def _weighted_ce(
    per_row: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor, r_bar: float
) -> torch.Tensor:
    """``sum_i(w[label_i] * CE_i) / (actual_batch_size * r_bar)``."""
    return (class_weights[labels] * per_row).sum() / (per_row.numel() * r_bar)


REQUIRED_PSEUDO_SOURCE_KEYS = (
    "model_name",
    "metric_embed_dim",
    "embedding_head",
    "loss",
    "subcenters",
    "compact_weight",
    "compact_cap",
    "supcon_temperature",
    "freeze_ratio",
    "finetune_lr",
    "effective_metric_head_lr",
    "weight_decay",
    "batch_size",
    "seed",
    "finetune_epochs",
    "image_size",
    "augmentation_profile",
    "augmentation_config",
    "orientation_policy",
    "long_tail",
    "optimizer_name",
    "optimizer_groups",
    "arcface_scale",
    "arcface_margin",
    "train_manifest_sha256",
    "ssl_initialization_checkpoint_sha256",
)


def _assert_pseudo_source_identity(
    source_cfg: dict, resolved: dict, *, ignore: frozenset[str] = frozenset()
) -> None:
    """Reject any resolved experiment difference from the finetune#1 source.

    Omitted options are inherited before this check, so a difference here is a
    genuine mismatch (or an explicit conflict), never a default. ``ignore``
    covers keys governed by another check (for example ``--finetune-epochs``,
    which legitimately grows on resume).
    """
    for key, value in resolved.items():
        if key in ignore:
            continue
        if key in source_cfg and source_cfg[key] != value:
            raise ValueError(
                "pseudo mode requires the finetune#2 experiment identity to "
                f"match finetune#1: {key} is {value!r}, source has "
                f"{source_cfg[key]!r}."
            )


def _rows_sha256(rows: list[dict]) -> str:
    """Canonical hash of ``[image, label]`` rows, independent of row order."""
    payload = json.dumps(
        sorted([[str(row["image"]), str(row["label"])] for row in rows]),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expert_rows_from_csv(path: Path) -> list[dict]:
    from otuformer.utils.io import read_csv

    frame = read_csv(path)
    return [
        {"image": str(row["image"]), "label": str(row["label"]), "source": "expert"}
        for _index, row in frame.iterrows()
    ]


def _class_row_counts(
    rows: list[dict] | None, class_to_idx: dict, source: str
) -> dict[str, int]:
    """Per-class training-row counts for one row source (record-keeping only)."""
    counts = {str(label): 0 for label in class_to_idx}
    for row in rows or []:
        if row.get("source") != source:
            continue
        label = str(row.get("label"))
        if label in counts:
            counts[label] += 1
    return counts


def _warn_pseudo_resume_drift(
    cfg: dict[str, Any], args: Any, out_dir: Path
) -> None:
    """Warning-only finetune#2 resume checks (design section 7).

    None of these block: the checkpointed accepted rows are authoritative. A
    moved root, a changed non-accepted candidate pool, or a missing/edited
    ``pseudo_labels.csv`` only warns and logs the current state.
    """
    recorded_root = cfg.get("input_images_dir")
    current_root = str(Path(args.input_images_dir).resolve())
    if recorded_root and str(recorded_root) != current_root:
        print(
            "[Warning] Resuming finetune#2 with a different image root: "
            f"recorded {recorded_root}, current {current_root}."
        )

    from otuformer.embedding.pseudo_label import discover_candidates

    try:
        expert_rows = _expert_rows_from_csv(Path(args.train_data))
        _candidates, _rejections, discovery = discover_candidates(
            expert_rows, Path(args.input_images_dir)
        )
    except Exception as exc:  # noqa: BLE001 - warning only, never blocking
        print(f"[Warning] Cannot rescan the pseudo candidate pool: {exc}")
    else:
        recorded_pool = cfg.get("pseudo_candidate_pool_sha256")
        current_pool = discovery.get("candidate_pool_sha256")
        if recorded_pool and current_pool and recorded_pool != current_pool:
            print(
                "[Warning] Pseudo candidate pool changed since finetune#2 was "
                "checkpointed; checkpointed accepted rows still apply."
            )

    labels_path = Path(out_dir) / "pseudo_labels.csv"
    if not labels_path.is_file():
        print(
            "[Warning] pseudo_labels.csv is missing; checkpointed accepted rows "
            "remain authoritative."
        )
        return
    current_hash = _sha256_file(labels_path)
    recorded_csv = cfg.get("pseudo_labels_sha256")
    if recorded_csv and recorded_csv != current_hash:
        print(
            "[Warning] pseudo_labels.csv changed since checkpointing; current "
            f"sha256 {current_hash[:12]}."
        )
    else:
        print(f"[Info] pseudo_labels.csv sha256 {current_hash[:12]}")


def _write_pseudo_outputs(out_dir: Path, diagnostics: list[dict], summary: dict) -> None:
    import pandas as pd

    from otuformer.utils.io import write_json

    logs_dir = Path(out_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostics).to_csv(
        Path(out_dir) / "pseudo_labels.csv", index=False
    )
    write_json(summary, Path(out_dir) / "pseudo_summary.json")


def _assert_pseudo_paths(out_dir: Path, image_root: Path, inputs: list[Path]) -> None:
    """Reject output/root overlap and any file-valued input inside ``out_dir``."""
    out = Path(out_dir).resolve()
    root = Path(image_root).resolve()
    if out == root or out in root.parents or root in out.parents:
        raise ValueError(
            "pseudo out_dir must not overlap the biological image root: "
            f"out_dir={out} root={root}"
        )
    for path in inputs:
        resolved = Path(path).resolve()
        if resolved == out or out in resolved.parents:
            raise ValueError(f"input file is inside out_dir: {path}")


def _resolve_pseudo_ssl_path(cfg: dict, checkpoint: Path | None) -> Path:
    """Locate the recorded SSL initialization, verifying its file SHA-256."""
    recorded_sha = cfg.get("ssl_initialization_checkpoint_sha256")
    if not recorded_sha:
        raise ValueError("pseudo source records no SSL initialization file SHA-256")
    if checkpoint is not None:
        if not Path(checkpoint).is_file():
            raise ValueError(f"relocated SSL checkpoint not found: {checkpoint}")
        if _sha256_file(checkpoint) != recorded_sha:
            raise ValueError("relocated SSL checkpoint file SHA-256 does not match")
        return Path(checkpoint)
    recorded = cfg.get("ssl_initialization_checkpoint_path")
    if not recorded:
        raise ValueError(
            "pseudo source records no SSL checkpoint path; pass --checkpoint to locate it"
        )
    path = Path(recorded)
    if not path.is_file():
        raise ValueError(f"recorded SSL checkpoint is missing: {path}; pass --checkpoint")
    if _sha256_file(path) != recorded_sha:
        raise ValueError("recorded SSL checkpoint file SHA-256 does not match")
    return path


def resolve_pseudo_ssl_source(
    ssl_checkpoint: dict[str, Any], source_config: dict[str, Any]
) -> tuple[str, str] | None:
    """Validate the original SSL identity for a NEW pseudo round.

    The original payload's actual ``model_state_dict`` decides its identity.
    Fine-tune#1 may record a pair, but it is provenance only: it must match the
    original and be SHA-anchored. An unknown original identity with no recorded
    pair stays valid legacy behaviour; a recorded pair that cannot be verified
    is rejected instead of being inherited.
    """
    from otuformer.utils.checkpoint import (
        resolve_checkpoint_ssl_identity,
        resolve_checkpoint_ssl_source,
    )

    state = ssl_checkpoint.get("model_state_dict")
    original = None
    if isinstance(state, dict) and state:
        original = resolve_checkpoint_ssl_identity(ssl_checkpoint, state)
    recorded = resolve_checkpoint_ssl_source({"config": source_config})
    if recorded is None:
        return original
    if original is None:
        raise ValueError(
            "the pseudo source records an SSL source pair, but the original SSL "
            "checkpoint's identity is unknown; a SHA-256 alone cannot verify it."
        )
    if recorded != original:
        raise ValueError(
            f"the pseudo source records SSL source identity {recorded}, which "
            f"contradicts the original SSL checkpoint identity {original}."
        )
    return original


def pseudo_cli_identity(
    *,
    ssl_checkpoint: dict,
    source_cfg: dict,
    resolved_loss: dict,
    model_name: str,
    metric_embed_dim: int | None,
    finetune_epochs: int,
    finetune_lr: float,
    metric_head_lr: float | None,
    weight_decay: float,
    freeze_ratio: float,
    augmentation: str | None,
    orientation_policy: str | None,
    batch_size: int,
    seed: int,
    long_tail: str,
    source_class_labels: list[str] | None = None,
) -> dict:
    """Resolve the finetune#2 experiment identity from CLI inputs alone.

    Used by the CLI preflight so an identity conflict is reported before any
    output directory is created or cleared; ``run_finetune`` re-checks the same
    keys against its fully resolved values as a backstop.
    """
    profile = _select_augmentation_profile(augmentation, None, stage="finetune")
    policy = _select_orientation_policy(
        orientation_policy, ssl_checkpoint, stage="finetune", resume=False
    )
    image_size = resolve_training_image_size(ssl_checkpoint)
    ssl_cfg = ssl_checkpoint.get("config") or {}
    out_dim = metric_embed_dim
    if out_dim is None:
        out_dim = ssl_cfg.get("metric_embed_dim") or ssl_cfg.get("out_dim") or 256
    # The patch set of the initialized backbone is part of the experiment
    # identity: the same model name can train on different token sets (zero
    # versus four registers, or a CLS-free backbone).
    _register_mode, register_tokens = _finetune_register_plan(
        ssl_checkpoint, model_name, image_size, int(out_dim)
    )
    return {
        "model_name": model_name,
        "metric_embed_dim": int(out_dim),
        "embedding_head": _select_finetune_embedding_head(ssl_checkpoint),
        "loss": str(resolved_loss["loss"]),
        "subcenters": resolved_loss["subcenters"],
        "compact_weight": resolved_loss["compact_weight"],
        "compact_cap": resolved_loss["compact_cap"],
        "supcon_temperature": resolved_loss["supcon_temperature"],
        "freeze_ratio": float(freeze_ratio),
        "finetune_lr": float(finetune_lr),
        "effective_metric_head_lr": float(
            finetune_lr if metric_head_lr is None else metric_head_lr
        ),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "finetune_epochs": int(finetune_epochs),
        "image_size": int(image_size),
        "augmentation_profile": profile,
        "augmentation_config": build_finetune_augmentation_config(
            profile, image_size, orientation_policy=policy
        ),
        "orientation_policy": policy,
        "long_tail": long_tail,
        "class_labels": [
            str(label)
            for label in (
                source_class_labels
                if source_class_labels is not None
                else source_cfg.get("class_labels", [])
            )
        ],
        "optimizer_name": "adamw",
        "optimizer_groups": _finetune_optimizer_layout(
            ssl_checkpoint, False, str(resolved_loss["loss"])
        ),
        "arcface_scale": 64.0,
        "arcface_margin": 0.5,
        "register_tokens": int(register_tokens),
    }


def _load_pseudo_source(path: Path) -> dict[str, Any]:
    """Load a pseudo-source checkpoint as one lightweight error type."""
    from otuformer.embedding.pseudo_label import PseudoLabelError

    try:
        return load_checkpoint(Path(path))
    except PseudoLabelError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced as one lightweight error
        raise PseudoLabelError(
            f"cannot read pseudo source checkpoint: {exc}"
        ) from exc


def prepare_pseudo_round(
    *,
    pseudo_label_from: Path,
    train_data: Path,
    input_images_dir: Path,
    out_dir: Path,
    similarity_floor: float,
    min_gap: float,
    neighbors: int,
    long_tail: str,
    loss: str | None,
    checkpoint: Path | None,
    device: str,
    batch_size: int,
    num_workers: int,
    cap_multiplier: int | None = None,
    absolute_cap: int | None = None,
    visualize_data: Path | None = None,
    progress=None,
) -> dict:
    """Validate the pseudo source, generate one round in memory, and return it.

    Runs entirely before ``prepare_output_dir``: an empty or invalid round
    leaves no output directory or file behind.
    """
    from otuformer.embedding.pseudo_label import (
        ABSOLUTE_CAP,
        CAP_MULTIPLIER,
        PseudoLabelError,
        generate_pseudo_rows,
    )
    from otuformer.utils.io import read_csv

    if cap_multiplier is None:
        cap_multiplier = CAP_MULTIPLIER
    if absolute_cap is None:
        absolute_cap = ABSOLUTE_CAP

    source = _load_pseudo_source(pseudo_label_from)
    cfg = source.get("config") or {}
    if int(cfg.get("pseudo_round", 0)) != 0:
        raise ValueError("--pseudo-label-from cannot be a finetune#2 checkpoint")
    if not cfg.get("pseudo_source_eligible"):
        raise ValueError(
            "pseudo source is ineligible: "
            f"{cfg.get('pseudo_source_ineligible_reason') or 'unknown reason'}"
        )
    missing = [key for key in REQUIRED_PSEUDO_SOURCE_KEYS if key not in cfg]
    if missing:
        raise ValueError(
            "pseudo source checkpoint is missing required provenance/resolved "
            f"settings: {', '.join(sorted(missing))}"
        )
    if str(cfg.get("loss")) not in ARCFACE_FAMILY_LOSSES:
        raise ValueError("pseudo feedback supports only ArcFace-family losses")
    if loss is not None and str(cfg.get("loss")) != loss:
        raise ValueError(
            f"--loss must match the pseudo source ({cfg.get('loss')!r})"
        )
    if cfg.get("long_tail") != long_tail:
        raise ValueError(
            f"--long-tail must match the pseudo source ({cfg.get('long_tail')!r})"
        )

    frame = read_csv(train_data)
    expert_rows = [
        {"image": str(row["image"]), "label": str(row["label"])}
        for _index, row in frame.iterrows()
    ]
    expert_hash = _train_manifest_sha256(
        [row["image"] for row in expert_rows],
        [row["label"] for row in expert_rows],
        Path(input_images_dir),
    )
    if cfg.get("train_manifest_sha256") != expert_hash:
        raise ValueError(
            "--train-data does not match the pseudo source expert manifest"
        )

    # ``class_order`` is the actual classifier row order of the source run;
    # ``class_labels`` is the historical sorted-string metadata and can differ
    # for numeric labels. A missing table is rejected, never guessed, and these
    # checks run before the expensive candidate generation.
    class_labels = [str(label) for label in (source.get("class_order") or [])]
    if not class_labels:
        raise ValueError("pseudo source checkpoint has no recorded class order")
    recorded_labels = [str(label) for label in (source.get("class_labels") or [])]
    if not recorded_labels:
        raise ValueError("pseudo source checkpoint has no recorded class labels")
    if set(recorded_labels) != set(class_labels):
        raise ValueError(
            "pseudo source class order and historical class labels disagree"
        )
    if set(class_labels) != {row["label"] for row in expert_rows}:
        raise ValueError(
            "expert CSV labels do not match the pseudo source class table"
        )
    planned_epochs = int(cfg.get("finetune_epochs") or 0)
    completed_epochs = int(source.get("epoch", -1)) + 1
    if planned_epochs and completed_epochs < planned_epochs:
        raise ValueError(
            "pseudo source run is incomplete: "
            f"epoch {completed_epochs} of {planned_epochs}"
        )

    ssl_path = _resolve_pseudo_ssl_path(cfg, checkpoint)
    protected = [Path(pseudo_label_from), ssl_path, Path(train_data)]
    if visualize_data is not None:
        protected.append(Path(visualize_data))
    if checkpoint is not None:
        protected.append(Path(checkpoint))
    _assert_pseudo_paths(Path(out_dir), Path(input_images_dir), protected)

    accepted, diagnostics, summary = generate_pseudo_rows(
        expert_rows=expert_rows,
        image_root=Path(input_images_dir),
        pseudo_checkpoint=Path(pseudo_label_from),
        similarity_floor=similarity_floor,
        min_gap=min_gap,
        neighbors=neighbors,
        cap_multiplier=cap_multiplier,
        absolute_cap=absolute_cap,
        model_name=str(cfg.get("model_name") or "vit_tiny_patch16_224"),
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        progress=progress,
    )
    if not accepted:
        raise PseudoLabelError(
            "no candidate accepted; candidate count: "
            + str(summary.get("candidate_count"))
            + "; accepted count: "
            + str(summary.get("accepted_count"))
            + "; acceptance rate: "
            + json.dumps(summary.get("acceptance_rate"))
            + "; rejection counts: "
            + json.dumps(summary.get("rejection_reason_counts", {}), sort_keys=True)
            + "; score quantiles: "
            + json.dumps(summary.get("score_quantiles", {}), sort_keys=True)
            + "; counterfactual pass counts: "
            + json.dumps(
                summary.get("counterfactual_pass_counts", {}), sort_keys=True
            )
            + "; seed-count bins: "
            + json.dumps(summary.get("seed_count_bins", {}), sort_keys=True)
        )

    rows = [
        {"image": row["image"], "label": row["label"], "source": "expert"}
        for row in expert_rows
    ] + [
        {"image": row["image"], "label": row["label"], "source": "known-pseudo"}
        for row in accepted
    ]
    return {
        "rows": rows,
        "accepted": accepted,
        "diagnostics": diagnostics,
        "summary": summary,
        "class_labels": class_labels,
        "source_class_labels": class_labels,
        "ssl_checkpoint": str(ssl_path),
        "source_checkpoint": str(pseudo_label_from),
        "source_config": cfg,
    }


def _classify_finetune_source(checkpoint: dict[str, Any]) -> str:
    """Classify a finetune source per the v0.8.0 checkpoint decision table.

    Loss metadata and embedding-head metadata are independent: ``config.loss``
    selects the loss, ``config.embedding_head`` selects the head, and a
    present-but-empty ``loss_state_dict`` is never the same as an absent one.
    An SSL source additionally needs real encoder weights, so a checkpoint with
    neither loss metadata nor a non-empty ``model_state_dict`` is rejected
    instead of being mistaken for SSL initialization.
    Returns ``ssl``, ``v080``, ``legacy_arcface`` or ``head_only``; ambiguous
    or ref-script sources raise, because guessing a head or a loss from them is
    exactly what the v0.8.0 contract forbids.
    """
    if (
        "loss_func" in checkpoint
        or "model" in checkpoint
        or ("args" in checkpoint and "config" not in checkpoint)
    ):
        raise ValueError(
            "Unsupported legacy fine-tune checkpoint format: expected "
            "'model_state_dict' and 'loss_state_dict'. Ref-script checkpoints "
            "('model'/'loss_func', or 'args' without 'config') can be read by "
            "extract/export/cam but not resumed by finetune."
        )
    config = checkpoint.get("config") or {}
    if not _has_encoder_weights(checkpoint.get("model_state_dict")):
        raise ValueError(
            "Cannot use this checkpoint for finetune: it has no valid encoder "
            "weights in 'model_state_dict', so the encoder could only be "
            "re-initialized. Refusing an empty or unrecognized state dict."
        )
    declared_head = config.get("embedding_head")
    state = checkpoint.get("loss_state_dict", _MISSING)
    classifier = isinstance(state, dict) and "head.weight" in state
    if "loss" in config:
        if not declared_head:
            raise ValueError(
                "Malformed v0.8.0 fine-tune checkpoint: config.loss without "
                "config.embedding_head."
            )
        _validate_v080_loss_state(config, state)
        return "v080"
    if classifier:
        return "legacy_arcface"
    if declared_head:
        if state is not _MISSING:
            raise ValueError(
                "Ambiguous fine-tune checkpoint: it declares an embedding head "
                "but has a present loss_state_dict without a classifier, so it "
                "is neither a head-only initialization nor a fine-tune "
                "checkpoint."
            )
        # A declared head and no loss state at all: usable for a new run, but
        # there is nothing for --resume to restore.
        return "head_only"
    if state is _MISSING and "class_labels" not in checkpoint:
        return "ssl"
    raise ValueError(
        "Ambiguous fine-tune checkpoint: it declares neither an embedding head "
        "nor a classifier state, so the loss and head cannot be recovered; "
        "refusing to guess."
    )


def _validate_v080_loss_state(config: dict[str, Any], state: object) -> None:
    """Reject a v0.8.0 loss state that does not match its recorded mode.

    The recorded K and embedding width must agree with the classifier tensor,
    so a corrupt resume is rejected before any output directory is created.
    """
    loss = str(config["loss"])
    if loss not in LOSS_REGISTRY:
        raise ValueError(f"Malformed v0.8.0 checkpoint: unknown loss '{loss}'.")
    if loss == "supcon":
        if state != {}:
            raise ValueError(
                "Malformed v0.8.0 checkpoint: loss 'supcon' has no classifier "
                "and requires an empty loss_state_dict."
            )
        return
    weight = state.get("head.weight") if isinstance(state, dict) else None
    expected_ndim = 2 if loss == "arcface" else 3
    if not isinstance(weight, torch.Tensor) or weight.ndim != expected_ndim:
        raise ValueError(
            f"Malformed v0.8.0 checkpoint: loss '{loss}' requires a "
            f"{expected_ndim}-D loss_state_dict['head.weight']."
        )
    expected_k = config.get("subcenters")
    if expected_k is None and loss == "subcenter-arcface-compact":
        # Compact checkpoints written before K was recorded defaulted to 2.
        expected_k = 2
    if expected_ndim == 3 and expected_k is not None and int(weight.shape[1]) != int(
        expected_k
    ):
        raise ValueError(
            f"Malformed v0.8.0 checkpoint: loss '{loss}' records K={expected_k} "
            f"but head.weight has {int(weight.shape[1])} centers."
        )
    recorded_embed = config.get("metric_embed_dim") or config.get("out_dim")
    if recorded_embed is not None and int(weight.shape[-1]) != int(recorded_embed):
        raise ValueError(
            f"Malformed v0.8.0 checkpoint: recorded embedding dimension "
            f"{recorded_embed} but head.weight is {int(weight.shape[-1])}-wide."
        )


def _select_finetune_embedding_head(checkpoint: dict[str, Any]) -> str:
    kind = _classify_finetune_source(checkpoint)
    declared_head = (checkpoint.get("config") or {}).get("embedding_head")
    if declared_head:
        return declared_head
    if kind == "ssl":
        # SSL initialization installs a fresh fine-tune head.
        return ARCFACE_EMBEDDING_HEAD
    state_dict = checkpoint.get("model_state_dict") or {}
    if not any(key in state_dict for key in _PROJECTOR_FIRST_LINEAR_KEYS):
        raise ValueError(
            "Cannot determine the embedding head: the checkpoint has no "
            "config.embedding_head and no projector weights to infer it from."
        )
    return resolve_checkpoint_embedding_head(checkpoint, state_dict)


def _validate_finetune_embedding_dim(cfg: dict[str, Any], out_dim: int) -> None:
    """Reject a resume that changes the fine-tune embedding width.

    The saved classifier and optimizer state are shaped for the recorded
    dimension, so resizing is only valid for a new ``--checkpoint`` run.
    """
    saved = cfg.get("metric_embed_dim") or cfg.get("out_dim")
    if saved is None or int(saved) == int(out_dim):
        return
    raise ValueError(
        f"Cannot resume: checkpoint embedding dimension is {saved} but "
        f"--metric-embed-dim is {out_dim}; resize only when starting a new "
        "--checkpoint run."
    )


def _validate_finetune_freeze_ratio(checkpoint: dict[str, Any], freeze_ratio: float) -> None:
    """Reject a resume whose ``--freeze-ratio`` differs from the saved run.

    The optimizer only holds ``requires_grad`` backbone parameters, so a
    different freeze ratio changes a group's parameter count and
    ``optimizer.load_state_dict`` would fail with an opaque message.
    """
    saved = (checkpoint.get("config") or {}).get("freeze_ratio")
    if saved is None:
        return
    if not math.isclose(float(saved), float(freeze_ratio), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"Cannot resume: checkpoint was trained with freeze_ratio={saved} "
            f"but --freeze-ratio is {freeze_ratio}; pass --freeze-ratio {saved}."
        )


_OPTIMIZER_LAYOUT_GROUPS = {
    "legacy_single": 1,
    "legacy_split": 2,
    "v080_supcon": 2,
    "v080_prototype": 3,
}


def _finetune_optimizer_layout(
    checkpoint: dict[str, Any] | None,
    resume: bool,
    loss: str,
) -> str:
    """Decide which optimizer layout this run must build and load into.

    Group count alone cannot separate a v0.8.0 two-group SupCon checkpoint from
    a legacy two-group ArcFace one. The recorded ``config.optimizer_groups`` is
    the authority when present (a legacy resume keeps its historical layout
    through later resumes); otherwise the recorded ``config.loss`` policy
    decides, and an unmarked unexpected group count is rejected rather than
    guessed.
    """
    if not resume:
        return "v080_supcon" if loss == "supcon" else "v080_prototype"
    config = (checkpoint or {}).get("config") or {}
    groups = len(((checkpoint or {}).get("optimizer") or {}).get("param_groups") or [])
    recorded = config.get("optimizer_groups")
    if recorded is not None:
        if recorded not in _OPTIMIZER_LAYOUT_GROUPS:
            raise ValueError(
                f"Cannot resume: unknown recorded optimizer layout '{recorded}'."
            )
        expected = _OPTIMIZER_LAYOUT_GROUPS[recorded]
        if groups != expected:
            raise ValueError(
                f"Cannot resume: recorded optimizer layout '{recorded}' expects "
                f"{expected} optimizer parameter groups but the checkpoint has "
                f"{groups}."
            )
        return recorded
    saved_loss = config.get("loss")
    if saved_loss is not None:
        layout = "v080_supcon" if saved_loss == "supcon" else "v080_prototype"
        expected = _OPTIMIZER_LAYOUT_GROUPS[layout]
        if groups != expected:
            raise ValueError(
                f"Cannot resume: recorded loss '{saved_loss}' expects {expected} "
                f"optimizer parameter groups but the checkpoint has {groups}."
            )
        return layout
    if groups == 1:
        return "legacy_single"
    if groups == 2:
        return "legacy_split"
    raise ValueError(
        f"Cannot resume: unexpected optimizer parameter-group count {groups}."
    )


def _validate_finetune_resume_source(
    checkpoint: dict[str, Any], loss: str | None = None
) -> None:
    """Reject a resume whose source cannot restore a fine-tune run.

    The CLI preflight passes the resolved loss so the optimizer layout is also
    checked before any output exists; ``run_finetune`` uses it for the source
    check and relies on its own layout decision later.
    """
    kind = _classify_finetune_source(checkpoint)
    if kind in ("ssl", "head_only"):
        raise ValueError(
            "Cannot resume: the checkpoint is an SSL/initialization checkpoint "
            "with no fine-tune loss state; pass it as --checkpoint to start a "
            "new run."
        )
    if loss is not None:
        _finetune_optimizer_layout(checkpoint, True, loss)


def _build_finetune_optimizer_for_layout(
    layout: str,
    model: OTUFormerEncoder,
    loss_fn: nn.Module,
    backbone_lr: float,
    metric_head_lr: float | None,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build the optimizer for a resolved layout, by Parameter identity.

    Legacy layouts reproduce their historical grouping and decay exactly;
    only ``v080_*`` layouts use the zero-decay prototype group.
    """
    head_lr = backbone_lr if metric_head_lr is None else metric_head_lr
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    projector_params = list(model.projector.parameters())
    prototype_params = list(loss_fn.parameters())
    if layout == "v080_prototype":
        return _build_finetune_optimizer(
            model, loss_fn, backbone_lr, metric_head_lr, weight_decay
        )
    if layout == "legacy_single":
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad] + prototype_params,
            lr=backbone_lr,
            weight_decay=weight_decay,
        )
    group_specs = {
        "legacy_split": [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
            {
                "params": projector_params + prototype_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
            },
        ],
        "v080_supcon": [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": projector_params, "lr": head_lr, "weight_decay": weight_decay},
        ],
    }
    if layout not in group_specs:
        raise ValueError(f"Unsupported optimizer layout: {layout}")
    return torch.optim.AdamW(group_specs[layout])


def _resolve_finetune_loss_config(
    args: argparse.Namespace,
    checkpoint: dict[str, Any] | None,
    *,
    explicit_options: frozenset[str] | set[str] = frozenset(),
) -> dict[str, object]:
    """Resolve the effective loss mode and only its applicable settings.

    ``checkpoint`` is the saved fine-tune checkpoint on ``--resume`` and
    ``None`` for a new run, so an omitted ``--loss`` is ArcFace on a new run
    but inherits the recorded mode on resume. Only options the caller marked
    explicit may conflict with saved settings or be rejected as inapplicable;
    a CLI default value is not an explicit choice. Direct Python callers that
    omit ``explicit_options`` therefore get new-run defaults, not validation
    errors.
    """
    explicit = set(explicit_options)
    saved = (checkpoint or {}).get("config") or {}
    saved_loss = saved.get("loss")
    requested = getattr(args, "loss", None) or "arcface"

    if saved_loss is None:
        if checkpoint is not None and "loss" in explicit and requested != "arcface":
            raise ValueError(
                f"Cannot resume: legacy checkpoint has no recorded loss, so only "
                f"--loss arcface can be resumed, not '{requested}'."
            )
        loss = requested
    else:
        if "loss" in explicit and requested != saved_loss:
            raise ValueError(
                f"Cannot resume: checkpoint loss is '{saved_loss}' but --loss "
                f"is '{requested}'."
            )
        loss = saved_loss
    if loss not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown --loss '{loss}'; choose from {sorted(LOSS_REGISTRY)}."
        )

    # A compact checkpoint written before K was recorded stored
    # ``subcenters: None`` while training with the then-fixed K=2. Treat that as
    # a recorded 2 so an explicit conflicting --subcenters is rejected here
    # instead of later by a head-size mismatch. Copy rather than mutate the
    # checkpoint's own config dict.
    if (
        checkpoint is not None
        and loss == "subcenter-arcface-compact"
        and saved.get("subcenters") is None
    ):
        saved = {**saved, "subcenters": 2}

    def _effective(
        name: str,
        saved_key: str,
        default: object,
        applicable: bool,
        validate: Callable[[object], None],
    ) -> object:
        option = f"--{name.replace('_', '-')}"
        if name in explicit:
            if not applicable:
                raise ValueError(f"{option} does not apply to --loss {loss}.")
            value = getattr(args, name, None)
            validate(value)
            if saved.get(saved_key) is not None and saved[saved_key] != value:
                raise ValueError(
                    f"Cannot resume: checkpoint {name} is {saved[saved_key]} but "
                    f"{option} is {value}."
                )
            return value
        # A recorded ``None`` means the setting did not apply to the saved mode.
        if saved.get(saved_key) is not None:
            validate(saved[saved_key])
            return saved[saved_key]
        return default if applicable else None

    def _validate_subcenters(value: object) -> None:
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"--subcenters must be an integer, got {value}.")
        if not MIN_SUBCENTERS <= int(number) <= MAX_SUBCENTERS:
            raise ValueError(
                f"--subcenters must be between {MIN_SUBCENTERS} and "
                f"{MAX_SUBCENTERS}, got {value}."
            )

    def _validate_weight(value: object) -> None:
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(
                f"--compact-weight must be a finite value >= 0, got {value}."
            )

    def _validate_temperature(value: object) -> None:
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(
                f"--supcon-temperature must be a finite value > 0, got {value}."
            )

    is_compact = loss == "subcenter-arcface-compact"
    # Both Sub-center modes share the K setting; compact no longer fixes K=2.
    subcenters = _effective(
        "subcenters",
        "subcenters",
        2,
        loss == "subcenter-arcface" or is_compact,
        _validate_subcenters,
    )

    return {
        "loss": loss,
        "subcenters": subcenters,
        "compact_weight": _effective(
            "compact_weight", "compact_weight", 0.1, is_compact, _validate_weight
        ),
        "compact_cap": 0.5 if is_compact else None,
        "supcon_temperature": _effective(
            "supcon_temperature",
            "supcon_temperature",
            0.07,
            loss == "supcon",
            _validate_temperature,
        ),
    }


def _build_finetune_loss(
    loss_config: dict[str, object], embed_dim: int, num_classes: int
) -> nn.Module:
    """Instantiate the resolved loss by explicit mode dispatch.

    The four modes do not share a constructor signature, so each is built with
    only its applicable settings.
    """
    mode = loss_config["loss"]
    if mode == "arcface":
        return ArcFaceLoss(embed_dim, num_classes)
    if mode == "supcon":
        return SupConLoss(temperature=loss_config["supcon_temperature"])
    if mode == "subcenter-arcface":
        return SubCenterArcFaceLoss(
            embed_dim,
            num_classes,
            k=loss_config["subcenters"],
            compact_weight=0.0,
        )
    if mode == "subcenter-arcface-compact":
        return SubCenterArcFaceLoss(
            embed_dim,
            num_classes,
            k=loss_config["subcenters"],
            compact_weight=loss_config["compact_weight"],
            cap=loss_config["compact_cap"],
        )
    raise ValueError(f"Unsupported loss mode: {mode}")


def _build_finetune_optimizer(
    model: OTUFormerEncoder,
    loss_fn: nn.Module,
    backbone_lr: float,
    metric_head_lr: float | None,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """v0.8.0 prototype layout: backbone, projector, zero-decay prototype.

    Only normalized class prototypes drop weight decay; every other parameter
    keeps the requested policy.
    """
    head_lr = backbone_lr if metric_head_lr is None else metric_head_lr
    return torch.optim.AdamW(
        [
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
                "lr": backbone_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": list(model.projector.parameters()),
                "lr": head_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": list(loss_fn.parameters()),
                "lr": head_lr,
                "weight_decay": 0.0,
            },
        ]
    )
