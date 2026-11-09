"""Augmentation profiles, register plans and their validation."""

from __future__ import annotations

import json
from typing import Any, Callable, NamedTuple
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
from otuformer.utils.size import (
    _positive_int,
    resolve_backbone_native_size,
    resolve_training_image_size,
    validate_input_size,
)

_AUGMENTATION_STAGES = ("pretrain", "finetune")


def _validate_stage(stage: str) -> None:
    if stage not in _AUGMENTATION_STAGES:
        raise ValueError(
            f"Unsupported augmentation stage '{stage}'; "
            f"choose from: {', '.join(_AUGMENTATION_STAGES)}"
        )


def _valid_local_crops(value: Any) -> int | None:
    """Return ``value`` as a non-negative non-boolean int, or ``None`` if invalid."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _checkpoint_augmentation_metadata(
    checkpoint: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]] | None:
    """Return ``(profile, expanded_config)`` or ``None`` for absent/old metadata.

    A checkpoint is old only when both augmentation keys are absent from its
    ``config`` dict. Any other malformed combination raises ``ValueError``: the
    top-level ``augmentation_profile`` must match the self-contained
    ``augmentation_config['profile']``, and the saved orientation policy must be
    a known policy.
    """
    if checkpoint is None:
        return None
    cfg = checkpoint.get("config")
    if not isinstance(cfg, dict):
        return None
    has_profile = "augmentation_profile" in cfg
    has_config = "augmentation_config" in cfg
    if not has_profile and not has_config:
        return None
    if not has_profile or not has_config:
        raise ValueError(
            "malformed augmentation metadata: checkpoint must contain both "
            "'augmentation_profile' and 'augmentation_config'."
        )
    profile = cfg["augmentation_profile"]
    config = cfg["augmentation_config"]
    if not isinstance(profile, str):
        raise ValueError(
            "malformed augmentation metadata: 'augmentation_profile' must be a string."
        )
    if not isinstance(config, dict):
        raise ValueError(
            "malformed augmentation metadata: 'augmentation_config' must be a dict."
        )
    nested_profile = config.get("profile")
    if not isinstance(nested_profile, str):
        raise ValueError(
            "malformed augmentation metadata: 'augmentation_config.profile' must "
            "be a string."
        )
    if nested_profile != profile:
        raise ValueError(
            "malformed augmentation metadata: 'augmentation_profile' "
            f"({profile!r}) does not match 'augmentation_config.profile' "
            f"({nested_profile!r})."
        )
    if config.get("orientation_policy") not in ORIENTATION_POLICIES:
        raise ValueError(
            "malformed augmentation metadata: "
            "'augmentation_config.orientation_policy' must be one of: "
            f"{', '.join(ORIENTATION_POLICIES)}."
        )
    return profile, config


def _select_augmentation_profile(
    requested_profile: str | None,
    checkpoint: dict[str, Any] | None,
    *,
    stage: str,
) -> str:
    """Resolve the effective augmentation profile for ``stage``.

    Omitted values default per stage, inherit the saved profile on resume, and
    never inherit the pretraining profile when starting a new fine-tuning run
    (the caller passes ``checkpoint=None`` for that case).
    """
    _validate_stage(stage)
    default_profile = "global-barcode" if stage == "pretrain" else "none"
    metadata = _checkpoint_augmentation_metadata(checkpoint)
    if metadata is None:
        if checkpoint is None:
            if requested_profile is None:
                return default_profile
            return requested_profile
        compatible = "legacy" if stage == "pretrain" else "none"
        if requested_profile is None or requested_profile == compatible:
            return compatible
        raise ValueError(
            f"Cannot resume: checkpoint predates augmentation metadata; the only "
            f"compatible profile is '{compatible}'. Start a new run to change "
            "augmentation settings."
        )
    saved_profile, _ = metadata
    allowed_profiles = (
        PRETRAIN_AUGMENTATIONS if stage == "pretrain" else FINETUNE_AUGMENTATIONS
    )
    if saved_profile not in allowed_profiles:
        raise ValueError(
            f"Checkpoint augmentation profile '{saved_profile}' is not valid for "
            f"{stage}; choose from: {', '.join(allowed_profiles)}."
        )
    if requested_profile is None or requested_profile == saved_profile:
        return saved_profile
    raise ValueError(
        f"Cannot resume: augmentation profile '{requested_profile}' differs from "
        f"checkpoint profile '{saved_profile}'. Start a new run to change "
        "augmentation settings."
    )


def _select_orientation_policy(
    requested_policy: str | None,
    checkpoint: dict[str, Any] | None,
    *,
    stage: str,
    resume: bool = False,
) -> str:
    """Resolve the effective orientation policy for ``stage``.

    ``resume=True`` means the checkpoint is being continued and must match the
    requested policy exactly. ``resume=False`` with a checkpoint means a new
    fine-tuning run initialized via ``--checkpoint``: the saved policy is
    inherited when omitted and an explicit policy is honored.
    """
    _validate_stage(stage)
    if requested_policy is not None and requested_policy not in ORIENTATION_POLICIES:
        raise ValueError(
            f"Unknown orientation policy '{requested_policy}'; "
            f"choose from: {', '.join(ORIENTATION_POLICIES)}"
        )
    metadata = _checkpoint_augmentation_metadata(checkpoint)
    if metadata is None:
        if checkpoint is None:
            return requested_policy if requested_policy is not None else "sensitive"
        if stage == "pretrain" or resume:
            if requested_policy in (None, "invariant"):
                return "invariant"
            raise ValueError(
                "Cannot resume: checkpoint predates augmentation metadata and has no "
                "saved orientation policy. Use invariant or start a new run."
            )
        # New fine-tuning run initialized from an old pretraining checkpoint:
        # an explicit policy is honored, otherwise fall back to sensitive.
        return requested_policy if requested_policy is not None else "sensitive"
    _, saved_config = metadata
    saved_policy = saved_config["orientation_policy"]
    if stage == "pretrain" or resume:
        if requested_policy is None or requested_policy == saved_policy:
            return saved_policy
        raise ValueError(
            f"Cannot resume: orientation policy '{requested_policy}' differs from "
            f"checkpoint policy '{saved_policy}'. Start a new run to change it."
        )
    if requested_policy is None:
        return saved_policy
    return requested_policy


def _resolve_pretrain_local_views(
    requested_local_crop_size: int | None,
    requested_local_crops: int | None,
    checkpoint: dict[str, Any] | None,
) -> tuple[int, int]:
    """Resolve pretraining local-view size/count for a new run or resume.

    A new run uses the requested values or the historical defaults ``96``/``6``.
    A resume inherits the saved expanded config (new-style checkpoint) or saved
    ``args`` (old checkpoint), falling back to ``96``/``6`` only for missing or
    invalid old fields. Explicit values must match the resolved saved values.
    """
    if requested_local_crop_size is not None and _positive_int(
        requested_local_crop_size
    ) is None:
        raise ValueError(
            "local_crop_size must be a non-boolean positive integer, got "
            f"{requested_local_crop_size!r}."
        )
    if requested_local_crops is not None and _valid_local_crops(
        requested_local_crops
    ) is None:
        raise ValueError(
            "local_crops must be a non-boolean non-negative integer, got "
            f"{requested_local_crops!r}."
        )
    if checkpoint is None:
        return (
            int(requested_local_crop_size)
            if requested_local_crop_size is not None
            else 96,
            int(requested_local_crops) if requested_local_crops is not None else 6,
        )

    metadata = _checkpoint_augmentation_metadata(checkpoint)
    if metadata is not None:
        _, saved_config = metadata
        local_crop = saved_config.get("local_crop")
        if not isinstance(local_crop, dict):
            raise ValueError(
                "malformed augmentation metadata: 'augmentation_config.local_crop' "
                "must be a dict."
            )
        saved_size = _positive_int(local_crop.get("size"))
        saved_crops = _valid_local_crops(saved_config.get("local_crops"))
        if saved_size is None or saved_crops is None:
            raise ValueError(
                "malformed augmentation metadata: checkpoint records invalid "
                "local-view settings."
            )
    else:
        saved_args = checkpoint.get("args")
        if not isinstance(saved_args, dict):
            saved_args = {}
        saved_size = _positive_int(saved_args.get("local_crop_size")) or 96
        saved_crops = _valid_local_crops(saved_args.get("local_crops"))
        if saved_crops is None:
            saved_crops = 6

    if requested_local_crop_size is not None and (
        int(requested_local_crop_size) != saved_size
    ):
        raise ValueError(
            f"Cannot resume: local crop size {requested_local_crop_size} differs from "
            f"checkpoint value {saved_size}. Start a new run to change local-view "
            "settings."
        )
    if requested_local_crops is not None and int(requested_local_crops) != saved_crops:
        raise ValueError(
            f"Cannot resume: local crops {requested_local_crops} differs from "
            f"checkpoint value {saved_crops}. Start a new run to change local-view "
            "settings."
        )
    return saved_size, saved_crops


def _validate_augmentation_config(
    profile: str,
    current_config: dict[str, object],
    checkpoint: dict[str, Any] | None,
    *,
    stage: str,
) -> dict[str, object]:
    """Validate the freshly built config against checkpoint metadata.

    Returns ``current_config`` unchanged when no comparable metadata exists or
    when it matches exactly; raises on malformed metadata or any difference.
    """
    _validate_stage(stage)
    if not isinstance(current_config, dict):
        raise ValueError(
            "malformed augmentation metadata: current augmentation config must "
            "be a dict."
        )
    metadata = _checkpoint_augmentation_metadata(checkpoint)
    if metadata is None:
        return current_config
    _, saved_config = metadata
    if current_config != saved_config:
        raise ValueError(
            f"Augmentation configuration differs from checkpoint for profile "
            f"'{profile}'. Start a new run or resume with the exact saved "
            "configuration."
        )
    return current_config


_REGISTER_RESUME_KEYS = ("student", "teacher", "model_state_dict")


def _assert_encoder_state_matches(
    state_dict: dict, model: OTUFormerEncoder, label: str
) -> None:
    """Require every ``backbone.*`` key and shape of a resume state to fit.

    Only the backbone is compared, so ``center`` and the ``_pretrain_final_norm.*``
    alias remain allowed inside an encoder state dict.
    """
    expected = {
        key: value
        for key, value in model.state_dict().items()
        if key.startswith("backbone.")
    }
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
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Resume checkpoint '{label}' does not fit '{model.model_name}': "
            f"{len(missing)} missing, {len(unexpected)} unexpected, "
            f"{len(mismatched)} shape-mismatched backbone tensor(s) "
            f"(e.g. {(missing + unexpected + mismatched)[0]})."
        )


def _resume_register_plan(
    resume_ckpt: dict[str, Any], model_name: str, image_size: int, out_dim: int
) -> tuple[str, int]:
    """Resolve ``(mode, actual_count)`` for a pretraining resume.

    ``mode`` is ``"legacy"`` (no registers), ``"added"`` (four added registers)
    or ``"native"`` (the model's own registers). A registered resume requires
    explicit ``student`` and ``teacher`` states that agree on the layout, and
    every present encoder state is checked against the rebuilt backbone before
    any training encoder exists. Probes are RNG-isolated and never download
    pretrained weights.
    """
    from otuformer.utils.checkpoint import (
        preserved_rng,
        reject_noncanonical_register_keys,
        resolve_register_layout,
    )

    present = {
        key: resume_ckpt[key]
        for key in _REGISTER_RESUME_KEYS
        if isinstance(resume_ckpt.get(key), dict)
    }
    for state in present.values():
        reject_noncanonical_register_keys(state)

    with preserved_rng():
        probe = OTUFormerEncoder(
            model_name=model_name,
            out_dim=out_dim,
            return_patch_tokens=True,
            img_size=image_size,
            pretrained=False,
        )
    native = native_register_count(probe.backbone)
    recorded = recorded_register_count(resume_ckpt)
    registered = (
        (recorded is not None and recorded > 0)
        or native > 0
        or any("backbone.reg_token" in state for state in present.values())
    )
    if not registered:
        # Legacy zero-register resume keeps its historical permissive partial
        # load: the design explicitly excludes this path from the registered
        # complete-backbone requirement.
        return "legacy", 0

    missing = [key for key in ("student", "teacher") if key not in present]
    if missing:
        raise ValueError(
            "Registered resume requires explicit "
            + " and ".join(f"'{key}'" for key in missing)
            + " encoder state(s): the legacy model_state_dict fallback is a "
            "teacher copy, not a student replacement."
        )
    layouts = {
        key: resolve_register_layout(resume_ckpt, state, probe.backbone)
        for key, state in present.items()
    }
    if len(set(layouts.values())) != 1:
        raise ValueError(
            "Registered resume encoder states disagree on the register layout: "
            + json.dumps({key: list(value) for key, value in layouts.items()})
            + "."
        )
    actual, added = layouts["student"]
    mode = "added" if added == 4 else "native"

    with preserved_rng():
        expected = OTUFormerEncoder(
            model_name=model_name,
            out_dim=out_dim,
            return_patch_tokens=True,
            img_size=image_size,
            pretrained=False,
            **({"reg_tokens": 4} if mode == "added" else {}),
        )
    for key, state in present.items():
        _assert_encoder_state_matches(state, expected, key)
    return mode, actual


def _check_register_option_against_resume(
    mode: str, actual: int, register_tokens: int | None
) -> None:
    """Reject an explicit option that conflicts with the saved architecture."""
    if register_tokens is None:
        return
    if mode == "native":
        raise ValueError(
            "Cannot resume with --register-tokens: the checkpoint natively has "
            f"{actual} register tokens, which the option cannot add or replace."
        )
    if mode == "added" and register_tokens != 4:
        raise ValueError(
            "Cannot resume a four-register checkpoint with --register-tokens "
            f"{register_tokens}; resume with 4 or omit the option."
        )
    if mode == "legacy" and register_tokens != 0:
        raise ValueError(
            "Cannot add registers while resuming a zero-register checkpoint; "
            "start a new run with --register-tokens 4."
        )


def _finetune_register_plan(
    checkpoint: dict[str, Any], model_name: str, image_size: int, out_dim: int
) -> tuple[str, int]:
    """Resolve ``(mode, actual_count)`` for a fine-tune source.

    Fine-tuning always reads the ``model_state_dict`` source. A legacy
    zero-register checkpoint keeps its existing ``pretrained=True``
    construction; a registered source rebuilds without a download and must
    match every ``backbone.*`` key and shape, including under the head
    replacement's ``strict=False`` load.
    """
    from otuformer.utils.checkpoint import (
        preserved_rng,
        reject_noncanonical_register_keys,
        resolve_register_layout,
    )

    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        # The trainer reports the missing-source error in its own words.
        return "legacy", 0
    reject_noncanonical_register_keys(state)
    with preserved_rng():
        probe = OTUFormerEncoder(
            model_name=model_name,
            out_dim=out_dim,
            return_patch_tokens=True,
            img_size=image_size,
            pretrained=False,
        )
    native = native_register_count(probe.backbone)
    recorded = recorded_register_count(checkpoint)
    registered = (
        (recorded is not None and recorded > 0)
        or native > 0
        or "backbone.reg_token" in state
    )
    if not registered:
        return "legacy", 0
    actual, added = resolve_register_layout(checkpoint, state, probe.backbone)
    mode = "added" if added == 4 else "native"
    with preserved_rng():
        expected = OTUFormerEncoder(
            model_name=model_name,
            out_dim=out_dim,
            return_patch_tokens=True,
            img_size=image_size,
            pretrained=False,
            **({"reg_tokens": 4} if mode == "added" else {}),
        )
    _assert_encoder_state_matches(state, expected, "model_state_dict")
    return mode, actual
