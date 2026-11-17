"""Backbone-only new-run initialization from a recognized OTU SSL checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pytest
import torch

from otuformer.training import pretrain_loop as pretrain_loop_module
from otuformer.utils import checkpoint as checkpoint_module

SIZE = 32


def _write_ssl(tmp_path, name, *, teacher_scale=1.0, student_scale=2.0):
    """A recognized SSL payload whose teacher and student branches differ."""
    from otuformer.training.model import OTUFormerEncoder

    torch.manual_seed(0)
    encoder = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224",
        out_dim=16,
        return_patch_tokens=True,
        img_size=SIZE,
        pretrained=False,
        reg_tokens=None,
    )
    teacher = {k: v.clone() * teacher_scale for k, v in encoder.state_dict().items()}
    student = {k: v.clone() * student_scale for k, v in encoder.state_dict().items()}
    payload = {
        "teacher": teacher,
        "student": student,
        "config": {
            "model_name": "vit_tiny_patch16_224",
            "image_size": SIZE,
            "out_dim": 16,
            "ssl_method": "relational",
            "ssl_head": "projection_mlp_2048",
            "register_tokens": 0,
        },
        "iteration": 5,
        "epoch": 0,
    }
    path = tmp_path / name
    checkpoint_module.save_checkpoint(payload, path)
    return path


def test_selected_teacher_branch_only_uses_teacher(tmp_path):
    path = _write_ssl(tmp_path, "ssl.pth")
    prepared = checkpoint_module.prepare_pretrain_initialization(
        path, source_key="teacher"
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for name, tensor in prepared.backbone_state.items():
        torch.testing.assert_close(
            tensor, payload["teacher"][name], rtol=0, atol=0
        )


def test_missing_selected_branch_never_falls_back(tmp_path):
    path = _write_ssl(tmp_path, "ssl.pth")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["student"]
    checkpoint_module.save_checkpoint(payload, path)
    with pytest.raises((ValueError, KeyError), match="student"):
        checkpoint_module.prepare_pretrain_initialization(path, source_key="student")


def test_metric_and_raw_payloads_are_rejected(tmp_path):
    path = tmp_path / "metric.pth"
    checkpoint_module.save_checkpoint(
        {
            "model_state_dict": {"projector.net.0.weight": torch.zeros(512, 4)},
            "config": {"model_name": "vit_tiny_patch16_224", "image_size": SIZE},
        },
        path,
    )
    # No selected branch at all: rejected instead of silently substituting.
    with pytest.raises((ValueError, KeyError)):
        checkpoint_module.prepare_pretrain_initialization(path, source_key="teacher")

    # A metric payload that does carry a branch is rejected by SSL identity.
    metric = {
        "teacher": {"backbone.cls_token": torch.zeros(1, 1, 4)},
        "config": {"model_name": "vit_tiny_patch16_224", "image_size": SIZE},
    }
    checkpoint_module.save_checkpoint(metric, path)
    with pytest.raises(ValueError, match="SSL|identity|backbone"):
        checkpoint_module.prepare_pretrain_initialization(path, source_key="teacher")


def test_hash_is_of_the_loaded_bytes(tmp_path):
    path = _write_ssl(tmp_path, "ssl.pth")
    prepared = checkpoint_module.prepare_pretrain_initialization(
        path, source_key="teacher"
    )
    expected = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert prepared.sha256 == expected


def test_initialize_loads_exact_backbone_keys(tmp_path):
    from otuformer.training.model import OTUFormerEncoder

    path = _write_ssl(tmp_path, "ssl.pth")
    prepared = checkpoint_module.prepare_pretrain_initialization(
        path, source_key="teacher"
    )
    torch.manual_seed(1)
    target = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224",
        out_dim=16,
        return_patch_tokens=True,
        img_size=SIZE,
        pretrained=False,
        reg_tokens=None,
    )
    before = {k: v.clone() for k, v in target.state_dict().items()}
    checkpoint_module.initialize_pretrain_backbone(target, prepared)
    after = target.state_dict()
    for name, tensor in prepared.backbone_state.items():
        torch.testing.assert_close(after[name], tensor, rtol=0, atol=0)
    # Non-backbone state keeps the fresh initialization.
    for name, tensor in before.items():
        if not name.startswith("backbone."):
            torch.testing.assert_close(after[name], tensor, rtol=0, atol=0)


def test_architecture_and_identity_are_recorded(tmp_path):
    path = _write_ssl(tmp_path, "ssl.pth")
    prepared = checkpoint_module.prepare_pretrain_initialization(
        path, source_key="student"
    )
    assert prepared.source_key == "student"
    assert prepared.architecture.model_name == "vit_tiny_patch16_224"
    assert prepared.image_size == SIZE
    assert prepared.register_tokens == 0
    assert prepared.source_identity is not None


def _args(tmp_path, name, **overrides):
    base = dict(
        seed=42,
        cpus=0,
        device="cpu",
        out_dir=str(tmp_path / name),
        model_name="vit_tiny_patch16_224",
        register_tokens=None,
        out_dim=16,
        ssl_method="relational",
        ssl_method_explicit=False,
        koleo_weight=0.0,
        koleo_weight_explicit=False,
        max_epochs=1,
        lr=1e-3,
        weight_decay=0.05,
        warmup_epochs=0,
        augmentation=None,
        orientation_policy=None,
        global_crop_size=SIZE,
        local_crop_size=96,
        local_crops=6,
        mask_ratio="auto",
        lambda_local=1.5,
        lambda_mask=1.0,
        patch_loss="none",
        masking_strategy="random",
        teacher_momentum=0.995,
        teacher_momentum_end=0.999,
        student_temp=0.1,
        teacher_temp_start=0.04,
        teacher_temp_end=0.07,
        teacher_temp_warmup_ratio=None,
        relational_teacher_views="weak",
        relational_center_source=None,
        relational_pairing="cross",
        relational_queue_size=0,
        resume="",
        log_every_n_steps=1000,
        save_every_epochs=1,
        keep_last_checkpoints=1,
        visualize_data="",
        extract_size=None,
        metrics_sample_size=10,
        umap_n_neighbors=5,
        umap_min_dist=0.1,
        umap_metric="cosine",
        visualize_class_number=5,
        compute_embedding_metrics=False,
        batch_size=2,
        num_workers=0,
        coordinate_weight=0.0,
        coordinate_weight_explicit=False,
        dense_query_points=49,
        dense_query_points_explicit=False,
        ranking_weight=0.0,
        ranking_weight_explicit=False,
        ranking_reference_points=16,
        ranking_reference_points_explicit=False,
        init_checkpoint="",
        init_source=None,
        init_source_explicit=False,
    )
    import pandas as pd
    from PIL import Image

    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    for index in range(4):
        Image.new("RGB", (64, 64), color=(index * 30, 5, 5)).save(
            images / f"img_{index}.jpg"
        )
    frame = pd.DataFrame({"image": [f"img_{i}.jpg" for i in range(4)]})
    csv = tmp_path / "data.csv"
    frame.to_csv(csv, index=False)
    base.update(train_data=str(csv), input_images_dir=str(images))
    base.update(overrides)
    return argparse.Namespace(**base)


def test_init_checkpoint_transfers_and_starts_at_zero(tmp_path):
    source = _write_ssl(tmp_path, "ssl.pth")
    args = _args(tmp_path, "initrun", init_checkpoint=str(source))
    pretrain_loop_module.run_pretrain(args)

    saved = torch.load(
        Path(args.out_dir) / "SSL_latest.pth", map_location="cpu", weights_only=False
    )
    provenance = saved["config"]["pretrain_initialization"]
    assert provenance["sha256"] == hashlib.sha256(Path(source).read_bytes()).hexdigest()
    assert provenance["source_key"] == "teacher"
    assert provenance["transfer"] == "backbone-only"
    assert provenance["version"] == 1
    # The transfer is backbone-only: the source projector is not the new head.
    payload = torch.load(source, map_location="cpu", weights_only=False)
    assert not torch.equal(
        saved["model_state_dict"]["projector.net.0.weight"],
        payload["teacher"]["projector.net.0.weight"],
    )


def test_init_with_resume_is_rejected(tmp_path):
    source = _write_ssl(tmp_path, "ssl.pth")
    args = _args(
        tmp_path,
        "both",
        init_checkpoint=str(source),
        resume=str(source),
    )
    with pytest.raises(ValueError, match="init-checkpoint"):
        pretrain_loop_module.run_pretrain(args)


def test_init_source_without_checkpoint_is_rejected(tmp_path):
    args = _args(tmp_path, "nosource", init_source="student", init_source_explicit=True)
    with pytest.raises(ValueError, match="init-checkpoint"):
        pretrain_loop_module.run_pretrain(args)


def test_resume_does_not_require_the_source_file(tmp_path):
    source = _write_ssl(tmp_path, "ssl.pth")
    args = _args(tmp_path, "later", init_checkpoint=str(source))
    pretrain_loop_module.run_pretrain(args)

    latest = Path(args.out_dir) / "SSL_latest.pth"
    source.unlink()
    resumed = _args(
        tmp_path, "later", max_epochs=2, resume=str(latest)
    )
    pretrain_loop_module.run_pretrain(resumed)
    saved = torch.load(
        Path(resumed.out_dir) / "SSL_latest.pth", map_location="cpu", weights_only=False
    )
    assert saved["config"]["pretrain_initialization"]["source_key"] == "teacher"


def test_init_failure_leaves_output_untouched(tmp_path):
    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"not a checkpoint")
    out_dir = tmp_path / "rejected"
    out_dir.mkdir()
    sentinel = out_dir / "keep.txt"
    sentinel.write_text("original")
    args = _args(tmp_path, "rejected", init_checkpoint=str(bad))
    with pytest.raises(Exception):
        pretrain_loop_module.run_pretrain(args)
    assert sentinel.read_text() == "original"


def test_sizes_and_architecture_are_inherited_from_the_source(tmp_path):
    """An omitted size/model resolves from the source instead of the CLI default."""
    from otuformer.training.pretrain_recipe import resolve_pretrain_settings

    source = _write_ssl(tmp_path, "ssl.pth")
    prepared = checkpoint_module.prepare_pretrain_initialization(
        source, source_key="teacher"
    )
    inherited = _args(
        tmp_path, "inherit", init_checkpoint=str(source), global_crop_size=None
    )
    settings = resolve_pretrain_settings(inherited, None, initialization=prepared)
    assert settings.model_name == "vit_tiny_patch16_224"
    assert settings.global_crop_size == int(prepared.image_size)
    assert prepared.register_tokens == 0


def test_initialized_run_rejects_a_conflicting_backbone_or_size(tmp_path):
    from otuformer.training.pretrain_recipe import resolve_pretrain_settings

    source = _write_ssl(tmp_path, "ssl.pth")
    prepared = checkpoint_module.prepare_pretrain_initialization(
        source, source_key="teacher"
    )
    other_model = _args(tmp_path, "conflict", init_checkpoint=str(source))
    other_model.model_name = "vit_small_patch16_224"
    other_model.model_name_explicit = True
    with pytest.raises(ValueError, match="cannot change the backbone"):
        resolve_pretrain_settings(other_model, None, initialization=prepared)

    # The CLI default value is not a request: omitting --model-name must inherit
    # the source architecture even when it differs from the default.
    omitted = _args(tmp_path, "conflict", init_checkpoint=str(source))
    omitted.model_name = "vit_tiny_patch16_224"
    omitted.model_name_explicit = False
    settings = resolve_pretrain_settings(omitted, None, initialization=prepared)
    assert settings.model_name == prepared.architecture.model_name

    other_size = _args(
        tmp_path, "conflict", init_checkpoint=str(source), global_crop_size=SIZE * 2
    )
    with pytest.raises(ValueError, match="cannot change the input size"):
        resolve_pretrain_settings(other_size, None, initialization=prepared)


def test_omitted_model_name_inherits_a_non_default_source(tmp_path):
    """A source smaller/larger than the CLI default still initializes a new run."""
    from otuformer.training.model import OTUFormerEncoder

    torch.manual_seed(0)
    encoder = OTUFormerEncoder(
        model_name="vit_small_patch16_224",
        out_dim=16,
        return_patch_tokens=True,
        img_size=SIZE,
        pretrained=False,
        reg_tokens=None,
    )
    source = tmp_path / "small.pth"
    checkpoint_module.save_checkpoint(
        {
            "teacher": encoder.state_dict(),
            "student": encoder.state_dict(),
            "config": {
                "model_name": "vit_small_patch16_224",
                "image_size": SIZE,
                "out_dim": 16,
                "ssl_method": "relational",
                "ssl_head": "projection_mlp_2048",
                "register_tokens": 0,
            },
            "iteration": 0,
            "epoch": 0,
        },
        source,
    )
    # The caller's default model name is the tiny backbone and was not requested.
    args = _args(tmp_path, "inherit-small", init_checkpoint=str(source))
    args.model_name_explicit = False
    pretrain_loop_module.run_pretrain(args)
    saved = torch.load(
        Path(args.out_dir) / "SSL_latest.pth", map_location="cpu", weights_only=False
    )
    assert saved["config"]["model_name"] == "vit_small_patch16_224"



def test_historical_payload_without_model_metadata_uses_the_caller_fallback(tmp_path):
    """Recorded metadata wins; an absent name falls back to the caller's model."""
    path = _write_ssl(tmp_path, "legacy.pth")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["config"]["model_name"]
    checkpoint_module.save_checkpoint(payload, path)
    with pytest.raises(ValueError, match="no model name"):
        checkpoint_module.prepare_pretrain_initialization(path, source_key="teacher")
    prepared = checkpoint_module.prepare_pretrain_initialization(
        path, source_key="teacher", default_model_name="vit_tiny_patch16_224"
    )
    assert prepared.architecture.model_name == "vit_tiny_patch16_224"
    assert prepared.architecture.model_name_from_metadata is False


def test_initialized_run_never_requests_pretrained_weights(tmp_path, monkeypatch):
    """The complete source backbone makes a timm download pointless."""
    calls: list[object] = []
    real = pretrain_loop_module.OTUFormerEncoder

    def spy(*args, **kwargs):
        calls.append(kwargs.get("pretrained"))
        return real(*args, **kwargs)

    monkeypatch.setattr(pretrain_loop_module, "OTUFormerEncoder", spy)
    source = _write_ssl(tmp_path, "ssl.pth")
    args = _args(tmp_path, "noweight", init_checkpoint=str(source))
    pretrain_loop_module.run_pretrain(args)
    assert calls and all(value is False for value in calls)


def test_init_inherits_the_source_size_when_the_option_is_omitted(tmp_path):
    source = _write_ssl(tmp_path, "ssl.pth")
    args = _args(
        tmp_path, "inherit-run", init_checkpoint=str(source), global_crop_size=None
    )
    pretrain_loop_module.run_pretrain(args)
    saved = torch.load(
        Path(args.out_dir) / "SSL_latest.pth", map_location="cpu", weights_only=False
    )
    assert int(saved["config"]["image_size"]) == SIZE


def test_incomplete_source_backbone_fails_at_preparation(tmp_path):
    """A source missing backbone tensors must fail before any load or transfer."""
    path = _write_ssl(tmp_path, "partial.pth")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    removed = sorted(
        key for key in payload["teacher"] if key.startswith("backbone.")
    )[0]
    del payload["teacher"][removed]
    checkpoint_module.save_checkpoint(payload, path)
    with pytest.raises(ValueError, match="complete backbone"):
        checkpoint_module.prepare_pretrain_initialization(path, source_key="teacher")


@pytest.mark.parametrize(
    "damage", ["nan_projector", "projector_width", "missing_projector_bias", "backbone_overflow"]
)
def test_corrupt_source_branch_is_rejected_at_preparation(tmp_path, damage):
    """Only a complete, dtype-finite selected branch is a valid source."""
    path = _write_ssl(tmp_path, "damaged.pth")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    branch = payload["teacher"]
    projector_keys = sorted(key for key in branch if key.startswith("projector."))
    if damage == "nan_projector":
        key = next(key for key in projector_keys if key.endswith("bias"))
        branch[key] = torch.full_like(branch[key], float("nan"))
    elif damage == "projector_width":
        key = next(key for key in projector_keys if key.endswith("weight"))
        rows, columns = branch[key].shape
        branch[key] = torch.zeros(rows, columns + 1)
    elif damage == "missing_projector_bias":
        key = next(key for key in projector_keys if key.endswith("bias"))
        del branch[key]
    else:
        key = next(key for key in branch if key.startswith("backbone."))
        branch[key] = branch[key].to(torch.float64) * 0.0 + 1e100
    checkpoint_module.save_checkpoint(payload, path)
    with pytest.raises(ValueError):
        checkpoint_module.prepare_pretrain_initialization(path, source_key="teacher")


def test_historical_projector_layout_is_accepted_after_normalization(tmp_path):
    """``projector.<i>.*`` is a recognized historical layout, not a missing head."""
    path = _write_ssl(tmp_path, "legacy-projector.pth")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("teacher", "student"):
        branch = payload[key]
        renamed = {}
        for name, value in branch.items():
            parts = name.split(".")
            if len(parts) > 2 and parts[0] == "projector" and parts[1] == "net":
                name = ".".join((parts[0], *parts[2:]))
            renamed[name] = value
        payload[key] = renamed
    checkpoint_module.save_checkpoint(payload, path)
    prepared = checkpoint_module.prepare_pretrain_initialization(
        path, source_key="teacher"
    )
    assert prepared.backbone_state
    # The legacy names are not laundered into the new payload: normalization is
    # applied to a copy for validation only.
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    assert any(key.startswith("projector.0.") for key in reloaded["teacher"])


@pytest.mark.parametrize("damage", ["nan_canonical", "wrong_width_canonical"])
def test_conflicting_legacy_and_canonical_projector_keys_are_rejected(
    tmp_path, damage
):
    """Two spellings of one tensor is ambiguous, not a license to pick one."""
    path = _write_ssl(tmp_path, "dual-spelling.pth")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    branch = payload["teacher"]
    canonical = {
        name: value for name, value in branch.items()
        if name.startswith("projector.net.")
    }
    legacy = {
        ".".join((name.split(".")[0], *name.split(".")[2:])): value.detach().clone()
        for name, value in canonical.items()
    }
    if damage == "nan_canonical":
        key = next(name for name in canonical if name.endswith(".bias"))
        branch[key] = torch.full_like(branch[key], float("nan"))
    else:
        key = next(name for name in canonical if name.endswith(".weight"))
        rows, columns = branch[key].shape
        branch[key] = torch.zeros(rows, columns + 1)
    branch.update(legacy)  # a healthy alias for every canonical tensor
    checkpoint_module.save_checkpoint(payload, path)
    with pytest.raises(ValueError, match="ambiguous"):
        checkpoint_module.prepare_pretrain_initialization(path, source_key="teacher")
