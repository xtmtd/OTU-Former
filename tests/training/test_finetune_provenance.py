"""Pseudo-source provenance: expert path identity, eligibility, identity keys."""

from pathlib import Path

import argparse

import pandas as pd
import pytest

from otuformer.cli.finetune import (
    DERIVED_IDENTITY_KEYS,
    EXPERIMENT_IDENTITY_KEYS,
    FIXED_IDENTITY_VALUES,
    PARAM_CLASSIFICATION,
)
from otuformer.training import trainer
from otuformer.training.dataset import MetricDataset


def _write(path: Path, payload: bytes = b"bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _pseudo_source_config(ssl: Path, manifest: str, **overrides) -> dict:
    """A complete v0.9.0 finetune#1 config, as prepare_pseudo_round requires."""
    cfg = {
        "model_name": "vit_tiny_patch16_224",
        "metric_embed_dim": 16,
        "out_dim": 16,
        "embedding_head": "arcface_mlp_512",
        "loss": "arcface",
        "subcenters": None,
        "compact_weight": None,
        "compact_cap": None,
        "supcon_temperature": None,
        "freeze_ratio": 0.7,
        "finetune_lr": 1e-4,
        "effective_metric_head_lr": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 4,
        "seed": 42,
        "finetune_epochs": 1,
        "image_size": 32,
        "augmentation_profile": "none",
        "augmentation_config": {},
        "orientation_policy": "sensitive",
        "long_tail": "none",
        "optimizer_name": "adamw",
        "optimizer_groups": "v080_prototype",
        "arcface_scale": 64.0,
        "arcface_margin": 0.5,
        "train_manifest_sha256": manifest,
        "ssl_initialization_checkpoint_path": str(ssl),
        "ssl_initialization_checkpoint_sha256": trainer._sha256_file(ssl),
        "pseudo_round": 0,
        "pseudo_source_eligible": True,
    }
    cfg.update(overrides)
    return cfg


def test_expert_path_identity_distinct_files_same_bytes_eligible(tmp_path):
    root = tmp_path / "images"
    _write(root / "a.jpg", b"identical")
    _write(root / "b.jpg", b"identical")

    ok, reason = trainer._expert_path_identity(["a.jpg", "b.jpg"], root)
    assert ok is True
    assert reason is None


def test_expert_path_identity_nested_relative_eligible(tmp_path):
    root = tmp_path / "images"
    _write(root / "sub" / "a.jpg")

    ok, reason = trainer._expert_path_identity(["sub/a.jpg"], root)
    assert ok is True
    assert reason is None


def test_expert_path_identity_in_root_symlink_alias_is_duplicate(tmp_path):
    root = tmp_path / "images"
    target = _write(root / "a.jpg")
    (root / "alias.jpg").symlink_to(target)

    ok, reason = trainer._expert_path_identity(["a.jpg", "alias.jpg"], root)
    assert ok is False
    assert "duplicate expert path identity" in reason


def test_expert_path_identity_escaping_symlink_rejected(tmp_path):
    root = tmp_path / "images"
    root.mkdir(parents=True)
    outside = _write(tmp_path / "outside.jpg")
    (root / "escape.jpg").symlink_to(outside)

    ok, reason = trainer._expert_path_identity(["escape.jpg"], root)
    assert ok is False
    assert "escapes the image root" in reason


def test_expert_path_identity_missing_file_rejected(tmp_path):
    root = tmp_path / "images"
    root.mkdir(parents=True)

    ok, reason = trainer._expert_path_identity(["missing.jpg"], root)
    assert ok is False
    assert "missing expert image" in reason


def test_pseudo_source_status_requires_arcface_family_and_ssl_hash():
    eligible, reason = trainer._pseudo_source_status(
        loss="arcface", ssl_initialization_sha256="deadbeef", expert_reason=None
    )
    assert eligible is True
    assert reason is None

    eligible, reason = trainer._pseudo_source_status(
        loss="supcon", ssl_initialization_sha256="deadbeef", expert_reason=None
    )
    assert eligible is False
    assert "not ArcFace-family" in reason

    eligible, reason = trainer._pseudo_source_status(
        loss="subcenter-arcface", ssl_initialization_sha256=None, expert_reason=None
    )
    assert eligible is False
    assert "SSL initialization" in reason

    eligible, reason = trainer._pseudo_source_status(
        loss="arcface",
        ssl_initialization_sha256="deadbeef",
        expert_reason="duplicate expert path identity: 'a.jpg'",
    )
    assert eligible is False
    assert "duplicate expert path identity" in reason


def test_resolved_finetune_identity_effective_head_lr_and_fixed_constants():
    inherited = trainer._resolved_finetune_identity(
        optimizer_layout="v080_prototype",
        finetune_lr=3e-5,
        metric_head_lr=None,
        weight_decay=0.05,
        orientation_policy="sensitive",
        batch_size=8,
        finetune_epochs=3,
        long_tail="cb-drw",
    )
    assert inherited["finetune_lr"] == 3e-5
    # An omitted metric-head LR resolves to the backbone LR, not to a default.
    assert inherited["effective_metric_head_lr"] == 3e-5
    assert inherited["orientation_policy"] == "sensitive"
    assert inherited["batch_size"] == 8
    assert inherited["finetune_epochs"] == 3
    assert inherited["long_tail"] == "cb-drw"
    assert inherited["optimizer_name"] == "adamw"
    assert inherited["optimizer_groups"] == "v080_prototype"

    explicit = trainer._resolved_finetune_identity(
        optimizer_layout="v080_prototype",
        finetune_lr=3e-5,
        metric_head_lr=1e-4,
        weight_decay=1e-4,
        orientation_policy="invariant",
        batch_size=32,
        finetune_epochs=20,
        long_tail="none",
    )
    assert explicit["effective_metric_head_lr"] == 1e-4


def test_identity_key_maps_are_consistent():
    experiment_names = {
        name
        for name, kind in PARAM_CLASSIFICATION.items()
        if kind == "experiment"
    }
    # Every experiment CLI parameter maps to at least one resolved identity key.
    assert experiment_names == set(EXPERIMENT_IDENTITY_KEYS)
    derived = set(DERIVED_IDENTITY_KEYS)
    mapped = {key for keys in EXPERIMENT_IDENTITY_KEYS.values() for key in keys}
    assert mapped.isdisjoint(derived)
    assert set(FIXED_IDENTITY_VALUES) <= derived
    # The patch set is a derived (not CLI) identity attribute.
    assert "register_tokens" in derived
    assert "register_tokens" not in mapped


def _minimal_ssl_checkpoint(path: Path, **config) -> Path:
    """A readable SSL source for pseudo-identity resolution (no real weights)."""
    import torch

    payload = {
        "model_state_dict": {"backbone.cls_token": torch.zeros(1, 1, 192)},
        "config": {
            "model_name": "vit_tiny_patch16_224",
            "out_dim": 16,
            "metric_embed_dim": 16,
            "embedding_head": "arcface_mlp_512",
            "image_size": 32,
            **config,
        },
    }
    torch.save(payload, path)
    return path


def _pseudo_identity_for(ssl_checkpoint: Path, **overrides):
    resolved_loss = {
        "loss": "arcface",
        "subcenters": None,
        "compact_weight": None,
        "compact_cap": None,
        "supcon_temperature": None,
    }
    kwargs = dict(
        ssl_checkpoint=trainer._load_pseudo_source(ssl_checkpoint),
        source_cfg={"class_labels": ["a", "b"]},
        resolved_loss=resolved_loss,
        model_name="vit_tiny_patch16_224",
        metric_embed_dim=16,
        finetune_epochs=1,
        finetune_lr=1e-4,
        metric_head_lr=None,
        weight_decay=1e-4,
        freeze_ratio=0.0,
        augmentation="none",
        orientation_policy="sensitive",
        batch_size=2,
        seed=42,
        long_tail="none",
    )
    kwargs.update(overrides)
    return trainer.pseudo_cli_identity(**kwargs)


def test_pseudo_identity_carries_the_resolved_register_count(tmp_path: Path):
    """A zero-register source resolves to zero (the producer, not just the map)."""
    ssl = _minimal_ssl_checkpoint(tmp_path / "ssl.pth")

    identity = _pseudo_identity_for(ssl)

    assert identity["register_tokens"] == 0


def test_pseudo_identity_sees_a_four_register_source(tmp_path: Path, monkeypatch):
    """The count comes from the source's backbone layout, not a constant."""
    import timm as timm_module
    import torch
    from timm.models.vision_transformer import VisionTransformer

    import otuformer.training.model as model_module

    def factory(_model_name, **kwargs):
        requested = kwargs.get("reg_tokens")
        return VisionTransformer(
            img_size=32, patch_size=16, in_chans=3, embed_dim=16, depth=2,
            num_heads=2, num_classes=0, global_pool="",
            reg_tokens=0 if requested is None else int(requested),
            dynamic_img_size=True,
        )

    monkeypatch.setattr(timm_module, "create_model", factory)
    encoder = model_module.OTUFormerEncoder(
        model_name="tiny-vit", out_dim=8, pretrained=False, reg_tokens=4, img_size=32
    )
    ssl = _minimal_ssl_checkpoint(
        tmp_path / "registered.pth",
        model_name="tiny-vit",
        out_dim=8,
        metric_embed_dim=8,
        register_tokens=4,
    )
    torch.save(
        {
            "model_state_dict": encoder.state_dict(),
            "config": {
                "model_name": "tiny-vit",
                "out_dim": 8,
                "metric_embed_dim": 8,
                "embedding_head": "arcface_mlp_512",
                "image_size": 32,
                "register_tokens": 4,
            },
        },
        ssl,
    )

    identity = _pseudo_identity_for(
        ssl, model_name="tiny-vit", metric_embed_dim=8
    )

    assert identity["register_tokens"] == 4


def test_pseudo_identity_rejects_a_register_layout_mismatch():
    """finetune#2 must not mix patch sets with finetune#1's pseudo labels."""
    with pytest.raises(ValueError, match="register_tokens"):
        trainer._assert_pseudo_source_identity(
            {"register_tokens": 4}, {"register_tokens": 0}
        )
    with pytest.raises(ValueError, match="register_tokens"):
        trainer._assert_pseudo_source_identity(
            {"register_tokens": 0}, {"register_tokens": 4}
        )
    # A pre-v0.10.0 source without the key stays comparable.
    trainer._assert_pseudo_source_identity({}, {"register_tokens": 4})


def _label_fixture(tmp_path, name, labels):
    root = tmp_path / name
    root.mkdir()
    refs = [f"img_{i}.jpg" for i in range(len(labels))]
    csv = root / "labels.csv"
    pd.DataFrame({"image": refs, "label": labels}).to_csv(csv, index=False)
    return root, refs, csv


@pytest.mark.parametrize(
    "name,labels,expected_label_names,expected_class_labels",
    [
        ("string", ["A", "B"], ["A", "B"], ["A", "B"]),
        ("numeric", [1, 2], ["1", "2"], ["1", "2"]),
        ("zero_padded", ["001", "002"], ["1", "2"], ["1", "2"]),
        ("whitespace", [" A", "B "], [" A", "B "], [" A", "B "]),
        ("float_like", [1.0, 2.0], ["1.0", "2.0"], ["1.0", "2.0"]),
    ],
)
def test_label_fixtures_keep_existing_conversion(
    tmp_path, name, labels, expected_label_names, expected_class_labels
):
    """The existing label/hash path must stay byte-for-byte stable."""
    root, refs, csv = _label_fixture(tmp_path, name, labels)
    ds = MetricDataset(csv_path=csv, images_dir=root, image_size=32)

    assert ds.label_names == expected_label_names
    assert [str(label) for label in sorted(ds.class_to_idx)] == expected_class_labels

    baseline = trainer._train_manifest_sha256(ds.image_refs, ds.label_names, root)
    reversed_csv = root / "reversed.csv"
    pd.DataFrame(
        {"image": list(reversed(refs)), "label": list(reversed(labels))}
    ).to_csv(reversed_csv, index=False)
    reversed_ds = MetricDataset(
        csv_path=reversed_csv, images_dir=root, image_size=32
    )
    assert (
        trainer._train_manifest_sha256(
            reversed_ds.image_refs, reversed_ds.label_names, root
        )
        == baseline
    )


def test_missing_label_value_keeps_current_failure(tmp_path):
    """A NaN label still fails sorting exactly as before; not made tolerant."""
    root, _refs, csv = _label_fixture(tmp_path, "missing", ["A", ""])

    with pytest.raises(TypeError):
        MetricDataset(csv_path=csv, images_dir=root, image_size=32)


def test_assert_pseudo_paths_rejects_overlap_and_input_inside_out(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    inside = out / "ssl.pth"
    inside.write_bytes(b"x")

    trainer._assert_pseudo_paths(out, root, [tmp_path / "ssl.pth"])

    with pytest.raises(ValueError, match="overlap"):
        trainer._assert_pseudo_paths(root / "runs", root, [])

    with pytest.raises(ValueError, match="inside out_dir"):
        trainer._assert_pseudo_paths(out, root, [inside])


def test_resolve_pseudo_ssl_path_requires_matching_hash(tmp_path):
    ssl = tmp_path / "ssl.pth"
    ssl.write_bytes(b"weights")
    digest = trainer._sha256_file(ssl)
    cfg = {
        "ssl_initialization_checkpoint_path": str(ssl),
        "ssl_initialization_checkpoint_sha256": digest,
    }

    assert trainer._resolve_pseudo_ssl_path(cfg, None) == ssl
    assert trainer._resolve_pseudo_ssl_path(cfg, ssl) == ssl

    other = tmp_path / "other.pth"
    other.write_bytes(b"different")
    with pytest.raises(ValueError, match="does not match"):
        trainer._resolve_pseudo_ssl_path(cfg, other)
    with pytest.raises(ValueError, match="no SSL initialization"):
        trainer._resolve_pseudo_ssl_path({}, None)


def test_metric_dataset_explicit_class_order_supports_numeric_labels(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    csv = root / "labels.csv"
    pd.DataFrame({"image": ["a.jpg", "b.jpg"], "label": [2, 10]}).to_csv(
        csv, index=False
    )

    legacy = MetricDataset(csv_path=csv, images_dir=root, image_size=32)
    # The default path keeps numeric ordering: 2 before 10.
    assert [str(label) for label in legacy.class_to_idx] == ["2", "10"]

    ordered = MetricDataset(
        images_dir=root,
        image_size=32,
        rows=[{"image": "a.jpg", "label": "2"}, {"image": "b.jpg", "label": "10"}],
        class_labels=["2", "10"],
    )
    assert ordered.labels == [0, 1]
    assert [str(label) for label in ordered.class_to_idx] == ["2", "10"]


def test_prepare_pseudo_round_uses_source_top_level_class_order(tmp_path, monkeypatch):
    import torch

    from otuformer.embedding import pseudo_label as pl_module
    from otuformer.training import trainer as trainer_module

    root = tmp_path / "images"
    root.mkdir()
    refs = ["a.jpg", "b.jpg"]
    labels = ["2", "10"]
    for ref in refs:
        (root / ref).write_bytes(b"x")
    csv = tmp_path / "labels.csv"
    pd.DataFrame({"image": refs, "label": labels}).to_csv(csv, index=False)

    ssl = tmp_path / "ssl.pth"
    torch.save({"model_state_dict": {"backbone.cls_token": torch.zeros(1, 1, 8)}}, ssl)
    manifest = trainer_module._train_manifest_sha256(refs, labels, root)
    source = tmp_path / "ft1.pth"
    torch.save(
        {
            "epoch": 0,
            "config": _pseudo_source_config(ssl, manifest),
            "class_labels": ["10", "2"],
            "class_order": ["2", "10"],
        },
        source,
    )

    monkeypatch.setattr(
        pl_module,
        "generate_pseudo_rows",
        lambda **kwargs: (
            [{"image": "c.jpg", "label": "2"}],
            [{"image": "c.jpg", "accepted": True, "rejection_reasons": []}],
            {"thresholds": {"similarity_floor": 0.0, "min_gap": 0.0, "neighbors": 5}},
        ),
    )

    data = trainer_module.prepare_pseudo_round(
        pseudo_label_from=source,
        train_data=csv,
        input_images_dir=root,
        out_dir=tmp_path / "ft2",
        similarity_floor=0.0,
        min_gap=0.0,
        neighbors=5,
        long_tail="none",
        loss=None,
        checkpoint=None,
        device="cpu",
        batch_size=2,
        num_workers=0,
    )

    # Numeric classes stay in the source's saved order: 2 before 10.
    assert data["class_labels"] == ["2", "10"]

    # A class table that does not match the expert CSV is rejected, not guessed.
    bad_source = tmp_path / "bad.pth"
    torch.save(
        {
            "epoch": 0,
            "config": _pseudo_source_config(ssl, manifest),
            "class_labels": ["2"],
            "class_order": ["2"],
        },
        bad_source,
    )
    with pytest.raises(ValueError, match="class table"):
        trainer_module.prepare_pseudo_round(
            pseudo_label_from=bad_source,
            train_data=csv,
            input_images_dir=root,
            out_dir=tmp_path / "ft3",
            similarity_floor=0.0,
            min_gap=0.0,
            neighbors=5,
            long_tail="none",
            loss=None,
            checkpoint=None,
            device="cpu",
            batch_size=2,
            num_workers=0,
        )


def test_prepare_pseudo_round_rejects_incomplete_source(tmp_path):
    import torch

    from otuformer.training import trainer as trainer_module

    root = tmp_path / "images"
    root.mkdir()
    (root / "a.jpg").write_bytes(b"x")
    csv = tmp_path / "labels.csv"
    pd.DataFrame({"image": ["a.jpg"], "label": ["A"]}).to_csv(csv, index=False)
    ssl = tmp_path / "ssl.pth"
    torch.save({"model_state_dict": {}}, ssl)
    manifest = trainer_module._train_manifest_sha256(["a.jpg"], ["A"], root)
    source = tmp_path / "ft1.pth"
    torch.save(
        {
            "epoch": 0,  # one of three planned epochs completed
            "config": _pseudo_source_config(ssl, manifest, finetune_epochs=3),
            "class_labels": ["A"],
            "class_order": ["A"],
        },
        source,
    )

    with pytest.raises(ValueError, match="incomplete"):
        trainer_module.prepare_pseudo_round(
            pseudo_label_from=source,
            train_data=csv,
            input_images_dir=root,
            out_dir=tmp_path / "ft2",
            similarity_floor=0.0,
            min_gap=0.0,
            neighbors=5,
            long_tail="none",
            loss=None,
            checkpoint=None,
            device="cpu",
            batch_size=2,
            num_workers=0,
        )


def _save_pseudo_source(
    tmp_path, *, refs, labels, class_order, class_labels, epoch=0, **cfg_overrides
):
    import torch

    root = tmp_path / "images"
    root.mkdir(exist_ok=True)
    for ref in refs:
        (root / ref).write_bytes(b"x")
    csv = tmp_path / "labels.csv"
    pd.DataFrame({"image": list(refs), "label": list(labels)}).to_csv(csv, index=False)
    ssl = tmp_path / "ssl.pth"
    ssl.write_bytes(b"weights")
    manifest = trainer._train_manifest_sha256(list(refs), list(labels), root)
    cfg = _pseudo_source_config(ssl, manifest, **cfg_overrides)
    source = tmp_path / "ft1.pth"
    torch.save(
        {
            "epoch": epoch,
            "config": cfg,
            "class_labels": list(class_labels),
            "class_order": list(class_order),
        },
        source,
    )
    return root, csv, source


def _call_prepare(source, root, csv, out):
    return trainer.prepare_pseudo_round(
        pseudo_label_from=source,
        train_data=csv,
        input_images_dir=root,
        out_dir=out,
        similarity_floor=0.0,
        min_gap=0.0,
        neighbors=1,
        long_tail="none",
        loss=None,
        checkpoint=None,
        device="cpu",
        batch_size=2,
        num_workers=0,
    )


def test_prepare_pseudo_round_requires_complete_provenance(tmp_path):
    import torch

    root, csv, source = _save_pseudo_source(
        tmp_path, refs=["a.jpg"], labels=["A"], class_order=["A"], class_labels=["A"]
    )
    cfg = torch.load(source, map_location="cpu", weights_only=False)["config"]
    del cfg["freeze_ratio"]
    incomplete = tmp_path / "incomplete.pth"
    torch.save(
        {"epoch": 0, "config": cfg, "class_labels": ["A"], "class_order": ["A"]},
        incomplete,
    )

    with pytest.raises(ValueError, match="missing required provenance"):
        _call_prepare(incomplete, root, csv, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_prepare_pseudo_round_rejects_class_order_labels_disagreement(tmp_path):
    root, csv, source = _save_pseudo_source(
        tmp_path,
        refs=["a.jpg"],
        labels=["A"],
        class_order=["A"],
        class_labels=["A", "B"],
    )

    with pytest.raises(ValueError, match="disagree"):
        _call_prepare(source, root, csv, tmp_path / "out")


def test_load_pseudo_source_wraps_corrupt_checkpoint(tmp_path):
    from otuformer.embedding.pseudo_label import PseudoLabelError

    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"not a checkpoint")

    with pytest.raises(PseudoLabelError, match="cannot read pseudo source"):
        trainer._load_pseudo_source(bad)


def test_empty_acceptance_message_reports_counts_and_bins(tmp_path, monkeypatch):
    from otuformer.embedding import pseudo_label as pl_module

    root, csv, source = _save_pseudo_source(
        tmp_path, refs=["a.jpg"], labels=["A"], class_order=["A"], class_labels=["A"]
    )
    summary = {
        "candidate_count": 5,
        "accepted_count": 0,
        "acceptance_rate": 0.0,
        "rejection_reason_counts": {"floor": 5},
        "score_quantiles": {"top1": {"p50": 0.1, "p90": 0.2, "p99": 0.3}},
        "counterfactual_pass_counts": {"removed_floor": 2},
        "seed_count_bins": {
            "none": {"candidates": 5, "accepted": 0, "acceptance_rate": 0.0}
        },
    }
    monkeypatch.setattr(
        pl_module, "generate_pseudo_rows", lambda **kwargs: ([], [], summary)
    )

    with pytest.raises(ValueError) as excinfo:
        _call_prepare(source, root, csv, tmp_path / "out")

    message = str(excinfo.value)
    assert "candidate count: 5" in message
    assert "acceptance rate: 0.0" in message
    assert "seed-count bins" in message
    assert not (tmp_path / "out").exists()


def test_resume_drift_warnings_are_non_blocking(tmp_path, capsys):
    root, csv, _source = _save_pseudo_source(
        tmp_path, refs=["a.jpg"], labels=["A"], class_order=["A"], class_labels=["A"]
    )
    (root / "cand.jpg").write_bytes(b"y")
    out = tmp_path / "out"
    out.mkdir()
    (out / "pseudo_labels.csv").write_text("image,accepted\n", encoding="utf-8")
    args = argparse.Namespace(input_images_dir=str(root), train_data=str(csv))
    cfg = {
        "input_images_dir": str(tmp_path / "old_root"),
        "pseudo_candidate_pool_sha256": "stale",
        "pseudo_labels_sha256": "stale",
    }

    trainer._warn_pseudo_resume_drift(cfg, args, out)

    output = capsys.readouterr().out
    assert "different image root" in output
    assert "candidate pool changed" in output
    assert "pseudo_labels.csv changed" in output


def test_resume_drift_warns_on_missing_labels_csv(tmp_path, capsys):
    root, csv, _source = _save_pseudo_source(
        tmp_path, refs=["a.jpg"], labels=["A"], class_order=["A"], class_labels=["A"]
    )
    args = argparse.Namespace(input_images_dir=str(root), train_data=str(csv))

    trainer._warn_pseudo_resume_drift({}, args, tmp_path / "out")

    assert "pseudo_labels.csv is missing" in capsys.readouterr().out


# --- v0.10.0 fine-tune register layout ---------------------------------------


def _tiny_backbone_state(native_regs=0, register_count=None, class_token=True):
    """A full ``backbone.*`` state dict from a tiny timm VisionTransformer."""
    import torch
    from timm.models.vision_transformer import VisionTransformer

    torch.manual_seed(0)
    model = VisionTransformer(
        img_size=32, patch_size=16, in_chans=3, embed_dim=16, depth=2,
        num_heads=2, mlp_ratio=2.0, num_classes=0, global_pool="",
        class_token=class_token,
        reg_tokens=native_regs if register_count is None else register_count,
        dynamic_img_size=True,
    )
    return {f"backbone.{key}": value for key, value in model.state_dict().items()}


def _install_tiny_factory(monkeypatch, native_regs=0, class_token=True):
    import torch
    import timm as timm_module
    from timm.models.vision_transformer import VisionTransformer

    def factory(_model_name, **kwargs):
        requested = kwargs.get("reg_tokens")
        count = native_regs if requested is None else int(requested)
        torch.manual_seed(0)
        return VisionTransformer(
            img_size=32, patch_size=16, in_chans=3, embed_dim=16, depth=2,
            num_heads=2, mlp_ratio=2.0, num_classes=0, global_pool="",
            class_token=class_token, reg_tokens=count, dynamic_img_size=True,
        )

    monkeypatch.setattr(timm_module, "create_model", factory)


def test_finetune_register_plan_detects_added_four(monkeypatch):
    _install_tiny_factory(monkeypatch)
    checkpoint = {
        "model_state_dict": _tiny_backbone_state(register_count=4),
        "config": {"register_tokens": 4},
    }
    assert trainer._finetune_register_plan(checkpoint, "tiny", 32, 8) == ("added", 4)


def test_finetune_register_plan_detects_native_registers(monkeypatch):
    _install_tiny_factory(monkeypatch, native_regs=2)
    checkpoint = {"model_state_dict": _tiny_backbone_state(native_regs=2)}
    assert trainer._finetune_register_plan(checkpoint, "tiny", 32, 8) == ("native", 2)


def test_finetune_register_plan_keeps_the_legacy_zero_register_path(monkeypatch):
    """A legacy source keeps its permissive ``pretrained=True`` construction."""
    import torch

    _install_tiny_factory(monkeypatch)
    checkpoint = {"model_state_dict": _tiny_backbone_state()}
    assert trainer._finetune_register_plan(checkpoint, "tiny", 32, 8) == ("legacy", 0)

    # Missing backbone keys and a missing source are never tightened here.
    incomplete = _tiny_backbone_state()
    incomplete.pop("backbone.blocks.0.attn.qkv.weight")
    assert trainer._finetune_register_plan(
        {"model_state_dict": incomplete}, "tiny", 32, 8
    ) == ("legacy", 0)
    assert trainer._finetune_register_plan({}, "tiny", 32, 8) == ("legacy", 0)

    # A wrongly sized register tensor is still not legacy.
    checkpoint = {
        "model_state_dict": {
            "backbone.reg_token": torch.zeros(1, 3, 16),
            "backbone.cls_token": torch.zeros(1, 1, 16),
        }
    }
    with pytest.raises(ValueError, match="reg_token"):
        trainer._finetune_register_plan(checkpoint, "tiny", 32, 8)


def test_finetune_register_plan_rejects_incomplete_registered_backbone(monkeypatch):
    _install_tiny_factory(monkeypatch)
    state = _tiny_backbone_state(register_count=4)
    state.pop("backbone.blocks.0.attn.qkv.weight")
    checkpoint = {"model_state_dict": state, "config": {"register_tokens": 4}}

    with pytest.raises(ValueError, match="attn.qkv.weight"):
        trainer._finetune_register_plan(checkpoint, "tiny", 32, 8)


def test_finetune_register_plan_rejects_a_prefixed_register_key(monkeypatch):
    import torch

    _install_tiny_factory(monkeypatch)
    checkpoint = {
        "model_state_dict": {"_orig_mod.backbone.reg_token": torch.zeros(1, 4, 16)},
        "config": {"register_tokens": 4},
    }

    with pytest.raises(ValueError, match="backbone.reg_token"):
        trainer._finetune_register_plan(checkpoint, "tiny", 32, 8)


def test_finetune_register_plan_rejects_no_cls_native_registers(monkeypatch):
    _install_tiny_factory(monkeypatch, native_regs=4, class_token=False)
    checkpoint = {
        "model_state_dict": _tiny_backbone_state(native_regs=4, class_token=False)
    }

    with pytest.raises(ValueError, match="no CLS token"):
        trainer._finetune_register_plan(checkpoint, "tiny", 32, 8)


def test_preflight_finetune_source_uses_recorded_model_name(monkeypatch):
    _install_tiny_factory(monkeypatch)
    checkpoint = {
        "model_state_dict": _tiny_backbone_state(),
        "config": {"model_name": "tiny", "out_dim": 8},
    }
    assert trainer.preflight_finetune_source(checkpoint, "vit_tiny_patch16_224") == (
        "legacy",
        0,
    )
