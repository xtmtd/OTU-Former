"""Shared relational/DINO KoLeo: applicability, bundle resume and metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

#: ``src/otuformer/utils/checkpoint.py`` as of commit ``84f845e``. Pinned as a
#: test resource so the compatibility boundary is checkable without Git.
BASELINE_READER = Path(__file__).with_name("baseline_readers") / "checkpoint_84f845e.py"
BASELINE_READER_SHA256 = (
    "3e44228e832c0bde4f70c18119ae5ae4a16e8610ef9b0345ef6c3e0e304db2ac"
)

from otuformer.training import finetune_loop as finetune_loop_module
from otuformer.training import pretrain_recipe
from otuformer.training import pretrain_loop as trainer
from otuformer.training.pretrain_recipe import (
    KOLEO_EPSILON_KEYS,
    KOLEO_FIXED_CONFIG,
    KOLEO_STORED_KEYS,
    KOLEO_WEIGHT_KEY,
    resolve_pretrain_ssl_config,
)


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        ssl_method="relational",
        ssl_method_explicit=True,
        out_dim=256,
        out_dim_explicit=False,
        model_name="vit_tiny_patch16_224",
        model_name_explicit=False,
        koleo_weight=0.0,
        koleo_weight_explicit=False,
        teacher_temp_warmup_ratio=0.7,
        teacher_temp_warmup_ratio_explicit=False,
        norm_bias_weight_decay=True,
        norm_bias_weight_decay_explicit=False,
        batch_size=4,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _relational_payload(config: dict | None = None, *, out_dim: int = 256) -> dict:
    """A resumable relational payload built from a real encoder pair."""
    from otuformer.training.model import OTUFormerEncoder

    encoder = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224",
        out_dim=out_dim,
        return_patch_tokens=True,
        img_size=32,
        pretrained=False,
    )
    state = encoder.state_dict()
    payload = {
        "config": {
            "ssl_method": "relational",
            "ssl_head": "projection_mlp_2048",
            "out_dim": out_dim,
            "model_name": "vit_tiny_patch16_224",
            "augmentation_config": {"global_crop": {"size": 32}},
        },
        "student": state,
        "teacher": state,
        "model_state_dict": state,
        "center": torch.zeros(1, out_dim),
        "args": {"max_epochs": 4, "global_crop_size": 32},
        "epoch": 0,
        "iteration": 2,
    }
    payload["config"].update(config or {})
    return payload


def test_relational_effective_config_carries_the_shared_koleo_bundle():
    effective = resolve_pretrain_ssl_config(_args(), None)

    assert effective[KOLEO_WEIGHT_KEY] == 0.0
    for name, value in KOLEO_FIXED_CONFIG.items():
        assert effective[name] == value


@pytest.mark.parametrize("weight", [True, -0.5, float("nan"), float("inf")])
def test_relational_rejects_an_invalid_koleo_weight(weight):
    with pytest.raises(ValueError, match="koleo_weight"):
        resolve_pretrain_ssl_config(
            _args(koleo_weight=weight, koleo_weight_explicit=True), None
        )


def test_relational_resume_without_a_bundle_stays_disabled():
    """Absence is relational's canonical zero encoding for old and new runs."""
    effective = resolve_pretrain_ssl_config(_args(), _relational_payload())

    assert effective[KOLEO_WEIGHT_KEY] == 0.0


@pytest.mark.parametrize("explicit", [True, False])
def test_relational_resume_cannot_enable_koleo_from_an_absent_bundle(explicit):
    """The recorded state decides, so a direct caller cannot opt in either."""
    with pytest.raises(ValueError, match="Start a new run"):
        resolve_pretrain_ssl_config(
            _args(koleo_weight=0.1, koleo_weight_explicit=explicit),
            _relational_payload(),
        )


def test_relational_resume_rejects_a_partial_bundle():
    partial = dict(KOLEO_FIXED_CONFIG)
    partial[KOLEO_WEIGHT_KEY] = 0.1
    partial.pop(KOLEO_EPSILON_KEYS[-1])

    with pytest.raises(ValueError, match="partial KoLeo bundle"):
        resolve_pretrain_ssl_config(_args(), _relational_payload(partial))


def test_relational_resume_treats_an_explicit_null_as_damaged_metadata():
    """A present key with a null value is not the "absent" encoding."""
    with pytest.raises(ValueError, match="partial KoLeo bundle"):
        resolve_pretrain_ssl_config(
            _args(), _relational_payload({KOLEO_WEIGHT_KEY: None})
        )


def test_relational_resume_rejects_a_complete_bundle_with_zero_weight():
    """Relational encodes disabled by absence, so a zero-weight bundle is bogus."""
    bundle = {KOLEO_WEIGHT_KEY: 0.0, **KOLEO_FIXED_CONFIG}

    with pytest.raises(ValueError, match="no KoLeo keys at all"):
        resolve_pretrain_ssl_config(_args(), _relational_payload(bundle))


@pytest.mark.parametrize("weight", [True, -0.1, float("nan")])
def test_relational_resume_rejects_an_invalid_recorded_weight(weight):
    bundle = {KOLEO_WEIGHT_KEY: weight, **KOLEO_FIXED_CONFIG}

    with pytest.raises(ValueError):
        resolve_pretrain_ssl_config(_args(), _relational_payload(bundle))


@pytest.mark.parametrize("epsilon", [0.0, -1e-8, float("inf"), "x", True])
def test_relational_resume_rejects_an_invalid_recorded_epsilon(epsilon):
    bundle = {KOLEO_WEIGHT_KEY: 0.1, **KOLEO_FIXED_CONFIG}
    bundle[KOLEO_EPSILON_KEYS[0]] = epsilon

    with pytest.raises(ValueError):
        resolve_pretrain_ssl_config(_args(), _relational_payload(bundle))


def test_relational_resume_inherits_a_recorded_positive_weight():
    bundle = {KOLEO_WEIGHT_KEY: 0.1, **KOLEO_FIXED_CONFIG}

    effective = resolve_pretrain_ssl_config(_args(), _relational_payload(bundle))

    assert effective[KOLEO_WEIGHT_KEY] == pytest.approx(0.1)


def test_relational_resume_rejects_a_changed_weight():
    bundle = {KOLEO_WEIGHT_KEY: 0.1, **KOLEO_FIXED_CONFIG}

    with pytest.raises(ValueError, match="requires a new run"):
        resolve_pretrain_ssl_config(
            _args(koleo_weight=0.2, koleo_weight_explicit=True),
            _relational_payload(bundle),
        )


def test_relational_resume_accepts_an_equal_explicit_weight():
    bundle = {KOLEO_WEIGHT_KEY: 0.1, **KOLEO_FIXED_CONFIG}

    effective = resolve_pretrain_ssl_config(
        _args(koleo_weight=0.1, koleo_weight_explicit=True),
        _relational_payload(bundle),
    )

    assert effective[KOLEO_WEIGHT_KEY] == pytest.approx(0.1)


def test_relational_resume_rejects_a_foreign_epsilon():
    bundle = {KOLEO_WEIGHT_KEY: 0.1, **KOLEO_FIXED_CONFIG}
    bundle[KOLEO_EPSILON_KEYS[0]] = 1e-6

    with pytest.raises(ValueError, match="cannot change silently"):
        resolve_pretrain_ssl_config(_args(), _relational_payload(bundle))


def test_relational_resume_still_rejects_dino_objective_state():
    payload = _relational_payload({KOLEO_WEIGHT_KEY: 0.1, **KOLEO_FIXED_CONFIG})
    payload["dino_objective"] = {}

    with pytest.raises(ValueError):
        resolve_pretrain_ssl_config(_args(), payload)


def test_common_koleo_keys_alone_do_not_establish_dino_identity():
    from otuformer.utils.checkpoint import (
        PROJECTION_EMBEDDING_HEAD,
        resolve_checkpoint_ssl_identity,
    )

    payload = _relational_payload({KOLEO_WEIGHT_KEY: 0.0, **KOLEO_FIXED_CONFIG})

    # KoLeo keys alone must not turn the payload into DINO metadata.
    assert resolve_checkpoint_ssl_identity(payload, payload["model_state_dict"]) == (
        "relational",
        PROJECTION_EMBEDDING_HEAD,
    )


def test_display_projects_koleo_for_relational_but_storage_omits_it():
    effective = resolve_pretrain_ssl_config(_args(), None)

    display = pretrain_recipe._ssl_display_config(effective)
    assert display[KOLEO_WEIGHT_KEY] == 0.0
    for name, value in KOLEO_FIXED_CONFIG.items():
        assert display[name] == value
    # Display is not persistence.
    assert display["dino_prototypes"] is None


def test_zero_weight_relational_run_omits_the_four_saved_keys(tmp_path, capsys):
    saved = _train_relational(tmp_path, koleo_weight=0.0)

    for name in KOLEO_STORED_KEYS:
        assert name not in saved["config"], name
    out = capsys.readouterr().out
    assert '"koleo_weight": 0.0' in out
    from otuformer.training.runtime import _resolve_device

    assert f"Resolved device: {_resolve_device('auto')}" in out


def _train_relational(tmp_path, *, koleo_weight: float, **overrides) -> dict:
    from PIL import Image

    images = tmp_path / "images"
    images.mkdir(parents=True, exist_ok=True)
    for index in range(4):
        Image.new("RGB", (32, 32), (index * 30, 10, 90)).save(images / f"{index}.png")
    manifest = tmp_path / "data.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image"])
        for index in range(4):
            writer.writerow([f"{index}.png"])

    args = argparse.Namespace(
        train_data=str(manifest),
        input_images_dir=str(images),
        out_dir=str(tmp_path / "out"),
        overwrite=False,
        model_name="vit_tiny_patch16_224",
        register_tokens=None,
        model_name_explicit=False,
        out_dim=8,
        out_dim_explicit=True,
        ssl_method="relational",
        ssl_method_explicit=True,
        koleo_weight=koleo_weight,
        koleo_weight_explicit=True,
        max_epochs=1,
        lr=5e-4,
        weight_decay=0.05,
        warmup_epochs=0,
        no_norm_bias_weight_decay=True,
        no_norm_bias_weight_decay_explicit=False,
        augmentation="global-barcode",
        orientation_policy="sensitive",
        global_crop_size=32,
        local_crop_size=96,
        local_crops=0,
        mask_ratio="auto",
        lambda_local=1.5,
        lambda_mask=1.0,
        patch_loss="consistency",
        patch_loss_explicit=False,
        masking_strategy="random",
        masking_strategy_explicit=False,
        mask_ratio_explicit=False,
        ibot_prototypes=512,
        ibot_prototypes_explicit=False,
        lambda_mask_explicit=False,
        teacher_momentum=0.995,
        teacher_momentum_end=0.999,
        student_temp=0.1,
        teacher_temp_start=0.04,
        teacher_temp_end=0.07,
        teacher_temp_warmup_ratio=0.7,
        teacher_temp_warmup_ratio_explicit=False,
        relational_teacher_views="weak",
        relational_center_source=None,
        relational_pairing=None,
        log_every_n_steps=1000,
        save_every_epochs=1,
        keep_last_checkpoints=2,
        visualize_data=None,
        extract_size=None,
        metrics_sample_size=100,
        umap_n_neighbors=5,
        umap_min_dist=0.1,
        umap_metric="cosine",
        visualize_class_number=0,
        compute_embedding_metrics=False,
        batch_size=2,
        num_workers=0,
        cpus=1,
        device="auto",
        seed=42,
        resume="",
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    trainer.run_pretrain(args)
    return torch.load(
        tmp_path / "out" / "SSL_latest.pth", map_location="cpu", weights_only=False
    )


def test_positive_relational_koleo_persists_and_records_the_shared_config(tmp_path):
    """One real run covers the enabled checkpoint and reporting contracts."""
    saved = _train_relational(tmp_path, koleo_weight=0.1)
    rows = list(
        csv.DictReader(
            (tmp_path / "out" / "logs" / "instant_metrics.pretrain.csv").open(
                encoding="utf-8"
            )
        )
    )
    last = rows[-1]

    assert last["ssl_method"] == "relational"
    assert float(last["koleo_weight"]) == pytest.approx(0.1)
    assert last["koleo_status"] in {"active", "skipped"}
    assert last["koleo_loss_raw"] != ""
    # DINO-only diagnostics stay N/A under relational.
    assert last["dino_centering"] == ""
    assert last["cls_center_norm"] == ""
    assert saved["config"][KOLEO_WEIGHT_KEY] == pytest.approx(0.1)
    for name, value in KOLEO_FIXED_CONFIG.items():
        assert saved["config"][name] == value


def test_dino_module_owns_no_shared_koleo_constant():
    """The four public KoLeo keys have one definition, in the pretrain owner.

    A second copy in ``dino.py`` would silently diverge. This is a boundary
    check, not a behavioural one: the values are asserted to be the shared
    object's, and the DINO tables must not intersect the four keys.
    """
    import otuformer.training.dino as dino_module
    import otuformer.training.pretrain_recipe as recipe_module

    shared = set(recipe_module.KOLEO_STORED_KEYS)
    assert shared == {
        "koleo_weight",
        "koleo_normalize_eps",
        "koleo_distance_eps",
        "koleo_log_eps",
    }
    for table in ("DINO_FIXED_CONFIG", "DINO_DEFAULTS"):
        names = set(getattr(dino_module, table))
        assert not (names & shared), (table, sorted(names & shared))


@pytest.mark.parametrize("patch_mode", ["none", "consistency", "masked-feature", "ibot"])
def test_positive_weight_relational_koleo_composes_with_every_patch_mode(
    tmp_path, patch_mode
):
    """A positive KoLeo weight and any patch objective are independent choices.

    The checklist requires the interaction matrix, not just each half: the four
    keys must persist and the patch configuration must survive alongside them.
    """
    saved = _train_relational(
        tmp_path, koleo_weight=0.1, patch_loss=patch_mode, max_epochs=1
    )

    assert saved["config"][KOLEO_WEIGHT_KEY] == 0.1
    for name, value in KOLEO_FIXED_CONFIG.items():
        assert saved["config"][name] == value, name
    assert saved["config"]["patch_loss"] == patch_mode


def test_a_positive_weight_relational_payload_is_rejected_by_the_baseline_reader(
    tmp_path,
):
    """The compatibility boundary: readable at zero weight, not when enabled.

    ``84f845e`` has no relational KoLeo, so a payload that records the four keys
    must be refused by that reader rather than silently misread. The current
    reader accepts both shapes.
    """
    import textwrap

    zero = _train_relational(tmp_path / "zero", koleo_weight=0.0, max_epochs=1)
    enabled = _train_relational(tmp_path / "on", koleo_weight=0.1, max_epochs=1)
    assert not set(KOLEO_STORED_KEYS) & set(zero["config"])
    assert enabled["config"][KOLEO_WEIGHT_KEY] == 0.1

    # The current reader accepts both shapes.
    from otuformer.utils.checkpoint import resolve_checkpoint_ssl_identity

    for payload in (zero, enabled):
        assert resolve_checkpoint_ssl_identity(
            payload, payload["model_state_dict"]
        ) == ("relational", "projection_mlp_2048")

    # The baseline reader rejects the enabled shape and accepts the zero one.
    # The reader is a pinned copy of ``84f845e``'s module kept beside this test,
    # so the check needs no Git metadata and no checkout.
    zero_path = tmp_path / "zero.pth"
    enabled_path = tmp_path / "enabled.pth"
    torch.save(zero, zero_path)
    torch.save(enabled, enabled_path)

    script = textwrap.dedent(
        """
        import importlib.util, json, sys, torch

        spec = importlib.util.spec_from_file_location("baseline_reader", sys.argv[1])
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)

        out = {}
        for label, path in (("zero", sys.argv[2]), ("enabled", sys.argv[3])):
            payload = torch.load(path, weights_only=False)
            try:
                identity = baseline.resolve_checkpoint_ssl_identity(
                    payload, payload["model_state_dict"]
                )
                out[label] = "accepted" if identity else "none"
            except ValueError:
                out[label] = "rejected"
        print(json.dumps(out))
        """
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(BASELINE_READER),
            str(zero_path),
            str(enabled_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "zero": "accepted",
        "enabled": "rejected",
    }


def test_the_pinned_baseline_reader_is_the_recorded_revision():
    """The pinned copy must stay byte-identical to the commit it claims."""
    import hashlib

    digest = hashlib.sha256(BASELINE_READER.read_bytes()).hexdigest()
    assert digest == BASELINE_READER_SHA256
    assert BASELINE_READER.read_text(encoding="utf-8").count("resolve_checkpoint_ssl_identity") >= 1


def _finetune_manifest(tmp_path) -> Path:
    """A two-class manifest over the images the pretrain run already wrote."""
    import pandas as pd

    rows = [
        {"image": f"{index}.png", "label": "A" if index % 2 == 0 else "B"}
        for index in range(4)
    ]
    manifest = tmp_path / "finetune.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def test_a_positive_weight_relational_checkpoint_reaches_every_consumer(tmp_path):
    """extract, export, CAM and finetune initialization all read the new shape.

    One checkpoint is produced by a real run, then handed to each consumer's own
    entry point, not to a classifier that merely names the source. A reader that
    only understood the historical shape would raise here rather than at a
    user's first downstream run.
    """
    from otuformer.embedding.extractor import _load_model
    from otuformer.training.finetune_support import preflight_finetune_source
    from otuformer.utils.checkpoint import resolve_checkpoint_ssl_identity
    from otuformer.vision.cam import load_model_from_checkpoint
    import textwrap

    saved = _train_relational(tmp_path, koleo_weight=0.1, max_epochs=1)
    path = tmp_path / "out" / "SSL_latest.pth"
    assert saved["config"][KOLEO_WEIGHT_KEY] == 0.1
    assert resolve_checkpoint_ssl_identity(saved, saved["model_state_dict"]) == (
        "relational",
        "projection_mlp_2048",
    )

    # extract: the loader builds an encoder and reports the checkpoint size.
    model, checkpoint_size = _load_model(
        path, "vit_tiny_patch16_224", torch.device("cpu")
    )
    assert checkpoint_size == 32
    assert isinstance(model, torch.nn.Module)

    # CAM: returns (model, size, resolved name).
    cam_model, cam_size, resolved_name = load_model_from_checkpoint(
        path, "vit_tiny_patch16_224", device="cpu"
    )
    assert cam_size == 32
    assert isinstance(cam_model, torch.nn.Module)
    assert resolved_name == "vit_tiny_patch16_224"

    # export: run in a subprocess. The ONNX version converter is not reliable
    # twice in one process, so a shared process would make this order-dependent.
    onnx_path = tmp_path / "exported.onnx"
    script = textwrap.dedent(
        """
        import json, sys
        from pathlib import Path
        from otuformer.vision.export import export_to_onnx

        report = export_to_onnx(
            checkpoint_path=Path(sys.argv[1]),
            out_path=Path(sys.argv[2]),
            imgsz=32,
            opset=17,
        )
        print(json.dumps({"out_dim": report["out_dim"]}))
        """
    )
    exported = subprocess.run(
        [sys.executable, "-c", script, str(path), str(onnx_path)],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
    )
    assert exported.returncode == 0, exported.stderr[-800:]
    assert onnx_path.exists()
    assert json.loads(exported.stdout.strip().splitlines()[-1])["out_dim"] == 8

    # finetune: run the real entry point, which builds the model, replaces the
    # head and loads the encoder weights from this payload. A name-only
    # classifier would not touch any of that.
    finetune_out = tmp_path / "finetune_out"
    finetune_args = argparse.Namespace(
        train_data=str(_finetune_manifest(tmp_path)),
        input_images_dir=str(tmp_path / "images"),
        out_dir=str(finetune_out),
        overwrite=False,
        checkpoint=str(path),
        resume="",
        model_name="vit_tiny_patch16_224",
        metric_embed_dim=8,
        loss="arcface",
        subcenters=1,
        compact_weight=0.0,
        supcon_temperature=0.1,
        long_tail="none",
        finetune_epochs=1,
        finetune_lr=1e-3,
        weight_decay=0.05,
        compute_embedding_metrics=False,
        extract_size=None,
        warmup_epochs=0,
        augmentation="none",
        orientation_policy=None,
        eval_transform="none",
        batch_size=2,
        num_workers=0,
        cpus=0,
        device="cpu",
        seed=42,
        log_every_n_steps=1000,
        save_every_epochs=1,
        keep_last_checkpoints=1,
        # Frozen backbone: the weights in the saved checkpoint are the ones
        # loaded from the source payload, so equality is a real check.
        freeze_ratio=1.0,
        pseudo_label_from="",
        trace_batch_ids=False,
        cb_drw=False,
    )
    finetune_loop_module.run_finetune(finetune_args)
    finetuned = torch.load(
        finetune_out / "finetune_latest.pth", map_location="cpu", weights_only=False
    )
    assert finetuned["config"]["ssl_source_method"] == "relational"
    # The encoder weights came from this payload, not from a fresh init. The
    # block weights are frozen at freeze_ratio=1.0, so they cannot have moved.
    compared = 0
    for name, value in finetuned["model_state_dict"].items():
        if name.startswith("backbone.blocks."):
            assert torch.equal(value, saved["model_state_dict"][name]), name
            compared += 1
    assert compared > 0
