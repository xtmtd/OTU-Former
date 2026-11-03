"""Effective SSL config resolution, DINO preflight and pre-output safety."""

import argparse
import json
import random

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from otuformer.cli.main import app
from otuformer.training.dino import DINO_FIXED_CONFIG
from otuformer.training.trainer import (
    resolve_pretrain_ssl_config,
    run_pretrain,
)

runner = CliRunner()

MODEL = "vit_tiny_patch16_224"
SIZE = 32
DINO_FIELDS = (
    "dino_prototypes",
    "dino_centering",
    "koleo_weight",
    "dino_hidden_dim",
    "dino_mlp_init_std",
    "dino_mlp_init_bias",
    "dino_weight_norm_api",
    "dino_prototype_scale",
    "dino_freeze_last_layer_epochs",
    "dino_target_dtype",
    "koleo_normalize_eps",
    "koleo_distance_eps",
    "koleo_log_eps",
)


def _encoder(out_dim: int, registers: int | None = None):
    from otuformer.training.model import OTUFormerEncoder

    return OTUFormerEncoder(
        model_name=MODEL,
        out_dim=out_dim,
        pretrained=False,
        img_size=SIZE,
        **({"reg_tokens": registers} if registers else {}),
    )


def _adamw_group(params: list[int]) -> dict:
    """A real AdamW parameter group, not the lr/params subset.

    ``Optimizer.load_state_dict()`` accepts a group missing ``betas`` and the
    resumed ``step()`` then raises KeyError, so fixtures must carry the fields a
    restore actually needs.
    """
    return {
        "lr": 5e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.05,
        "amsgrad": False,
        "params": params,
    }


def dino_checkpoint(
    out_dim: int = 17,
    prototypes: int = 11,
    centering: str = "sinkhorn-knopp",
    koleo_weight: float = 0.1,
    registers: int | None = None,
) -> dict:
    from otuformer.training.dino import DINOObjective

    student = _encoder(out_dim, registers)
    teacher = _encoder(out_dim, registers)
    # Every checkpoint written by run_pretrain carries the registered pretrain
    # final-norm alias, and the strict restore always expects it.
    student.validate_pretrain_backbone()
    teacher.validate_pretrain_backbone()
    objective = DINOObjective(out_dim, prototypes, centering)
    # A complete AdamW state for every parameter the resumed optimizer holds;
    # absolute epoch 0 leaves the frozen prototype direction without state.
    from otuformer.training import trainer

    params = trainer._student_optimizer_parameters(student, None, objective)
    frozen = objective.trainable_parameters()[0]
    optimizer_state = {
        index: {
            "step": torch.tensor(1.0),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
        for index, parameter in enumerate(params)
        if parameter is not frozen
    }
    config = {
        "model_name": MODEL,
        "out_dim": out_dim,
        "ssl_method": "dino",
        "ssl_head": "dino_mlp_2048",
        "dino_prototypes": prototypes,
        "dino_centering": centering,
        "koleo_weight": koleo_weight,
        "augmentation_config": {"global_crop": {"size": SIZE}},
        **DINO_FIXED_CONFIG,
    }
    if centering == "ema":
        config["dino_center_momentum"] = 0.9
    else:
        config["dino_sinkhorn_iterations"] = 3
    return {
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "model_state_dict": teacher.state_dict(),
        "dino_objective": objective.state_dict(),
        "config": config,
        "args": {"max_epochs": 4, "global_crop_size": SIZE},
        "epoch": 0,
        "iteration": 2,
        # A genuinely resumable DINO payload also carries the joint optimizer
        # state, complete schedule metadata and RNG state.
        "optimizer": {
            "state": optimizer_state,
            "param_groups": [
                _adamw_group(list(range(len(params))))
            ],
        },
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
        "schedule": {
            "original_max_epochs": 4,
            "total_steps": 8,
            "steps_per_epoch": 2,
            "warmup_steps": 0,
            "last_lr": 5e-4,
            "final_lr": 5e-6,
        },
    }


def relational_checkpoint(out_dim: int = 256, registers: int | None = None) -> dict:
    student = _encoder(out_dim, registers)
    teacher = _encoder(out_dim, registers)
    config = {
        "model_name": MODEL,
        "out_dim": out_dim,
        "ssl_method": "relational",
        "ssl_head": "projection_mlp_2048",
        "augmentation_config": {"global_crop": {"size": SIZE}},
    }
    return {
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "model_state_dict": teacher.state_dict(),
        "center": torch.zeros(1, out_dim),
        "config": config,
        "args": {"max_epochs": 4, "global_crop_size": SIZE},
        "epoch": 0,
        "iteration": 2,
    }


def ssl_args(**overrides) -> argparse.Namespace:
    """Invocation defaults, i.e. every SSL option omitted."""
    base = dict(
        ssl_method="relational",
        ssl_method_explicit=False,
        out_dim=256,
        out_dim_explicit=False,
        dino_prototypes=64,
        dino_prototypes_explicit=False,
        dino_centering="ema",
        dino_centering_explicit=False,
        koleo_weight=0.0,
        koleo_weight_explicit=False,
        disable_cross_view_loss=False,
        disable_cross_view_loss_explicit=False,
        teacher_temp_warmup_ratio=0.7,
        teacher_temp_warmup_ratio_explicit=False,
        no_norm_bias_weight_decay=False,
        no_norm_bias_weight_decay_explicit=False,
        model_name=MODEL,
        register_tokens=None,
        global_crop_size=SIZE,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_dino_resume_inherits_only_omitted_values():
    checkpoint = dino_checkpoint()
    requested = ssl_args()

    effective = resolve_pretrain_ssl_config(requested, checkpoint)

    assert effective["ssl_method"] == "dino"
    assert effective["ssl_head"] == "dino_mlp_2048"
    assert effective["out_dim"] == 17
    assert effective["dino_prototypes"] == 11
    assert effective["dino_centering"] == "sinkhorn-knopp"
    assert effective["koleo_weight"] == 0.1
    assert effective["dino_sinkhorn_iterations"] == 3
    assert "dino_center_momentum" not in effective
    for name in DINO_FIELDS:
        assert name in effective, name
    assert effective["dino_hidden_dim"] == 2048
    assert effective["dino_prototype_scale"] == 1.0
    # The requested invocation record is untouched.
    assert requested.out_dim == 256
    assert requested.ssl_method == "relational"

    # Matching explicit values are still accepted; conflicts fail.
    matched = ssl_args(
        ssl_method="dino",
        ssl_method_explicit=True,
        out_dim=17,
        out_dim_explicit=True,
        dino_prototypes=11,
        dino_prototypes_explicit=True,
        dino_centering="sinkhorn-knopp",
        dino_centering_explicit=True,
        koleo_weight=0.1,
        koleo_weight_explicit=True,
    )
    assert resolve_pretrain_ssl_config(matched, checkpoint)["out_dim"] == 17

    # An explicit value equal to the CLI default still conflicts with D=17.
    for overrides in (
        {"ssl_method": "relational", "ssl_method_explicit": True},
        {"out_dim": 256, "out_dim_explicit": True},
        {"dino_prototypes": 64, "dino_prototypes_explicit": True},
        {"dino_centering": "ema", "dino_centering_explicit": True},
        {"koleo_weight": 0.0, "koleo_weight_explicit": True},
        {"disable_cross_view_loss": True},
    ):
        with pytest.raises(ValueError):
            resolve_pretrain_ssl_config(ssl_args(**overrides), checkpoint)

    # An inference-only DINO payload is not a resumable DINO checkpoint.
    inference_only = dino_checkpoint()
    del inference_only["dino_objective"]
    with pytest.raises(ValueError, match="dino_objective|objective"):
        resolve_pretrain_ssl_config(ssl_args(), inference_only)

    # Relational resume keeps its requested dimension: a matching default is
    # accepted, a non-default saved dimension fails instead of being inherited.
    default_width = relational_checkpoint(out_dim=256)
    assert resolve_pretrain_ssl_config(ssl_args(), default_width)["out_dim"] == 256
    non_default = relational_checkpoint(out_dim=64)
    with pytest.raises(ValueError):
        resolve_pretrain_ssl_config(ssl_args(), non_default)
    assert (
        resolve_pretrain_ssl_config(
            ssl_args(out_dim=64, out_dim_explicit=True), non_default
        )["out_dim"]
        == 64
    )


def test_new_run_validation_rejects_invalid_dino_options():
    for overrides in (
        {"dino_prototypes": 1, "dino_prototypes_explicit": True},
        {"dino_prototypes": 0, "dino_prototypes_explicit": True},
        {"koleo_weight": -0.1, "koleo_weight_explicit": True},
        {"koleo_weight": float("nan"), "koleo_weight_explicit": True},
        {"koleo_weight": float("inf"), "koleo_weight_explicit": True},
        {"out_dim": 0, "out_dim_explicit": True},
        {"dino_centering": "ema-knockoff", "dino_centering_explicit": True},
    ):
        with pytest.raises(ValueError):
            resolve_pretrain_ssl_config(
                ssl_args(ssl_method="dino", ssl_method_explicit=True, **overrides),
                None,
            )
    # Explicit DINO-only options under relational are rejected, including
    # explicitly supplied defaults.
    for overrides in (
        {"dino_prototypes": 64, "dino_prototypes_explicit": True},
        {"dino_centering": "ema", "dino_centering_explicit": True},
        {"koleo_weight": 0.0, "koleo_weight_explicit": True},
    ):
        with pytest.raises(ValueError, match="relational|not applicable"):
            resolve_pretrain_ssl_config(ssl_args(**overrides), None)
    # Omitted inactive defaults have no effect.
    assert resolve_pretrain_ssl_config(ssl_args(), None) == {
        "ssl_method": "relational",
        "ssl_head": "projection_mlp_2048",
        "out_dim": 256,
        "teacher_temp_warmup_ratio": 0.7,
        "norm_bias_weight_decay": True,
    }
    with pytest.raises(ValueError, match="cross-view|disable"):
        resolve_pretrain_ssl_config(
            ssl_args(
                ssl_method="dino",
                ssl_method_explicit=True,
                disable_cross_view_loss=True,
            ),
            None,
        )


@pytest.mark.parametrize("key", ["student", "teacher", "model_state_dict"])
@pytest.mark.parametrize(
    "corruption", ["missing_state", "dropped_projector", "wrong_dim", "missing_backbone"]
)
def test_dino_resume_rejects_each_corrupt_encoder_before_output(key, corruption):
    checkpoint = dino_checkpoint()
    if corruption == "missing_state":
        del checkpoint[key]
    elif corruption == "dropped_projector":
        state = checkpoint[key]
        state.pop("projector.net.4.weight")
    elif corruption == "wrong_dim":
        state = checkpoint[key]
        state["projector.net.4.weight"] = torch.zeros(9, 2048)
    else:
        state = checkpoint[key]
        key_name = next(k for k in state if k.startswith("backbone.blocks.0."))
        state.pop(key_name)

    with pytest.raises(ValueError):
        resolve_pretrain_ssl_config(ssl_args(), checkpoint)


def test_dino_resume_rejects_damaged_student_at_zero_registers():
    checkpoint = dino_checkpoint(registers=None)
    checkpoint["student"].pop("projector.net.4.weight")

    with pytest.raises(ValueError):
        resolve_pretrain_ssl_config(ssl_args(), checkpoint)


def test_dino_resume_rejects_bad_objective_and_center_state():
    checkpoint = dino_checkpoint(centering="sinkhorn-knopp")

    bad_scale = dino_checkpoint(centering="sinkhorn-knopp")
    bad_scale["dino_objective"]["student_prototypes.weight_g"] = torch.full(
        (11, 1), 0.5
    )
    unexpected_center = dino_checkpoint(centering="sinkhorn-knopp")
    unexpected_center["dino_objective"]["cls_center"] = torch.zeros(1, 11)
    missing_center = dino_checkpoint(centering="ema")
    del missing_center["dino_objective"]["cls_center"]
    legacy_center = dino_checkpoint()
    legacy_center["center"] = torch.zeros(1, 17)

    for corrupt in (bad_scale, unexpected_center, missing_center, legacy_center):
        with pytest.raises(ValueError):
            resolve_pretrain_ssl_config(ssl_args(), corrupt)

    # The matching EMA payload, with its center, is accepted.
    assert resolve_pretrain_ssl_config(ssl_args(), dino_checkpoint(centering="ema"))[
        "dino_centering"
    ] == "ema"
    assert resolve_pretrain_ssl_config(ssl_args(), checkpoint)["out_dim"] == 17


def test_new_dino_preflight_preserves_overwrite_sentinel(tmp_path, monkeypatch):
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    sentinel = out_dir / "keep.txt"
    sentinel.write_bytes(b"original")

    # A prototype count below the numeric minimum must be rejected before
    # overwrite handling, leaving the existing output untouched.
    result = runner.invoke(
        app,
        [
            "pretrain",
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--overwrite",
            "--ssl-method",
            "dino",
            "--dino-prototypes",
            "1",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code != 0
    assert sentinel.read_bytes() == b"original"
    assert not (out_dir / "logs" / "pretrain.log").exists()


def test_relational_preflight_ignores_the_prototype_parameterisation(monkeypatch):
    from otuformer.training.dino import WeightNormLinear

    def broken_init(self, *args, **kwargs):
        raise AssertionError("relational must not construct DINO prototypes")

    monkeypatch.setattr(WeightNormLinear, "__init__", broken_init)
    assert resolve_pretrain_ssl_config(ssl_args(), None)["ssl_method"] == "relational"


def test_dino_resume_preflight_preserves_existing_logs(tmp_path):
    out_dir = tmp_path / "run"
    (out_dir / "logs").mkdir(parents=True)
    log_path = out_dir / "logs" / "pretrain.log"
    log_path.write_bytes(b"previous run log\n")
    sentinel = out_dir / "keep.txt"
    sentinel.write_bytes(b"original")

    checkpoint = dino_checkpoint(out_dim=17)
    path = tmp_path / "dino.pth"
    torch.save(checkpoint, path)

    result = runner.invoke(
        app,
        [
            "pretrain",
            "--input-images-dir",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--resume",
            str(path),
            "--out-dim",
            "256",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code != 0
    assert sentinel.read_bytes() == b"original"
    assert log_path.read_bytes() == b"previous run log\n"
    assert not (out_dir / "logs" / "instant_metrics.pretrain.csv").exists()


def test_direct_dino_preflight_creates_no_output(tmp_path):
    out_dir = tmp_path / "direct"
    args = ssl_args(
        ssl_method="dino",
        ssl_method_explicit=True,
        dino_prototypes=1,
        out_dir=str(out_dir),
        seed=42,
        cpus=1,
        device="cpu",
        resume="",
    )

    with pytest.raises(ValueError, match="dino-prototypes"):
        run_pretrain(args)
    assert not out_dir.exists()


def test_effective_slot_is_reused(tmp_path, monkeypatch):
    import otuformer.training.trainer as trainer

    supplied = {
        "ssl_method": "relational",
        "ssl_head": "projection_mlp_2048",
        "out_dim": 256,
    }
    calls = []

    def fail_second_resolution(args, resume_ckpt):
        calls.append(resume_ckpt)
        raise AssertionError("the supplied effective slot must be reused")

    monkeypatch.setattr(
        trainer, "resolve_pretrain_ssl_config", fail_second_resolution
    )
    args = ssl_args(
        out_dir=str(tmp_path / "reused"),
        seed=42,
        cpus=1,
        device="cpu",
        resume="",
        ssl_config_effective=supplied,
        input_images_dir=str(tmp_path),
        train_data="",
    )

    # Fails later on missing training data, but never re-resolves the slot.
    with pytest.raises(Exception):
        run_pretrain(args)
    assert calls == []


def test_absent_slot_resolves_before_mkdir(tmp_path, monkeypatch):
    import otuformer.training.trainer as trainer

    out_dir = tmp_path / "absent"
    seen = {}

    def record(args, resume_ckpt):
        seen["out_dir_exists"] = out_dir.exists()
        raise RuntimeError("stop before training")

    monkeypatch.setattr(trainer, "resolve_pretrain_ssl_config", record)
    args = ssl_args(
        out_dir=str(out_dir),
        seed=42,
        cpus=1,
        device="cpu",
        resume="",
    )

    with pytest.raises(RuntimeError, match="stop before training"):
        run_pretrain(args)
    assert seen["out_dir_exists"] is False
    assert not out_dir.exists()


def test_dino_resume_rejects_conflicting_fixed_center_settings():
    """Mode-specific fixed constants must match a recorded value, not drift."""
    for field, value, centering in (
        ("dino_center_momentum", 0.5, "ema"),
        ("dino_sinkhorn_iterations", 5, "sinkhorn-knopp"),
    ):
        checkpoint = dino_checkpoint(centering=centering)
        checkpoint["config"][field] = value
        with pytest.raises(ValueError, match="center_momentum|sinkhorn|fixed"):
            resolve_pretrain_ssl_config(ssl_args(), checkpoint)


def test_ssl_display_projection_differs_from_persisted_config():
    from otuformer.training.trainer import _ssl_display_config

    relational = {
        "ssl_method": "relational",
        "ssl_head": "projection_mlp_2048",
        "out_dim": 256,
        "teacher_temp_warmup_ratio": 0.7,
        "norm_bias_weight_decay": True,
    }
    display = _ssl_display_config(relational)
    # Display nulls exist for inactive fields, but the persisted config keeps
    # the sparse relational identity plus the two common recipe controls.
    assert display["dino_prototypes"] is None
    assert display["koleo_weight"] is None
    assert display["teacher_temp_warmup_ratio"] == 0.7
    assert display["norm_bias_weight_decay"] is True
    assert set(relational) == {
        "ssl_method",
        "ssl_head",
        "out_dim",
        "teacher_temp_warmup_ratio",
        "norm_bias_weight_decay",
    }


@pytest.mark.parametrize(
    "missing",
    [
        "dino_prototypes",
        "dino_centering",
        "koleo_weight",
        "out_dim",
        "dino_weight_norm_api",
        "dino_prototype_scale",
        "dino_center_momentum",
    ],
)
def test_dino_resume_rejects_missing_recorded_config(missing):
    """A DINO payload must record its own recipe; never fall back to defaults."""
    checkpoint = dino_checkpoint(centering="ema")
    del checkpoint["config"][missing]

    with pytest.raises(ValueError, match=missing.split("_")[-1] if False else "."):
        resolve_pretrain_ssl_config(ssl_args(), checkpoint)


def test_dino_resume_rejects_missing_rng_state():
    checkpoint = dino_checkpoint()
    del checkpoint["rng_state"]
    with pytest.raises(ValueError, match="rng"):
        resolve_pretrain_ssl_config(ssl_args(), checkpoint)

    truncated = dino_checkpoint()
    truncated["rng_state"] = {"torch": torch.get_rng_state()}
    with pytest.raises(ValueError, match="rng"):
        resolve_pretrain_ssl_config(ssl_args(), truncated)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("epoch", "invalid", "epoch"),
        ("iteration", -3, "iteration"),
        ("schedule", {**{"original_max_epochs": 4, "total_steps": "invalid",
                         "steps_per_epoch": 2, "warmup_steps": 0,
                         "last_lr": 5e-4, "final_lr": 5e-6}}, "total_steps"),
        ("optimizer", {"state": {}, "param_groups": [{"lr": 5e-4, "params": []}]},
         "param"),
    ],
)
def test_dino_resume_rejects_corrupt_progress_and_optimizer_shape(key, value, match):
    checkpoint = dino_checkpoint()
    checkpoint[key] = value
    with pytest.raises(ValueError, match=match):
        resolve_pretrain_ssl_config(ssl_args(), checkpoint)


def test_relational_resume_rejects_weight_width_conflict_without_recorded_out_dim():
    """A historical payload that records no width still has an actual one."""
    from otuformer.training.model import OTUFormerEncoder

    encoder = OTUFormerEncoder(
        model_name=MODEL, out_dim=17, pretrained=False, img_size=SIZE
    )
    checkpoint = {
        "student": encoder.state_dict(),
        "teacher": encoder.state_dict(),
        "model_state_dict": encoder.state_dict(),
        "center": torch.zeros(1, 17),
        "config": {"model_name": MODEL},
        "args": {"max_epochs": 4, "global_crop_size": SIZE},
    }
    assert resolve_pretrain_ssl_config(
        ssl_args(out_dim=17, out_dim_explicit=True), checkpoint
    )["out_dim"] == 17
    with pytest.raises(ValueError, match="out_dim|width"):
        resolve_pretrain_ssl_config(ssl_args(), checkpoint)


def _checkpoint_with_patch_optimizer(mode="masked-feature", epoch=1):
    from otuformer.training import trainer
    from otuformer.training.dino import DINOObjective
    from otuformer.training.model import PatchObjective

    checkpoint = dino_checkpoint()
    student = _encoder(17)
    patch = PatchObjective(mode, student.backbone.num_features, 11)
    objective = DINOObjective(17, 11, "sinkhorn-knopp")
    checkpoint["config"].update(trainer._resolve_patch_config(
        argparse.Namespace(patch_loss=mode, ibot_prototypes=11), None
    ))
    checkpoint["patch_objective"] = patch.state_dict()
    params = trainer._student_optimizer_parameters(student, patch, objective)
    checkpoint["optimizer"] = {
        "param_groups": [_adamw_group(list(range(len(params))))],
        "state": {
            index: {"step": torch.tensor(2.0),
                    "exp_avg": torch.zeros_like(parameter),
                    "exp_avg_sq": torch.zeros_like(parameter)}
            for index, parameter in enumerate(params)
            if epoch > 0 or parameter is not objective.trainable_parameters()[0]
        },
    }
    checkpoint["epoch"] = epoch
    checkpoint["iteration"] = 2 * (epoch + 1)
    return checkpoint


@pytest.mark.parametrize("mode", ["masked-feature", "ibot"])
def test_dino_patch_optimizer_requires_state_at_each_parameter_id(mode):
    checkpoint = _checkpoint_with_patch_optimizer(mode)
    # mask_token and backbone.cls_token have the same shape. Patch state must
    # not hide a missing backbone moment just because the shape still occurs.
    del checkpoint["optimizer"]["state"][0]
    with pytest.raises(ValueError, match="AdamW|optimizer"):
        resolve_pretrain_ssl_config(ssl_args(), checkpoint)


@pytest.mark.parametrize("corruption", ["swapped", "extra_id", "invalid_step"])
def test_dino_optimizer_validates_order_count_and_step(corruption):
    checkpoint = _checkpoint_with_patch_optimizer()
    optimizer = checkpoint["optimizer"]
    if corruption == "swapped":
        optimizer["state"][0], optimizer["state"][1] = (
            optimizer["state"][1], optimizer["state"][0]
        )
    elif corruption == "extra_id":
        optimizer["param_groups"][0]["params"].append(9999)
    else:
        optimizer["state"][0]["step"] = torch.tensor(float("nan"))
    with pytest.raises(ValueError, match="AdamW|optimizer"):
        resolve_pretrain_ssl_config(ssl_args(), checkpoint)


@pytest.mark.parametrize("mode", ["none", "consistency", "masked-feature", "ibot"])
@pytest.mark.parametrize("epoch", [0, 1])
def test_dino_complete_patch_optimizer_and_epoch_zero_freeze_are_valid(mode, epoch):
    checkpoint = _checkpoint_with_patch_optimizer(mode, epoch)
    assert resolve_pretrain_ssl_config(ssl_args(), checkpoint)["ssl_method"] == "dino"


@pytest.mark.parametrize("corruption", ["center_shape", "norm_shape", "norm_key", "short_plan"])
def test_dino_preflight_rejects_payload_that_cannot_restore_strictly(corruption):
    checkpoint = dino_checkpoint()
    if corruption == "center_shape":
        checkpoint["student"]["center"] = torch.zeros(1, 18)
    elif corruption == "norm_shape":
        checkpoint["student"]["_pretrain_final_norm.weight"] = torch.zeros(1)
    elif corruption == "norm_key":
        checkpoint["student"]["_pretrain_final_norm.unknown"] = torch.zeros(1)
    else:
        # All individual counts look plausible, but four two-step epochs need 8.
        checkpoint["schedule"]["total_steps"] = 7
    with pytest.raises(ValueError, match="restore|shape|schedule"):
        resolve_pretrain_ssl_config(ssl_args(), checkpoint)


@pytest.mark.parametrize(
    "corruption",
    ["betas", "epilogue", "missing_eps", "missing_weight_decay", "nan_lr",
     "missing_alias", "partial_alias", "iteration_past_end",
     "iteration_short_of_epoch"],
)
def test_dino_preflight_rejects_progress_optimizer_and_alias_gaps(corruption):
    """Each case passes the old validation but fails before the model restores."""
    checkpoint = dino_checkpoint()
    if corruption == "betas":
        # AdamW.load_state_dict() accepts a group without betas; step() then
        # raises KeyError, so the preflight must reject it first.
        checkpoint["optimizer"]["param_groups"][0].pop("betas")
    elif corruption == "epilogue":
        checkpoint["optimizer"]["param_groups"][0]["betas"] = (0.9, 1.5)
    elif corruption in ("missing_eps", "missing_weight_decay"):
        # AdamW.step() reads both; load_state_dict() silently accepts their loss.
        checkpoint["optimizer"]["param_groups"][0].pop(
            "eps" if corruption == "missing_eps" else "weight_decay"
        )
    elif corruption == "nan_lr":
        checkpoint["optimizer"]["param_groups"][0]["lr"] = float("nan")
    elif corruption == "missing_alias":
        for key in ("student", "teacher", "model_state_dict"):
            for name in list(checkpoint[key]):
                if name.startswith("_pretrain_final_norm."):
                    checkpoint[key].pop(name)
    elif corruption == "partial_alias":
        checkpoint["student"].pop("_pretrain_final_norm.bias")
    elif corruption == "iteration_past_end":
        checkpoint["iteration"] = 999_999
    else:
        # The final iteration of a 4 x 2 plan cannot describe any loader length
        # once more epochs are requested.
        checkpoint["iteration"] = checkpoint["schedule"]["total_steps"] - 1
    # The CLI always supplies --max-epochs, so the preflight can judge the
    # target; without it no target-dependent check applies.
    with pytest.raises(ValueError, match="AdamW|alias|restore|iteration|cannot continue"):
        resolve_pretrain_ssl_config(ssl_args(max_epochs=4), checkpoint)


def test_dino_preflight_accepts_an_extended_plan_with_short_progress():
    """A resumed run may change loader length, so progress is not recomputed."""
    checkpoint = dino_checkpoint()
    checkpoint["epoch"] = 0
    checkpoint["iteration"] = 6
    assert resolve_pretrain_ssl_config(
        ssl_args(max_epochs=8), checkpoint
    )["ssl_method"] == "dino"


def test_dino_preflight_accepts_the_last_recorded_iteration_when_extending():
    """The final iteration of the recorded plan is still resumable forward.

    Episode boundary case: no step of the recorded plan is left, but a resumed
    epoch of a different loader length starting at this step is legitimate.
    """
    checkpoint = dino_checkpoint()
    checkpoint["epoch"] = 3
    checkpoint["iteration"] = checkpoint["schedule"]["total_steps"]
    # A post-freeze epoch has AdamW state for the prototype direction too; the
    # fixture omits it because it models absolute epoch 0.
    student = _encoder(17)
    from otuformer.training.dino import DINOObjective
    from otuformer.training import trainer

    objective = DINOObjective(17, 11, "sinkhorn-knopp")
    params = trainer._student_optimizer_parameters(student, None, objective)
    state = checkpoint["optimizer"]["state"]
    for index, parameter in enumerate(params):
        state.setdefault(
            index,
            {
                "step": torch.tensor(3.0),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
            },
        )
    assert resolve_pretrain_ssl_config(
        ssl_args(max_epochs=8), checkpoint
    )["ssl_method"] == "dino"


def test_relational_resume_rejects_config_width_disagreeing_with_weights():
    """Recorded width and actual width are checked separately, not either/or."""
    from otuformer.training.model import OTUFormerEncoder

    encoder = OTUFormerEncoder(
        model_name=MODEL, out_dim=17, pretrained=False, img_size=SIZE
    )
    checkpoint = {
        "student": encoder.state_dict(),
        "teacher": encoder.state_dict(),
        "model_state_dict": encoder.state_dict(),
        "center": torch.zeros(1, 17),
        # Config claims 256, the stored projector is 17 wide.
        "config": {"model_name": MODEL, "out_dim": 256},
        "args": {"max_epochs": 4, "global_crop_size": SIZE},
    }
    with pytest.raises(ValueError, match="out_dim|width|disagree|conflict"):
        resolve_pretrain_ssl_config(ssl_args(), checkpoint)


# --- Shared pretrain recipe controls (2026-11-02 amendment) -----------------


def _recipe_checkpoint(**config_overrides) -> dict:
    """A relational SSL payload whose config carries the shared recipe keys."""
    config = {
        "model_name": MODEL,
        "ssl_method": "relational",
        "ssl_head": "projection_mlp_2048",
        "out_dim": 256,
        "teacher_temp_warmup_ratio": 0.7,
        "norm_bias_weight_decay": True,
    }
    config.update(config_overrides)
    return {
        "student": _encoder(256).state_dict(),
        "config": config,
        "args": {"max_epochs": 4, "global_crop_size": SIZE},
    }


@pytest.mark.parametrize("method", ["relational", "dino"])
def test_pretrain_recipe_defaults_and_validation(method):
    if method == "relational":
        resolved = resolve_pretrain_ssl_config(ssl_args(), None)
    else:
        resolved = resolve_pretrain_ssl_config(
            ssl_args(ssl_method="dino", ssl_method_explicit=True), None
        )
    assert resolved["teacher_temp_warmup_ratio"] == 0.7
    assert resolved["norm_bias_weight_decay"] is True
    if method == "dino":
        assert resolved["dino_prototypes"] == 64
    for bad in (float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValueError, match="teacher_temp_warmup_ratio"):
            resolve_pretrain_ssl_config(
                ssl_args(
                    teacher_temp_warmup_ratio=bad,
                    teacher_temp_warmup_ratio_explicit=True,
                ),
                None,
            )
    # A nonboolean persisted flag is damaged metadata, not a truthy value.
    checkpoint = _recipe_checkpoint(norm_bias_weight_decay="yes")
    with pytest.raises(ValueError, match="norm_bias_weight_decay"):
        resolve_pretrain_ssl_config(ssl_args(), checkpoint)


def test_teacher_temp_warmup_ratio_schedule():
    from otuformer.training.trainer import _build_teacher_temp_schedule

    def historical(total: int, ratio: float):
        warmup = int(total * ratio)
        if warmup <= 0:
            return np.ones(total, dtype=np.float32) * 0.07
        return np.concatenate(
            [
                np.linspace(0.04, 0.07, warmup, dtype=np.float32),
                np.ones(total - warmup, dtype=np.float32) * 0.07,
            ]
        )

    for total, ratio in ((5050, 0.7), (5050, 0.3), (90, 0.7), (10, 1.0), (3, 0.0)):
        actual = _build_teacher_temp_schedule(total, 0.04, 0.07, ratio)
        assert np.array_equal(actual, historical(total, ratio))
    # Truncation of the float product, not an exact decimal floor.
    assert int(90 * 0.7) == 62
    thirty = _build_teacher_temp_schedule(5050, 0.04, 0.07, 0.3)
    assert int(5050 * 0.3) == 1515
    assert thirty[0] == pytest.approx(0.04)
    assert thirty[1513] < 0.07
    assert np.allclose(thirty[1514:], 0.07)
    # total_iters=1 rounds to zero warmup steps: the constant final temperature.
    assert np.array_equal(
        _build_teacher_temp_schedule(1, 0.04, 0.07, 0.7),
        np.ones(1, dtype=np.float32) * 0.07,
    )
    # Positional calls keep the historical 0.7 default.
    assert np.array_equal(
        _build_teacher_temp_schedule(10, 0.04, 0.07),
        _build_teacher_temp_schedule(10, 0.04, 0.07, 0.7),
    )


def test_pretrain_recipe_resume_inherits_and_rejects_conflicts():
    saved = _recipe_checkpoint(
        teacher_temp_warmup_ratio=0.3, norm_bias_weight_decay=False
    )
    inherited = resolve_pretrain_ssl_config(ssl_args(), saved)
    assert inherited["teacher_temp_warmup_ratio"] == 0.3
    assert inherited["norm_bias_weight_decay"] is False
    # Explicit equality passes and the requested args stay untouched.
    args = ssl_args(
        teacher_temp_warmup_ratio=0.3,
        teacher_temp_warmup_ratio_explicit=True,
        no_norm_bias_weight_decay=True,
        no_norm_bias_weight_decay_explicit=True,
    )
    assert resolve_pretrain_ssl_config(args, saved)["teacher_temp_warmup_ratio"] == 0.3
    assert args.teacher_temp_warmup_ratio == 0.3
    for overrides in (
        {"teacher_temp_warmup_ratio": 0.7, "teacher_temp_warmup_ratio_explicit": True},
        {"no_norm_bias_weight_decay": False, "no_norm_bias_weight_decay_explicit": True},
    ):
        with pytest.raises(ValueError, match="teacher_temp_warmup_ratio|norm_bias"):
            resolve_pretrain_ssl_config(ssl_args(**overrides), saved)
    # Partial metadata is malformed: it never inherits or defaults silently.
    partial = _recipe_checkpoint()
    del partial["config"]["norm_bias_weight_decay"]
    with pytest.raises(ValueError, match="part of the shared pretrain recipe"):
        resolve_pretrain_ssl_config(ssl_args(), partial)


@pytest.mark.parametrize("saved_wd", [0, 0.0, ""])
def test_pretrain_recipe_validates_the_saved_flag_before_the_request(saved_wd):
    """A truthy/falsy non-boolean is damaged metadata, never a policy value."""
    checkpoint = _recipe_checkpoint(norm_bias_weight_decay=saved_wd)
    with pytest.raises(ValueError, match="norm_bias_weight_decay"):
        resolve_pretrain_ssl_config(
            ssl_args(
                no_norm_bias_weight_decay=True,
                no_norm_bias_weight_decay_explicit=True,
            ),
            checkpoint,
        )
    with pytest.raises(ValueError, match="norm_bias_weight_decay"):
        resolve_pretrain_ssl_config(ssl_args(), checkpoint)


@pytest.mark.parametrize("method", ["relational", "dino"])
@pytest.mark.parametrize(
    ("saved_ratio", "requested"),
    [(None, 0.3), (True, 1.0), (False, 0.0)],
)
def test_pretrain_recipe_validates_the_saved_ratio_before_the_request(
    method, saved_ratio, requested
):
    """A null or boolean recorded ratio is damaged, never a value to override."""
    if method == "relational":
        checkpoint = _recipe_checkpoint(teacher_temp_warmup_ratio=saved_ratio)
        request = ssl_args(
            teacher_temp_warmup_ratio=requested,
            teacher_temp_warmup_ratio_explicit=True,
        )
        omitted = ssl_args()
    else:
        checkpoint = dino_checkpoint()
        checkpoint["config"].update(
            teacher_temp_warmup_ratio=saved_ratio, norm_bias_weight_decay=True
        )
        request = ssl_args(
            ssl_method="dino",
            ssl_method_explicit=True,
            teacher_temp_warmup_ratio=requested,
            teacher_temp_warmup_ratio_explicit=True,
        )
        omitted = ssl_args(ssl_method="dino", ssl_method_explicit=True)
    with pytest.raises(ValueError, match="teacher_temp_warmup_ratio"):
        resolve_pretrain_ssl_config(request, checkpoint)
    # The same payload is rejected when the ratio is simply omitted.
    with pytest.raises(ValueError, match="teacher_temp_warmup_ratio"):
        resolve_pretrain_ssl_config(omitted, checkpoint)


def test_pretrain_recipe_rejects_recorded_null_fields(capsys):
    """A recorded-but-null pair is damaged, not the historical missing pair."""
    for checkpoint in (
        _recipe_checkpoint(teacher_temp_warmup_ratio=None, norm_bias_weight_decay=None),
        _recipe_checkpoint(norm_bias_weight_decay=None),
    ):
        with pytest.raises(ValueError, match="teacher_temp_warmup_ratio|norm_bias_weight_decay"):
            resolve_pretrain_ssl_config(ssl_args(), checkpoint)
    assert "records neither" not in capsys.readouterr().out


def _legacy_recipe_checkpoint(relational: bool) -> dict:
    checkpoint = (
        {
            "student": _encoder(256).state_dict(),
            "config": {
                "model_name": MODEL,
                "ssl_method": "relational",
                "ssl_head": "projection_mlp_2048",
                "out_dim": 256,
            },
            "args": {"max_epochs": 4, "global_crop_size": SIZE},
        }
        if relational
        else dino_checkpoint()
    )
    for name in ("teacher_temp_warmup_ratio", "norm_bias_weight_decay"):
        checkpoint["config"].pop(name, None)
    return checkpoint


@pytest.mark.parametrize("relational", [True, False])
def test_pretrain_legacy_recipe_default_parity(relational, capsys):
    checkpoint = _legacy_recipe_checkpoint(relational)
    unchanged = {key: dict(value) if isinstance(value, dict) else value
                 for key, value in checkpoint.items()}
    request = (
        ssl_args(resume="ckpt.pth")
        if relational
        else ssl_args(
            resume="ckpt.pth", ssl_method="dino", ssl_method_explicit=True
        )
    )
    resolved = resolve_pretrain_ssl_config(request, checkpoint)
    assert resolved["teacher_temp_warmup_ratio"] == 0.7
    assert resolved["norm_bias_weight_decay"] is True
    assert "records neither teacher_temp_warmup_ratio" in capsys.readouterr().out
    # The loaded payload is never mutated to add inferred metadata.
    assert checkpoint["config"].get("teacher_temp_warmup_ratio") is None
    assert checkpoint["config"].keys() == unchanged["config"].keys()
    # DINO's strict recipe enumeration must not reject the missing common pair.
    # Explicit legacy nondefaults are rejected with truthful messages that never
    # claim the checkpoint recorded a value.
    with pytest.raises(ValueError) as ratio_error:
        resolve_pretrain_ssl_config(
            ssl_args(
                resume="ckpt.pth",
                teacher_temp_warmup_ratio=0.3,
                teacher_temp_warmup_ratio_explicit=True,
            ),
            _legacy_recipe_checkpoint(True),
        )
    message = str(ratio_error.value)
    assert "does not record teacher_temp_warmup_ratio" in message
    assert "records teacher_temp_warmup_ratio" not in message
    with pytest.raises(ValueError) as wd_error:
        resolve_pretrain_ssl_config(
            ssl_args(
                resume="ckpt.pth",
                no_norm_bias_weight_decay=True,
                no_norm_bias_weight_decay_explicit=True,
            ),
            _legacy_recipe_checkpoint(True),
        )
    message = str(wd_error.value)
    assert "does not record norm_bias_weight_decay" in message
    assert "optimizer-partition conversion" in message


def test_pretrain_recipe_schema_and_replay():
    from otuformer.cli_schema import (
        build_argv,
        export_cli_schema,
        parameter_index,
        validate_parameters,
    )

    schema = export_cli_schema("pretrain")["commands"]["pretrain"]
    entries = {item["name"]: item for item in schema["parameters"]}
    flag = entries["no_norm_bias_weight_decay"]
    assert flag["opts"] == ["--no-norm-bias-weight-decay"]
    assert flag["secondary_opts"] == []
    assert flag["default"] == {"kind": "value", "value": False}
    ratio = entries["teacher_temp_warmup_ratio"]
    assert ratio["default"] == {"kind": "value", "value": 0.7}
    card = json.dumps(schema, default=str)
    assert "--no-norm-bias-weight-decay" in card
    # One negative flag only: no positive spelling to keep the help short.
    assert "--norm-bias-weight-decay" not in card
    # Explicit False replays with the negative spelling; omission invents neither.
    required = {"input_images_dir": "./images", "model_name": MODEL}
    explicit_false = validate_parameters(
        "pretrain", {**required, "no_norm_bias_weight_decay": True}
    )
    assert explicit_false["provenance"]["no_norm_bias_weight_decay"] == "explicit"
    argv = build_argv(
        "pretrain",
        {**required, "no_norm_bias_weight_decay": True},
        executable=["otuformer"],
    )
    assert "--no-norm-bias-weight-decay" in argv
    omitted = validate_parameters("pretrain", dict(required))
    assert omitted["provenance"]["no_norm_bias_weight_decay"] == "omitted-default"
    argv = build_argv("pretrain", dict(required), executable=["otuformer"])
    assert "--no-norm-bias-weight-decay" not in argv
    # Resume inheritance is a claim about the checkpoint, not a static fact.
    resumed = validate_parameters(
        "pretrain", {**required, "resume": "ckpt.pth"}
    )
    assert resumed["provenance"]["teacher_temp_warmup_ratio"] == "omitted-inherited"
    assert resumed["provenance"]["no_norm_bias_weight_decay"] == "omitted-inherited"
    # Fine-tune exposes neither option.
    finetune = parameter_index("finetune")
    assert "teacher_temp_warmup_ratio" not in finetune
    assert "no_norm_bias_weight_decay" not in finetune


def _wd_model():
    """A tiny student with a norm, biases and plain weights, plus an objective."""
    from otuformer.training.dino import DINOObjective
    from otuformer.training.model import PatchObjective
    from otuformer.training import trainer

    student = _encoder(17)
    patch = PatchObjective("ibot", student.backbone.num_features, 11)
    objective = DINOObjective(17, 11, "ema")
    groups = trainer._pretrain_optimizer_groups(
        student,
        patch,
        objective,
        weight_decay=0.05,
        norm_bias_weight_decay=False,
    )
    return trainer, student, patch, objective, groups


def test_pretrain_norm_bias_wd_groups():
    trainer, student, patch, objective, groups = _wd_model()
    assert len(groups) == 2
    decay, no_decay = (group["params"] for group in groups)
    assert groups[0]["weight_decay"] == 0.05
    assert groups[1]["weight_decay"] == 0.0
    every = trainer._student_optimizer_parameters(student, patch, objective)
    source_order = [id(p) for p in every]
    # Every parameter exactly once, and each group keeps source-relative order.
    assert sorted(id(p) for p in list(decay) + list(no_decay)) == sorted(source_order)
    assert len({id(p) for p in list(decay) + list(no_decay)}) == len(every)
    for group in (decay, no_decay):
        positions = [source_order.index(id(p)) for p in group]
        assert positions == sorted(positions)
    no_decay_ids = {id(p) for p in no_decay}
    for module in student.modules():
        if isinstance(module, torch.nn.LayerNorm):
            assert id(module.weight) in no_decay_ids
            if module.bias is not None:
                assert id(module.bias) in no_decay_ids
    # Every bias is exempt, whatever module owns it.
    for name, parameter in student.named_parameters():
        if name.endswith(".bias") or name == "bias":
            assert id(parameter) in no_decay_ids, name
    # backbones' plain weight matrices, tokens and the prototype direction keep WD.
    for name, parameter in student.named_parameters():
        if name.endswith("weight") and not name.startswith("backbone.norm"):
            if ".blocks." in name and ".norm" not in name:
                assert id(parameter) not in no_decay_ids, name
    prototype = objective.trainable_parameters()[0]
    assert id(prototype) not in no_decay_ids
    assert objective_student_scale_ids(objective).isdisjoint(no_decay_ids)


def objective_student_scale_ids(objective) -> set[int]:
    return {id(parameter) for parameter in objective.student_prototypes.parameters()}


def test_pretrain_wd_default_preserves_parameter_order():
    from otuformer.training import trainer
    from otuformer.training.dino import DINOObjective
    from otuformer.training.model import PatchObjective

    student = _encoder(17)
    patch = PatchObjective("masked-feature", student.backbone.num_features, 11)
    objective = DINOObjective(17, 11, "ema")
    params = trainer._student_optimizer_parameters(student, patch, objective)
    baseline = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.05)
    actual = torch.optim.AdamW(
        trainer._pretrain_optimizer_groups(
            student,
            patch,
            objective,
            weight_decay=0.05,
            norm_bias_weight_decay=True,
        ),
        lr=5e-4,
        weight_decay=0.05,
    )
    assert len(actual.param_groups) == len(baseline.param_groups) == 1
    assert [id(p) for p in actual.param_groups[0]["params"]] == [
        id(p) for p in params
    ]
    # Two separate required checks: the identity/order list above, and the full
    # saved group dictionary below (which alone cannot authenticate identity).
    assert actual.state_dict()["param_groups"] == baseline.state_dict()["param_groups"]


def test_pretrain_joint_clip_includes_no_decay_group():
    trainer, student, patch, objective, groups = _wd_model()
    optimizer = torch.optim.AdamW(groups, lr=5e-4, weight_decay=0.05)
    flat = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(flat) == len({id(p) for p in flat})
    for index, parameter in enumerate(flat):
        parameter.grad = torch.full_like(parameter, 1.0)
    lr_before = [group["lr"] for group in optimizer.param_groups]
    expected = trainer._compute_grad_norm_parameters(flat)
    assert expected == pytest.approx(float(len(flat) and expected))
    measured = torch.nn.utils.clip_grad_norm_(flat, max_norm=3.0).item()
    assert measured == pytest.approx(expected, rel=1e-5)
    optimizer.step()
    assert [group["lr"] for group in optimizer.param_groups] == lr_before
    # Clipping used every group, not just the first.
    assert sum(
        1 for parameter in optimizer.param_groups[1]["params"] if parameter.grad is not None
    ) > 0


def test_pretrain_no_decay_zero_gradient_step():
    trainer, student, patch, objective, groups = _wd_model()
    optimizer = torch.optim.AdamW(groups, lr=5e-4, weight_decay=0.05)
    decay_params = optimizer.param_groups[0]["params"]
    no_decay_params = optimizer.param_groups[1]["params"]
    with torch.no_grad():
        for parameter in decay_params + no_decay_params:
            parameter.add_(0.5)
    before_decay = [p.detach().clone() for p in decay_params]
    before_no_decay = [p.detach().clone() for p in no_decay_params]
    for parameter in decay_params + no_decay_params:
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    # AdamW still decays a parameter whose gradient is exactly zero.
    for before, after in zip(before_decay, decay_params):
        assert not torch.allclose(before, after)
    # WD=0 means the no-decay group's values stay put.
    for before, after in zip(before_no_decay, no_decay_params):
        assert torch.allclose(before, after)


def _saved_group(count: int, weight_decay: float, offset: int = 0) -> dict:
    return {
        "lr": 5e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": weight_decay,
        "amsgrad": False,
        # ``Optimizer.state_dict()`` numbers parameters 0..N-1 in group order.
        "params": list(range(offset, offset + count)),
    }


def test_pretrain_resume_rejects_nonzero_wd_in_the_only_no_decay_group():
    """A no-decay-only layout has no decay coefficient that could be nonzero."""
    trainer, student, patch, objective, groups = _wd_model()
    params = [torch.nn.Parameter(torch.zeros(2)) for _ in range(2)]
    optimizer = torch.optim.AdamW(
        [{"params": params, "weight_decay": 0.0}], lr=5e-4
    )
    corrupt = {"param_groups": [_saved_group(2, 0.1)], "state": {}}
    with pytest.raises(ValueError, match="must be exactly 0.0"):
        trainer._validate_pretrain_optimizer_layout(
            optimizer,
            corrupt,
            norm_bias_weight_decay=False,
            decay_partition_empty=True,
            label="pretrain resume",
        )


def test_pretrain_resume_rejects_noncanonical_optimizer_ids():
    """Equal-shaped parameters make a permuted id order invisible otherwise."""
    trainer, student, patch, objective, groups = _wd_model()
    decay, no_decay = (group["params"] for group in groups)
    optimizer = torch.optim.AdamW(groups, lr=5e-4, weight_decay=0.05)
    saved = {
        "param_groups": [
            _saved_group(len(decay), 0.05, offset=0),
            _saved_group(len(no_decay), 0.0, offset=len(decay)),
        ],
        "state": {},
    }
    trainer._validate_pretrain_optimizer_layout(
        optimizer, saved, norm_bias_weight_decay=False, label="pretrain resume"
    )
    # Two same-shaped parameters: every shape/count check still passes, only the
    # recorded id order shows that the moments belong to other parameters.
    same_shape = next(
        index
        for index in range(1, len(decay))
        if decay[index].shape == decay[0].shape
    )
    permuted = {
        "param_groups": [dict(group, params=list(group["params"])) for group in saved["param_groups"]],
        "state": {},
    }
    ids = permuted["param_groups"][0]["params"]
    ids[0], ids[same_shape] = ids[same_shape], ids[0]
    with pytest.raises(ValueError, match="parameter ids"):
        trainer._validate_pretrain_optimizer_layout(
            optimizer, permuted, norm_bias_weight_decay=False, label="pretrain resume"
        )


def test_pretrain_resume_wd_single_group_semantics(capsys):
    trainer, student, patch, objective, groups = _wd_model()
    assert len(groups) == 2
    two_group_optimizer = torch.optim.AdamW(groups, lr=5e-4, weight_decay=0.05)
    saved_two = {
        "param_groups": [
            _saved_group(len(groups[0]["params"]), 0.05, offset=0),
            _saved_group(
                len(groups[1]["params"]), 0.0, offset=len(groups[0]["params"])
            ),
        ],
        "state": {},
    }
    trainer._validate_pretrain_optimizer_layout(
        two_group_optimizer,
        saved_two,
        norm_bias_weight_decay=False,
        label="pretrain resume",
    )
    # A nonzero no-decay coefficient is corrupt, not a policy choice.
    corrupt = {
        "param_groups": [
            _saved_group(len(groups[0]["params"]), 0.05, offset=0),
            _saved_group(
                len(groups[1]["params"]), 0.05, offset=len(groups[0]["params"])
            ),
        ],
        "state": {},
    }
    with pytest.raises(ValueError, match="must be exactly 0.0"):
        trainer._validate_pretrain_optimizer_layout(
            torch.optim.AdamW(groups, lr=5e-4, weight_decay=0.05),
            corrupt,
            norm_bias_weight_decay=False,
            label="pretrain resume",
        )

    # A no-decay-only layout has no decay coefficient to reconstruct or compare.
    params = [torch.nn.Parameter(torch.zeros(2)) for _ in range(2)]
    no_decay_only = torch.optim.AdamW(
        [{"params": params, "weight_decay": 0.0}], lr=5e-4
    )
    saved_one = {"param_groups": [_saved_group(2, 0.0)], "state": {}}
    trainer._validate_pretrain_optimizer_layout(
        no_decay_only,
        saved_one,
        norm_bias_weight_decay=False,
        label="pretrain resume",
    )
    no_decay_only.load_state_dict(saved_one)
    trainer._report_restored_weight_decay(
        no_decay_only,
        requested_weight_decay=0.2,
        norm_bias_weight_decay=False,
        has_decay_partition=False,
        label="pretrain resume",
    )
    out = capsys.readouterr().out
    assert "no decay group" in out
    assert "weight decay 0.0" in out
    assert "resumed weight decay is" not in out

    # The mirrored decay-only single group still reports the saved coefficient.
    decay_only = torch.optim.AdamW(
        [{"params": params, "weight_decay": 0.05}], lr=5e-4
    )
    saved_decay_only = {"param_groups": [_saved_group(2, 0.05)], "state": {}}
    trainer._validate_pretrain_optimizer_layout(
        decay_only,
        saved_decay_only,
        norm_bias_weight_decay=False,
        label="pretrain resume",
    )
    decay_only.load_state_dict(saved_decay_only)
    trainer._report_restored_weight_decay(
        decay_only,
        requested_weight_decay=0.2,
        norm_bias_weight_decay=False,
        has_decay_partition=True,
        label="pretrain resume",
    )
    out = capsys.readouterr().out
    assert "resumed weight decay is 0.05" in out


def test_pretrain_resume_rejects_layout_conversion():
    trainer, student, patch, objective, groups = _wd_model()
    single = torch.optim.AdamW(
        trainer._pretrain_optimizer_groups(
            student,
            patch,
            objective,
            weight_decay=0.05,
            norm_bias_weight_decay=True,
        ),
        lr=5e-4,
        weight_decay=0.05,
    )
    saved_two = {
        "param_groups": [
            _saved_group(len(groups[0]["params"]), 0.05, offset=0),
            _saved_group(
                len(groups[1]["params"]), 0.0, offset=len(groups[0]["params"])
            ),
        ],
        "state": {},
    }
    # A two-group checkpoint is never converted into the historical single group.
    with pytest.raises(ValueError, match="never converted|parameter group"):
        trainer._validate_pretrain_optimizer_layout(
            single,
            saved_two,
            norm_bias_weight_decay=True,
            label="pretrain resume",
        )
    # A group whose size changed is rejected rather than remapped by shape.
    wrong_size = {"param_groups": [_saved_group(1, 0.05)], "state": {}}
    with pytest.raises(ValueError, match="cannot be mapped by shape"):
        trainer._validate_pretrain_optimizer_layout(
            single, wrong_size, norm_bias_weight_decay=True, label="pretrain resume"
        )
