"""DINO checkpoint publication and deterministic resume."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from typer.testing import CliRunner

from otuformer.cli.main import app
from otuformer.training import trainer

MODEL = "vit_tiny_patch16_224"
SIZE = 224
runner = CliRunner()


def _images(root: Path, rows: int = 2) -> Path:
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest = root / "data.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image"])
        for index in range(rows):
            Image.new("RGB", (SIZE, SIZE), (index * 30, 20, 60)).save(
                images / f"{index}.png"
            )
            writer.writerow([f"{index}.png"])
    return manifest


def _args(root: Path, **overrides) -> argparse.Namespace:
    base = dict(
        train_data=str(root / "data.csv"),
        input_images_dir=str(root / "images"),
        out_dir=str(root / "out"),
        overwrite=False,
        model_name=MODEL,
        register_tokens=None,
        model_name_explicit=False,
        out_dim=8,
        out_dim_explicit=True,
        ssl_method="dino",
        ssl_method_explicit=True,
        dino_prototypes=11,
        dino_prototypes_explicit=True,
        dino_centering="ema",
        dino_centering_explicit=True,
        koleo_weight=0.1,
        koleo_weight_explicit=True,
        max_epochs=1,
        lr=5e-4,
        weight_decay=0.05,
        warmup_epochs=0,
        augmentation="global-barcode",
        orientation_policy="sensitive",
        global_crop_size=SIZE,
        local_crop_size=96,
        local_crops=0,
        mask_ratio="auto",
        lambda_local=1.5,
        lambda_mask=1.0,
        patch_loss="none",
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
        disable_cross_view_loss=False,
        disable_cross_view_loss_explicit=False,
        teacher_temp_warmup_ratio=0.7,
        teacher_temp_warmup_ratio_explicit=False,
        no_norm_bias_weight_decay=False,
        no_norm_bias_weight_decay_explicit=False,
        resume="",
        log_every_n_steps=1,
        save_every_epochs=1,
        keep_last_checkpoints=2,
        visualize_data="",
        extract_size=None,
        metrics_sample_size=1000,
        umap_n_neighbors=5,
        umap_min_dist=0.1,
        umap_metric="cosine",
        visualize_class_number=5,
        compute_embedding_metrics=False,
        batch_size=2,
        num_workers=0,
        cpus=1,
        device="cpu",
        seed=42,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _saved(root: Path) -> dict:
    return torch.load(root / "out" / "SSL_latest.pth", weights_only=False)


def _train(root: Path, **overrides) -> dict:
    _images(root, rows=overrides.pop("rows", 2))
    trainer.run_pretrain(_args(root, **overrides))
    return _saved(root)


@pytest.mark.parametrize("entry", ["cli_new", "cli_resume"])
@pytest.mark.parametrize("corruption", ["keys", "forward", "gradient"])
def test_dino_prototype_contract_fails_before_output(
    tmp_path, monkeypatch, entry, corruption
):
    from otuformer.training.dino import WeightNormLinear

    _images(tmp_path)
    checkpoint = _train(tmp_path) if entry == "cli_resume" else None
    path = tmp_path / "source.pth"
    if checkpoint is not None:
        torch.save(checkpoint, path)
    real_init, real_forward = WeightNormLinear.__init__, WeightNormLinear.forward
    if corruption == "keys":
        def broken_init(self, *args, **kwargs):
            real_init(self, *args, **kwargs)
            self.register_buffer("unexpected", torch.zeros(1))
        monkeypatch.setattr(WeightNormLinear, "__init__", broken_init)
    else:
        def broken_forward(self, x):
            y = real_forward(self, x)
            return y[:, :1] if corruption == "forward" else y.detach().requires_grad_()
        monkeypatch.setattr(WeightNormLinear, "forward", broken_forward)
    out_dir = tmp_path / "run"
    out_dir.mkdir(exist_ok=True)
    sentinel = out_dir / "keep.txt"
    sentinel.write_bytes(b"original")
    logs = out_dir / "logs"
    logs.mkdir(exist_ok=True)
    log = logs / "pretrain.log"
    log.write_bytes(b"previous log\n")
    options = ["--resume", str(path)] if checkpoint is not None else [
        "--ssl-method", "dino", "--overwrite",
    ]
    result = runner.invoke(app, [
        "pretrain", "--input-images-dir", str(tmp_path / "images"),
        "--train-data", str(tmp_path / "data.csv"),
        "--out-dir", str(out_dir), "--model-name", MODEL,
        "--device", "cpu", "--num-workers", "0", *options,
    ])
    assert result.exit_code != 0
    assert "prototype" in result.output
    assert sentinel.read_bytes() == b"original"
    assert log.read_bytes() == b"previous log\n"
    assert {p.name for p in logs.iterdir()} == {"pretrain.log"}



def test_direct_dino_prototype_contract_fails_before_mkdir(tmp_path, monkeypatch):
    from otuformer.training.dino import WeightNormLinear

    _images(tmp_path)
    real_init = WeightNormLinear.__init__

    def broken_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        self.register_buffer("unexpected", torch.zeros(1))

    monkeypatch.setattr(WeightNormLinear, "__init__", broken_init)
    with pytest.raises(ValueError, match="DINO prototype layer"):
        trainer.run_pretrain(_args(tmp_path))
    assert not (tmp_path / "out").exists()


def test_dino_checkpoint_owns_prototypes_without_top_level_center(tmp_path):
    saved = _train(tmp_path, dino_centering="ema")

    assert set(saved["dino_objective"]) == {
        "student_prototypes.weight_g",
        "student_prototypes.weight_v",
        "teacher_prototypes.weight_g",
        "teacher_prototypes.weight_v",
        "cls_center",
    }
    assert saved["dino_objective"]["student_prototypes.weight_g"].shape == (11, 1)
    assert saved["dino_objective"]["student_prototypes.weight_v"].shape == (11, 8)
    assert saved["dino_objective"]["cls_center"].shape == (1, 11)
    assert "center" not in saved
    assert torch.count_nonzero(saved["teacher"]["center"]) == 0
    assert torch.count_nonzero(saved["student"]["center"]) == 0
    for key in ("student", "teacher", "model_state_dict"):
        assert not any(
            "prototype" in name or name.startswith("cls_center")
            for name in saved[key]
        )
    config = saved["config"]
    assert config["ssl_method"] == "dino"
    assert config["ssl_head"] == "dino_mlp_2048"
    assert config["out_dim"] == 8
    assert config["dino_prototypes"] == 11
    assert config["dino_centering"] == "ema"
    assert config["koleo_weight"] == 0.1
    assert config["dino_center_momentum"] == 0.9
    assert "dino_sinkhorn_iterations" not in config
    assert config["dino_weight_norm_api"] == "explicit"
    assert config["dino_prototype_scale"] == 1.0


def test_relational_checkpoint_omits_dino_fields(tmp_path):
    saved = _train(
        tmp_path,
        ssl_method="relational",
        ssl_method_explicit=False,
        out_dim_explicit=False,
        dino_prototypes=64,
        dino_centering="ema",
        koleo_weight=0.0,
        dino_prototypes_explicit=False,
        dino_centering_explicit=False,
        koleo_weight_explicit=False,
    )

    assert saved["config"]["ssl_method"] == "relational"
    assert saved["config"]["ssl_head"] == "projection_mlp_2048"
    assert "dino_objective" not in saved
    assert "center" in saved
    for name in (
        "dino_prototypes",
        "dino_centering",
        "koleo_weight",
        "dino_hidden_dim",
        "dino_sinkhorn_iterations",
        "dino_center_momentum",
    ):
        assert name not in saved["config"], name
    # Invocation args may still carry inactive defaults.
    assert saved["args"]["dino_prototypes"] == 64
    assert "ssl_config_effective" not in saved["args"]
    assert "resume_state" not in saved["args"]


def test_dino_checkpoint_preserves_requested_out_dim(tmp_path):
    saved = _train(tmp_path, out_dim=256)

    assert saved["args"]["out_dim"] == 256
    assert saved["config"]["out_dim"] == 256
    assert "ssl_config_effective" not in saved["args"]


def test_dino_resume_constructs_both_encoders_with_inherited_dim(tmp_path):
    _images(tmp_path)
    first = _args(tmp_path, out_dim=17, max_epochs=1)
    trainer.run_pretrain(first)

    dims = []
    real_init = trainer.OTUFormerEncoder.__init__

    def init_spy(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        dims.append(int(self.projector.net[-1].out_features))

    resume = _args(
        tmp_path,
        out_dim=256,
        out_dim_explicit=False,
        ssl_method="relational",
        ssl_method_explicit=False,
        dino_prototypes_explicit=False,
        dino_centering_explicit=False,
        koleo_weight_explicit=False,
        resume=str(tmp_path / "out" / "SSL_latest.pth"),
        max_epochs=2,
    )
    trainer.OTUFormerEncoder.__init__ = init_spy
    try:
        trainer.run_pretrain(resume)
    finally:
        trainer.OTUFormerEncoder.__init__ = real_init

    # Both encoders (and the RNG-isolated validation probes) build at D=17.
    assert len(dims) >= 2
    assert set(dims) == {17}
    assert resume.out_dim == 256


class _Interrupted(Exception):
    """Stop a run right after its first saved epoch."""


@pytest.mark.parametrize("centering", ["ema", "sinkhorn-knopp"])
def test_dino_same_plan_resume_matches_uninterrupted(tmp_path, monkeypatch, centering):
    straight = tmp_path / "straight"
    _train(straight, max_epochs=3, dino_centering=centering, rows=4)
    reference = _saved(straight)

    split = tmp_path / "split"
    _images(split, rows=4)
    # Interrupt right after epoch 0 is published (save + SSL_latest copy), so
    # the resumed run continues the same plan from a complete checkpoint.
    import shutil

    real_copy = shutil.copy

    def copy_spy(source, destination):
        result = real_copy(source, destination)
        if str(source).endswith("SSL_epoch_0001.pth"):
            raise _Interrupted
        return result

    monkeypatch.setattr(trainer.shutil, "copy", copy_spy)
    with pytest.raises(_Interrupted):
        trainer.run_pretrain(_args(split, max_epochs=3, dino_centering=centering))
    monkeypatch.setattr(trainer.shutil, "copy", real_copy)

    # The interruption left a complete epoch-0 checkpoint with the same plan.
    resumed_partial = _saved(split)
    assert resumed_partial["epoch"] == 0
    assert resumed_partial["iteration"] == 2

    trainer.run_pretrain(
        _args(
            split,
            max_epochs=3,
            dino_centering=centering,
            resume=str(split / "out" / "SSL_latest.pth"),
        )
    )
    resumed = _saved(split)

    for section in ("student", "teacher", "dino_objective"):
        for name, value in reference[section].items():
            torch.testing.assert_close(resumed[section][name], value, rtol=0, atol=0)
    for name, value in reference["optimizer"]["param_groups"][0].items():
        if name == "params":
            assert resumed["optimizer"]["param_groups"][0]["params"] == value
        else:
            assert resumed["optimizer"]["param_groups"][0][name] == value
    # Optimizer *state* (AdamW step/exp_avg/exp_avg_sq per parameter) must match
    # exactly, not just the hyperparameter groups.
    assert set(map(str, reference["optimizer"]["state"])) == set(
        map(str, resumed["optimizer"]["state"])
    )
    for key, entries in reference["optimizer"]["state"].items():
        resumed_entries = resumed["optimizer"]["state"][key]
        assert set(entries) == set(resumed_entries)
        for name, tensor in entries.items():
            torch.testing.assert_close(
                resumed_entries[name], tensor, rtol=0, atol=0
            )
    assert reference["epoch"] == resumed["epoch"]
    assert reference["iteration"] == resumed["iteration"]
    assert torch.equal(reference["rng_state"]["torch"], resumed["rng_state"]["torch"])

    straight_rows = _loss_rows(straight)
    split_rows = _loss_rows(split)
    assert len(straight_rows) == len(split_rows) == 6
    for expected, actual in zip(straight_rows, split_rows):
        assert actual == expected


def _loss_rows(root: Path) -> list[dict[str, str]]:
    with (root / "out" / "logs" / "instant_metrics.pretrain.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return list(csv.DictReader(handle))


def test_dino_extension_does_not_reinitialize_or_refreeze(tmp_path, monkeypatch):
    _images(tmp_path, rows=2)
    trainer.run_pretrain(
        _args(tmp_path, batch_size=1, koleo_weight=0.0, koleo_weight_explicit=True,
              patch_loss="none")
    )
    saved = _saved(tmp_path)
    assert saved["epoch"] == 0
    assert saved["iteration"] == 2

    # A longer loader (8 steps per epoch) is an allowed schedule extension; the
    # restored absolute epoch is 1, so the direction must train immediately.
    manifest = _images(tmp_path, rows=8)
    events = {"projector_inits": 0, "teacher_syncs": 0, "clip_prototype": [],
              "direction": []}
    real_init = trainer._initialize_dino_student_projector
    real_sync = trainer._synchronize_dino_teacher
    real_clip = torch.nn.utils.clip_grad_norm_
    captured = []
    real_objective = trainer.DINOObjective

    def objective_spy(*args, **kwargs):
        objective = real_objective(*args, **kwargs)
        captured.append(objective)
        return objective

    monkeypatch.setattr(trainer, "DINOObjective", objective_spy)
    monkeypatch.setattr(
        trainer,
        "_initialize_dino_student_projector",
        lambda student: events.__setitem__("projector_inits", events["projector_inits"] + 1),
    )
    monkeypatch.setattr(
        trainer,
        "_synchronize_dino_teacher",
        lambda student, teacher, objective: events.__setitem__(
            "teacher_syncs", events["teacher_syncs"] + 1
        ),
    )

    def clip_spy(params, max_norm, **kwargs):
        direction = captured[0].trainable_parameters()[0]
        events["clip_prototype"].append(direction.grad is not None)
        events["direction"].append(direction.detach().clone())
        return real_clip(params, max_norm, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip_spy)

    trainer.run_pretrain(
        _args(
            tmp_path,
            batch_size=1,
            koleo_weight=0.0,
            koleo_weight_explicit=True,
            max_epochs=2,
            resume=str(tmp_path / "out" / "SSL_latest.pth"),
        )
    )

    assert events["projector_inits"] == 0
    assert events["teacher_syncs"] == 0
    assert events["clip_prototype"][0] is True
    assert not torch.equal(events["direction"][0], events["direction"][-1])
    assert manifest.exists()


def _seed_log(out_dir: Path) -> tuple[Path, bytes]:
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / "pretrain.log"
    log_path.write_bytes(b"previous dino run log\n")
    return log_path, log_path.read_bytes()


def _resume_cli(
    tmp_path: Path, checkpoint: Path, out_dir: Path, max_epochs: int | None = None
):
    extra = [] if max_epochs is None else ["--max-epochs", str(max_epochs)]
    return runner.invoke(
        app,
        [
            "pretrain",
            *extra,
            "--resume",
            str(checkpoint),
            "--input-images-dir",
            str(tmp_path / "images"),
            "--train-data",
            str(tmp_path / "data.csv"),
            "--out-dir",
            str(out_dir),
            "--out-dim",
            "8",
            "--device",
            "cpu",
            "--num-workers",
            "0",
        ],
    )


@pytest.mark.parametrize(
    "corruption",
    ["missing_objective", "truncated_objective", "missing_optimizer",
     "missing_schedule", "wrong_scale", "unexpected_center"],
)
def test_dino_incomplete_resume_fails_before_logs(tmp_path, corruption):
    saved = _train(tmp_path / "source")
    out_dir = tmp_path / "source" / "out"
    log_path, log_bytes = _seed_log(out_dir)
    checkpoint = dict(saved)
    if corruption == "missing_objective":
        checkpoint.pop("dino_objective")
    elif corruption == "truncated_objective":
        checkpoint["dino_objective"] = {
            "student_prototypes.weight_v": saved["dino_objective"][
                "student_prototypes.weight_v"
            ]
        }
    elif corruption == "missing_optimizer":
        checkpoint.pop("optimizer")
    elif corruption == "missing_schedule":
        checkpoint.pop("schedule")
    elif corruption == "wrong_scale":
        objective = dict(saved["dino_objective"])
        objective["teacher_prototypes.weight_g"] = torch.full((11, 1), 0.5)
        checkpoint["dino_objective"] = objective
    else:
        objective = dict(saved["dino_objective"])
        del objective["cls_center"]
        checkpoint["dino_objective"] = objective

    path = tmp_path / "corrupt.pth"
    torch.save(checkpoint, path)
    before = set(p.name for p in (out_dir / "logs").iterdir())

    result = _resume_cli(tmp_path / "source", path, out_dir)

    assert result.exit_code != 0
    assert log_path.read_bytes() == log_bytes
    assert set(p.name for p in (out_dir / "logs").iterdir()) == before


@pytest.mark.parametrize("key", ["student", "teacher", "model_state_dict"])
@pytest.mark.parametrize("corruption", ["missing", "dropped", "wrong_dim"])
def test_dino_corrupt_encoder_resume_preserves_logs(tmp_path, key, corruption):
    saved = _train(tmp_path / "source")
    out_dir = tmp_path / "source" / "out"
    log_path, log_bytes = _seed_log(out_dir)
    checkpoint = {name: value for name, value in saved.items()}
    states = {name: dict(value) for name, value in saved.items() if isinstance(value, dict) and name in ("student", "teacher", "model_state_dict")}
    state = states[key]
    if corruption == "missing":
        checkpoint.pop(key)
    elif corruption == "dropped":
        state.pop("projector.net.4.weight")
        checkpoint[key] = state
    else:
        state["projector.net.4.weight"] = torch.zeros(9, 2048)
        checkpoint[key] = state

    path = tmp_path / "corrupt_encoder.pth"
    torch.save(checkpoint, path)
    result = _resume_cli(tmp_path / "source", path, out_dir)

    assert result.exit_code != 0
    assert log_path.read_bytes() == log_bytes


@pytest.mark.parametrize(
    "corruption",
    ["betas", "final_norm_alias", "iteration_past_end", "iteration_short_of_epoch"],
)
def test_dino_preflight_gaps_fail_before_any_log_write(tmp_path, corruption):
    """These payloads passed the old preflight and failed only in the trainer."""
    saved = _train(tmp_path / "source", max_epochs=2)
    out_dir = tmp_path / "source" / "out"
    log_path, log_bytes = _seed_log(out_dir)
    checkpoint = {name: value for name, value in saved.items()}
    checkpoint["optimizer"] = {
        "state": saved["optimizer"]["state"],
        "param_groups": [dict(saved["optimizer"]["param_groups"][0])],
    }
    if corruption == "betas":
        checkpoint["optimizer"]["param_groups"][0].pop("betas")
    elif corruption == "final_norm_alias":
        for key in ("student", "teacher", "model_state_dict"):
            checkpoint[key] = {
                name: value
                for name, value in saved[key].items()
                if not name.startswith("_pretrain_final_norm.")
            }
    elif corruption == "iteration_past_end":
        checkpoint["iteration"] = saved["schedule"]["total_steps"] + 5
    else:
        # A same-plan resume must sit exactly on a recorded epoch boundary. This
        # checkpoint claims a completed epoch at an iteration the 2 x 1 plan does
        # not place on any boundary, so the schedule could not cover the epochs
        # that still have to train.
        checkpoint["epoch"] = 0
        checkpoint["iteration"] = 0

    path = tmp_path / "gap.pth"
    torch.save(checkpoint, path)
    before = set(p.name for p in (out_dir / "logs").iterdir())

    # Keep the recorded plan so the run is a same-plan resume; extending it would
    # legitimately change the remaining step count.
    result = _resume_cli(tmp_path / "source", path, out_dir, max_epochs=2)

    assert result.exit_code != 0
    assert log_path.read_bytes() == log_bytes
    assert set(p.name for p in (out_dir / "logs").iterdir()) == before


def _mutated(saved: dict, **changes) -> dict:
    payload = dict(saved)
    payload.update(changes)
    return payload


def test_dino_resume_rejects_partial_optimizer_state(tmp_path):
    """A lost per-parameter AdamW state must not silently restart that momentum."""
    saved = _train(tmp_path / "src", max_epochs=2)
    state = dict(saved["optimizer"]["state"])
    state.pop(next(iter(state)))

    checkpoint = _mutated(
        saved, optimizer={"state": state, "param_groups": saved["optimizer"]["param_groups"]}
    )
    path = tmp_path / "partial_state.pth"
    torch.save(checkpoint, path)
    log_path, log_bytes = _seed_log(tmp_path / "src" / "out")

    result = _resume_cli(tmp_path / "src", path, tmp_path / "src" / "out")

    assert result.exit_code != 0
    assert "AdamW" in result.output or "optimizer" in result.output
    assert log_path.read_bytes() == log_bytes


def test_dino_resume_rejects_orphaned_optimizer_state(tmp_path):
    saved = _train(tmp_path / "src", max_epochs=2)
    group = dict(saved["optimizer"]["param_groups"][0])
    group["params"] = list(group["params"])[1:]
    checkpoint = _mutated(
        saved, optimizer={"state": saved["optimizer"]["state"], "param_groups": [group]}
    )
    path = tmp_path / "orphan.pth"
    torch.save(checkpoint, path)
    log_path, log_bytes = _seed_log(tmp_path / "src" / "out")

    result = _resume_cli(tmp_path / "src", path, tmp_path / "src" / "out")

    assert result.exit_code != 0
    assert log_path.read_bytes() == log_bytes


def test_dino_resume_accepts_a_real_epoch_zero_checkpoint(tmp_path):
    """The frozen prototype direction legitimately has no momentum at epoch 0."""
    saved = _train(tmp_path / "src", max_epochs=1)
    assert saved["epoch"] == 0
    resume = _args(
        tmp_path / "src",
        max_epochs=2,
        resume=str(tmp_path / "src" / "out" / "SSL_latest.pth"),
    )

    trainer.run_pretrain(resume)

    assert _saved(tmp_path / "src")["epoch"] == 1


def test_dino_resume_preflight_matches_the_strict_restore(tmp_path):
    saved = _train(tmp_path / "src", max_epochs=2)
    out_dir = tmp_path / "src" / "out"
    log_path, log_bytes = _seed_log(out_dir)

    missing_center = {k: dict(v) if isinstance(v, dict) else v for k, v in saved.items()}
    missing_center["student"] = dict(saved["student"])
    missing_center["student"].pop("center")
    extra_key = {k: dict(v) if isinstance(v, dict) else v for k, v in saved.items()}
    extra_key["student"] = dict(saved["student"], **{"backbone.not_a_layer.weight": torch.zeros(2)})
    null_rng = _mutated(
        saved, rng_state={"python": None, "numpy": None, "torch": None}
    )
    short_schedule = _mutated(
        saved, schedule={**saved["schedule"], "total_steps": 1}
    )
    float_epochs = _mutated(
        saved, schedule={**saved["schedule"], "original_max_epochs": 4.5}
    )

    for index, corrupt in enumerate(
        (missing_center, extra_key, null_rng, short_schedule, float_epochs)
    ):
        path = tmp_path / f"corrupt_{index}.pth"
        torch.save(corrupt, path)
        result = _resume_cli(tmp_path / "src", path, out_dir)
        assert result.exit_code != 0, index
        assert log_path.read_bytes() == log_bytes, index


def test_pretrain_resume_weight_decay_uses_saved_value_with_notice(tmp_path, capsys):
    """The saved optimizer coefficient is authoritative; the request is not."""
    saved = _train(tmp_path, max_epochs=1)
    assert saved["config"]["norm_bias_weight_decay"] is True
    assert saved["config"]["teacher_temp_warmup_ratio"] == 0.7
    assert len(saved["optimizer"]["param_groups"]) == 1
    capsys.readouterr()

    trainer.run_pretrain(
        _args(
            tmp_path,
            max_epochs=2,
            weight_decay=0.2,
            resume=str(tmp_path / "out" / "SSL_latest.pth"),
        )
    )
    out = capsys.readouterr().out
    # The notice reports the loaded value, never the requested one.
    assert "resumed weight decay is 0.05" in out
    assert "invocation value 0.2" in out
    resumed = _saved(tmp_path)
    assert resumed["optimizer"]["param_groups"][0]["weight_decay"] == 0.05
    # The invocation record still carries what the operator asked for.
    assert resumed["args"]["weight_decay"] == 0.2


def test_pretrain_resume_two_group_policy_round_trip(tmp_path, capsys):
    saved = _train(tmp_path, max_epochs=1, no_norm_bias_weight_decay=True)
    assert saved["config"]["norm_bias_weight_decay"] is False
    assert len(saved["optimizer"]["param_groups"]) == 2
    assert saved["optimizer"]["param_groups"][1]["weight_decay"] == 0.0
    capsys.readouterr()

    trainer.run_pretrain(
        _args(
            tmp_path,
            max_epochs=2,
            weight_decay=0.2,
            no_norm_bias_weight_decay=True,
            resume=str(tmp_path / "out" / "SSL_latest.pth"),
        )
    )
    out = capsys.readouterr().out
    assert "resumed weight decay is 0.05" in out
    resumed = _saved(tmp_path)
    assert len(resumed["optimizer"]["param_groups"]) == 2
    assert resumed["optimizer"]["param_groups"][0]["weight_decay"] == 0.05
    assert resumed["optimizer"]["param_groups"][1]["weight_decay"] == 0.0


def test_pretrain_resume_rejects_changed_wd_policy(tmp_path):
    _train(tmp_path, max_epochs=1)
    with pytest.raises(ValueError, match="norm_bias_weight_decay|Cannot change"):
        trainer.run_pretrain(
            _args(
                tmp_path,
                max_epochs=2,
                no_norm_bias_weight_decay=True,
                no_norm_bias_weight_decay_explicit=True,
                resume=str(tmp_path / "out" / "SSL_latest.pth"),
            )
        )


def test_pretrain_recipe_fields_are_published_for_both_methods(tmp_path):
    saved = _train(tmp_path, max_epochs=1)
    assert saved["config"]["teacher_temp_warmup_ratio"] == 0.7
    assert saved["config"]["norm_bias_weight_decay"] is True
    assert saved["args"]["teacher_temp_warmup_ratio"] == 0.7
    assert saved["args"]["no_norm_bias_weight_decay"] is False

    other = tmp_path / "relational"
    other.mkdir()
    _images(other, rows=2)
    trainer.run_pretrain(
        _args(
            other,
            max_epochs=1,
            ssl_method="relational",
            ssl_method_explicit=True,
            # DINO-only options must not be explicitly supplied under relational.
            dino_prototypes=64,
            dino_prototypes_explicit=False,
            dino_centering="ema",
            dino_centering_explicit=False,
            koleo_weight=0.0,
            koleo_weight_explicit=False,
        )
    )
    saved = torch.load(other / "out" / "SSL_latest.pth", weights_only=False)
    assert saved["config"]["ssl_method"] == "relational"
    assert saved["config"]["teacher_temp_warmup_ratio"] == 0.7
    assert saved["config"]["norm_bias_weight_decay"] is True


def test_dino_repeated_extension_with_changed_loader_resumes(tmp_path):
    """A checkpoint produced by a legal extension must accept the next one.

    A 2-step epoch followed by an 8-step epoch records 10 steps, which never
    covers "two completed epochs of 8 recorded steps"; the progress check must
    not reapply the newest loader length to the earlier epochs.
    """
    _images(tmp_path, rows=2)
    trainer.run_pretrain(
        _args(tmp_path, batch_size=1, koleo_weight=0.0, koleo_weight_explicit=True)
    )
    saved = _saved(tmp_path)
    assert (saved["epoch"], saved["iteration"]) == (0, 2)

    latest = str(tmp_path / "out" / "SSL_latest.pth")
    _images(tmp_path, rows=8)
    trainer.run_pretrain(
        _args(
            tmp_path,
            batch_size=1,
            koleo_weight=0.0,
            koleo_weight_explicit=True,
            max_epochs=2,
            resume=latest,
        )
    )
    saved = _saved(tmp_path)
    # Two completed epochs: 2 steps with the short loader, 8 with the long one.
    assert (saved["epoch"], saved["iteration"]) == (1, 10)
    assert saved["schedule"]["steps_per_epoch"] == 8
    assert saved["schedule"]["total_steps"] == 10

    trainer.run_pretrain(
        _args(
            tmp_path,
            batch_size=1,
            koleo_weight=0.0,
            koleo_weight_explicit=True,
            max_epochs=3,
            resume=latest,
        )
    )
    saved = _saved(tmp_path)
    assert (saved["epoch"], saved["iteration"]) == (2, 18)


def test_dino_permuted_optimizer_ids_fail_before_logs(tmp_path):
    """Equal-shaped parameters hide a permuted id order from shape checks."""
    saved = _train(tmp_path / "source", max_epochs=2)
    out_dir = tmp_path / "source" / "out"
    log_path, log_bytes = _seed_log(out_dir)
    group = dict(saved["optimizer"]["param_groups"][0])
    params = list(group["params"])
    state = saved["optimizer"]["state"]
    shapes = [tuple(state[param]["exp_avg"].shape) for param in params]
    pair = next(
        (
            (first, second)
            for first in range(len(params))
            for second in range(first + 1, len(params))
            if shapes[first] == shapes[second]
        ),
        None,
    )
    assert pair is not None, "the fixture needs two equal-shaped parameters"
    params[pair[0]], params[pair[1]] = params[pair[1]], params[pair[0]]
    group["params"] = params
    checkpoint = _mutated(
        saved, optimizer={"state": state, "param_groups": [group]}
    )
    path = tmp_path / "permuted_ids.pth"
    torch.save(checkpoint, path)

    result = _resume_cli(tmp_path / "source", path, out_dir)

    assert result.exit_code != 0
    assert "parameter ids" in result.output or "optimizer" in result.output
    assert log_path.read_bytes() == log_bytes


def test_dino_nonzero_no_decay_group_fails_before_logs(tmp_path):
    """The no-decay coefficient is rejected by the preflight, not mid-run."""
    saved = _train(tmp_path / "source", max_epochs=1, no_norm_bias_weight_decay=True)
    assert len(saved["optimizer"]["param_groups"]) == 2
    out_dir = tmp_path / "source" / "out"
    log_path, log_bytes = _seed_log(out_dir)
    groups = [
        dict(group, params=list(group["params"]))
        for group in saved["optimizer"]["param_groups"]
    ]
    groups[1]["weight_decay"] = 0.1
    checkpoint = _mutated(
        saved, optimizer={"state": saved["optimizer"]["state"], "param_groups": groups}
    )
    path = tmp_path / "nonzero_no_decay.pth"
    torch.save(checkpoint, path)

    result = _resume_cli(tmp_path / "source", path, out_dir)

    assert result.exit_code != 0
    assert "exactly 0.0" in result.output
    assert log_path.read_bytes() == log_bytes


@pytest.mark.parametrize("method", ["relational", "dino"])
@pytest.mark.parametrize("no_decay", [False, True])
def test_pretrain_resume_at_ratio_03_matches_uninterrupted(
    tmp_path, monkeypatch, method, no_decay
):
    """Ratio 0.3 must resume identically for both methods and WD policies."""
    shared = dict(
        max_epochs=2,
        rows=4,
        teacher_temp_warmup_ratio=0.3,
        teacher_temp_warmup_ratio_explicit=True,
        no_norm_bias_weight_decay=no_decay,
        no_norm_bias_weight_decay_explicit=no_decay,
        ssl_method=method,
        ssl_method_explicit=True,
        # Inactive DINO options are never *explicitly* supplied under relational.
        koleo_weight=0.0,
        koleo_weight_explicit=False,
        dino_prototypes=11,
        dino_prototypes_explicit=False,
        dino_centering="ema",
        dino_centering_explicit=False,
    )
    straight = tmp_path / "straight"
    _train(straight, **shared)
    reference = _saved(straight)

    split = tmp_path / "split"
    _images(split, rows=4)
    real_copy = shutil.copy

    def copy_spy(source, destination):
        result = real_copy(source, destination)
        if str(source).endswith("SSL_epoch_0001.pth"):
            raise _Interrupted
        return result

    monkeypatch.setattr(trainer.shutil, "copy", copy_spy)
    with pytest.raises(_Interrupted):
        trainer.run_pretrain(_args(split, **shared))
    monkeypatch.setattr(trainer.shutil, "copy", real_copy)
    assert _saved(split)["iteration"] == 2

    trainer.run_pretrain(
        _args(split, **shared, resume=str(split / "out" / "SSL_latest.pth"))
    )
    resumed = _saved(split)

    assert reference["config"]["teacher_temp_warmup_ratio"] == 0.3
    assert reference["config"]["norm_bias_weight_decay"] is not no_decay
    for section in ("student", "teacher") + (
        ("dino_objective",) if method == "dino" else ()
    ):
        for name, value in reference[section].items():
            torch.testing.assert_close(resumed[section][name], value, rtol=0, atol=0)
    if method == "relational":
        torch.testing.assert_close(resumed["center"], reference["center"], rtol=0, atol=0)
    expected_groups = 2 if no_decay else 1
    assert len(resumed["optimizer"]["param_groups"]) == expected_groups
    for expected, actual in zip(
        reference["optimizer"]["param_groups"], resumed["optimizer"]["param_groups"]
    ):
        for name, value in expected.items():
            assert actual[name] == value
    for key, entries in reference["optimizer"]["state"].items():
        for name, tensor in entries.items():
            torch.testing.assert_close(
                resumed["optimizer"]["state"][key][name], tensor, rtol=0, atol=0
            )
    assert (reference["epoch"], reference["iteration"]) == (
        resumed["epoch"],
        resumed["iteration"],
    )
    assert reference["schedule"] == resumed["schedule"]
    assert reference["rng_state"].keys() == resumed["rng_state"].keys()
    for key, expected in reference["rng_state"].items():
        actual = resumed["rng_state"][key]
        if isinstance(expected, torch.Tensor):
            assert torch.equal(expected, actual)
        elif key == "numpy":
            assert expected[0] == actual[0]
            np.testing.assert_array_equal(expected[1], actual[1])
            assert expected[2:] == actual[2:]
        elif isinstance(expected, list):
            assert len(expected) == len(actual)
            for left, right in zip(expected, actual):
                assert torch.equal(left, right)
        else:
            assert expected == actual
    # Identical schedules and updates, not just identical weights.
    assert _loss_rows(straight) == _loss_rows(split)


@pytest.mark.parametrize("method", ["relational", "dino"])
@pytest.mark.parametrize("insufficient", [False, True])
def test_pretrain_recipe_one_step_remaining_or_insufficient_schedule(
    tmp_path, monkeypatch, method, insufficient
):
    _images(tmp_path)
    shared = dict(
        max_epochs=2, ssl_method=method, koleo_weight=0.0,
        koleo_weight_explicit=False, dino_prototypes_explicit=False,
        dino_centering_explicit=False,
    )
    real_copy = shutil.copy

    def interrupt(source, destination):
        result = real_copy(source, destination)
        if Path(source).name == "SSL_epoch_0001.pth":
            raise _Interrupted
        return result

    monkeypatch.setattr(trainer.shutil, "copy", interrupt)
    with pytest.raises(_Interrupted):
        trainer.run_pretrain(_args(tmp_path, **shared))
    monkeypatch.setattr(trainer.shutil, "copy", real_copy)
    saved = _saved(tmp_path)
    assert (saved["iteration"], saved["schedule"]["total_steps"]) == (1, 2)
    if insufficient:
        saved["schedule"]["total_steps"] = 1 if method == "dino" else 0
        path = tmp_path / "insufficient.pth"
        torch.save(saved, path)
        out_dir = tmp_path / "out"
        log, original = _seed_log(out_dir)
        before = {p.name for p in (out_dir / "logs").iterdir()}
        result = _resume_cli(tmp_path, path, out_dir, max_epochs=2)
        assert result.exit_code != 0
        assert any(text in result.output for text in ("total_steps", "original plan", "beyond"))
        assert log.read_bytes() == original
        assert {p.name for p in (out_dir / "logs").iterdir()} == before
    else:
        trainer.run_pretrain(_args(
            tmp_path, **shared, resume=str(tmp_path / "out" / "SSL_latest.pth")
        ))
        resumed = _saved(tmp_path)
        assert (resumed["epoch"], resumed["iteration"]) == (1, 2)
        assert len(_loss_rows(tmp_path)) == 2


def test_legacy_recipe_checkpoint_resumes_for_real(tmp_path, capsys):
    """A payload without the two common fields resumes with 0.7/True in practice."""
    _images(tmp_path, rows=2)
    trainer.run_pretrain(
        _args(tmp_path, batch_size=1, koleo_weight=0.0, koleo_weight_explicit=True)
    )
    checkpoint = _saved(tmp_path)
    for name in ("teacher_temp_warmup_ratio", "norm_bias_weight_decay"):
        checkpoint["config"].pop(name)
    legacy = tmp_path / "legacy.pth"
    torch.save(checkpoint, legacy)
    capsys.readouterr()

    trainer.run_pretrain(
        _args(
            tmp_path,
            batch_size=1,
            koleo_weight=0.0,
            koleo_weight_explicit=True,
            max_epochs=2,
            resume=str(legacy),
        )
    )
    assert "records neither teacher_temp_warmup_ratio" in capsys.readouterr().out
    resumed = _saved(tmp_path)
    assert resumed["epoch"] == 1
    assert (
        resumed["config"]["teacher_temp_warmup_ratio"],
        resumed["config"]["norm_bias_weight_decay"],
    ) == (0.7, True)
    assert len(resumed["optimizer"]["param_groups"]) == 1


def test_dino_extension_with_shorter_loader_and_nonzero_warmup(tmp_path, monkeypatch):
    """Original warmup metadata must not block a shorter-loader extension.

    4 x 8 plan with 3 warmup epochs records warmup_steps=24. Shrinking the
    loader to 1 step per epoch makes total_steps=12 while warmup_steps stays 24,
    and the *next* extension must still be accepted.
    """
    _images(tmp_path, rows=8)
    real_copy = shutil.copy

    def copy_spy(source, destination):
        result = real_copy(source, destination)
        if str(source).endswith("SSL_epoch_0001.pth"):
            raise _Interrupted
        return result

    shared = dict(
        batch_size=1,
        koleo_weight=0.0,
        koleo_weight_explicit=True,
        warmup_epochs=3,
    )
    monkeypatch.setattr(trainer.shutil, "copy", copy_spy)
    with pytest.raises(_Interrupted):
        trainer.run_pretrain(_args(tmp_path, max_epochs=4, **shared))
    monkeypatch.setattr(trainer.shutil, "copy", real_copy)
    saved = _saved(tmp_path)
    assert (saved["epoch"], saved["iteration"]) == (0, 8)
    assert saved["schedule"]["warmup_steps"] == 24
    assert saved["schedule"]["total_steps"] == 32

    latest = str(tmp_path / "out" / "SSL_latest.pth")
    _images(tmp_path, rows=1)
    trainer.run_pretrain(
        _args(tmp_path, max_epochs=5, resume=latest, **shared)
    )
    saved = _saved(tmp_path)
    assert (saved["epoch"], saved["iteration"]) == (4, 12)
    assert saved["schedule"]["total_steps"] == 12
    assert saved["schedule"]["warmup_steps"] == 24

    trainer.run_pretrain(
        _args(tmp_path, max_epochs=6, resume=latest, **shared)
    )
    assert _saved(tmp_path)["epoch"] == 5


@pytest.mark.parametrize("method", ["relational", "dino"])
@pytest.mark.parametrize("rows", [2, 8])
def test_pretrain_extended_plan_cannot_return_to_original_target(
    tmp_path, monkeypatch, method, rows
):
    """Do not rebuild an original ramp after a saved extension has begun."""
    _images(tmp_path, rows=2)
    shared = dict(
        batch_size=1,
        ssl_method=method,
        koleo_weight=0.0,
        koleo_weight_explicit=False,
        dino_prototypes_explicit=False,
        dino_centering_explicit=False,
    )
    real_copy = shutil.copy

    def interrupt_after(filename):
        def copy_spy(source, destination):
            result = real_copy(source, destination)
            if Path(source).name == filename:
                raise _Interrupted
            return result
        return copy_spy

    monkeypatch.setattr(trainer.shutil, "copy", interrupt_after("SSL_epoch_0001.pth"))
    with pytest.raises(_Interrupted):
        trainer.run_pretrain(_args(tmp_path, max_epochs=4, **shared))
    latest = str(tmp_path / "out" / "SSL_latest.pth")
    _images(tmp_path, rows=rows)
    monkeypatch.setattr(trainer.shutil, "copy", interrupt_after("SSL_epoch_0002.pth"))
    with pytest.raises(_Interrupted):
        trainer.run_pretrain(_args(tmp_path, max_epochs=6, resume=latest, **shared))
    monkeypatch.setattr(trainer.shutil, "copy", real_copy)
    saved = _saved(tmp_path)
    assert saved["args"]["max_epochs"] == 6
    assert saved["epoch"] == 1
    assert float(_loss_rows(tmp_path)[-1]["teacher_temp"]) == pytest.approx(0.07)

    out_dir = tmp_path / "out"
    log_path, log_bytes = _seed_log(out_dir)
    before = {p.name for p in (out_dir / "logs").iterdir()}
    result = runner.invoke(app, [
        "pretrain", "--resume", latest, "--max-epochs", "4",
        "--train-data", str(tmp_path / "data.csv"),
        "--input-images-dir", str(tmp_path / "images"),
        "--out-dir", str(out_dir), "--out-dim", "8",
        "--batch-size", "1", "--num-workers", "0", "--device", "cpu",
    ])

    assert result.exit_code != 0
    assert "extended" in result.output and "original" in result.output
    assert log_path.read_bytes() == log_bytes
    assert {p.name for p in (out_dir / "logs").iterdir()} == before
    assert _saved(tmp_path)["epoch"] == 1


@pytest.mark.parametrize("method,legacy", [
    ("relational", False), ("relational", True), ("dino", False),
])
@pytest.mark.parametrize("target", [1, 2])
def test_pretrain_completed_target_fails_before_logs(tmp_path, method, legacy, target):
    """An exhausted target is rejected even without legacy schedule metadata."""
    saved = _train(
        tmp_path, max_epochs=2, ssl_method=method, koleo_weight=0.0,
        koleo_weight_explicit=False, dino_prototypes_explicit=False,
        dino_centering_explicit=False,
    )
    if legacy:
        saved.pop("schedule")
    path = tmp_path / "resume.pth"
    torch.save(saved, path)
    out_dir = tmp_path / "out"
    log_path, log_bytes = _seed_log(out_dir)
    before = {p.name for p in (out_dir / "logs").iterdir()}

    result = _resume_cli(tmp_path, path, out_dir, max_epochs=target)

    assert result.exit_code != 0
    assert "must exceed" in result.output
    assert log_path.read_bytes() == log_bytes
    assert {p.name for p in (out_dir / "logs").iterdir()} == before


@pytest.mark.parametrize("target", [1, 3])
def test_legacy_pretrain_changed_target_fails_before_logs(
    tmp_path, monkeypatch, target
):
    """A legacy plan cannot change its target, even with epochs left to run."""
    _images(tmp_path)
    real_copy = shutil.copy

    def copy_spy(source, destination):
        result = real_copy(source, destination)
        if Path(source).name == "SSL_epoch_0001.pth":
            raise _Interrupted
        return result

    shared = dict(
        ssl_method="relational", koleo_weight=0.0,
        koleo_weight_explicit=False, dino_prototypes_explicit=False,
        dino_centering_explicit=False,
    )
    monkeypatch.setattr(trainer.shutil, "copy", copy_spy)
    with pytest.raises(_Interrupted):
        trainer.run_pretrain(_args(tmp_path, max_epochs=2, **shared))
    monkeypatch.setattr(trainer.shutil, "copy", real_copy)
    saved = _saved(tmp_path)
    assert saved["epoch"] == 0
    saved.pop("schedule")
    checkpoint = tmp_path / "legacy.pth"
    torch.save(saved, checkpoint)
    out_dir = tmp_path / "out"
    log_path, log_bytes = _seed_log(out_dir)
    before = {p.name for p in (out_dir / "logs").iterdir()}

    result = _resume_cli(tmp_path, checkpoint, out_dir, max_epochs=target)

    assert result.exit_code != 0
    assert log_path.read_bytes() == log_bytes
    assert {p.name for p in (out_dir / "logs").iterdir()} == before

    trainer.run_pretrain(
        _args(tmp_path, max_epochs=2, resume=str(checkpoint), **shared)
    )
    resumed = _saved(tmp_path)
    assert (resumed["epoch"], resumed["iteration"]) == (1, 2)


@pytest.mark.parametrize("no_decay", [False, True])
def test_relational_partition_size_error_fails_before_logs(tmp_path, no_decay):
    """The rebuilt partition is proven before output for relational resumes."""
    saved = _train(
        tmp_path / "source",
        max_epochs=1,
        no_norm_bias_weight_decay=no_decay,
        ssl_method="relational",
        ssl_method_explicit=True,
        koleo_weight=0.0,
        koleo_weight_explicit=False,
        dino_prototypes=11,
        dino_prototypes_explicit=False,
        dino_centering="ema",
        dino_centering_explicit=False,
    )
    expected_groups = 2 if no_decay else 1
    assert len(saved["optimizer"]["param_groups"]) == expected_groups
    out_dir = tmp_path / "source" / "out"
    log_path, log_bytes = _seed_log(out_dir)
    before = set(path.name for path in (out_dir / "logs").iterdir())

    groups = [
        dict(group, params=list(group["params"]))
        for group in saved["optimizer"]["param_groups"]
    ]
    state = dict(saved["optimizer"]["state"])
    # Dropping the last id keeps the remaining order canonical, so only a
    # rebuilt-partition size check can catch it.
    state.pop(groups[expected_groups - 1]["params"].pop())
    checkpoint = _mutated(
        saved, optimizer={"state": state, "param_groups": groups}
    )
    path = tmp_path / "short_partition.pth"
    torch.save(checkpoint, path)

    result = _resume_cli(tmp_path / "source", path, out_dir)

    assert result.exit_code != 0
    assert "rebuilds" in result.output or "parameter" in result.output
    assert log_path.read_bytes() == log_bytes
    assert set(path_.name for path_ in (out_dir / "logs").iterdir()) == before
