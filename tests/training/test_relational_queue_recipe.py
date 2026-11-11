"""Model-free queue request resolution and recovery-lock tests."""

from __future__ import annotations

import argparse

import pytest
import torch

import otuformer.training.pretrain_recipe as recipe
from otuformer.training.relational_queue import RelationalQueue


def _write_images(tmp_path, count=4):
    from PIL import Image

    names = []
    for index in range(count):
        name = f"img_{index}.jpg"
        Image.new("RGB", (32, 32), color=(index * 30, 10, 5)).save(tmp_path / name)
        names.append(name)
    csv = tmp_path / "images.csv"
    csv.write_text("image\n" + "\n".join(names) + "\n")
    return csv, names


def _request(tmp_path, csv, **overrides):
    values = dict(
        train_data=str(csv),
        input_images_dir=str(tmp_path),
        resume="",
        batch_size=2,
        batch_size_explicit=False,
        device="cpu",
        model_name="vit_tiny_patch16_224",
        out_dim=8,
        global_crop_size=64,
        local_crop_size=16,
        local_crops=1,
        max_epochs=2,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _encoder_state(out_dim=8, image_size=64):
    from otuformer.training.model import OTUFormerEncoder

    model = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224",
        out_dim=out_dim,
        pretrained=False,
        img_size=image_size,
    )
    return model.state_dict(), out_dim


def _enabled_payload(
    tmp_path,
    names,
    *,
    capacity=2,
    batch=2,
    epoch=0,
    keys=None,
    observed=None,
    fingerprint=None,
    image_count=None,
    size_bytes=None,
    out_dim=8,
    schedule_overrides=None,
    drop=(),
):
    from otuformer.data.provenance import canonical_digest, snapshot_bytes

    state_dict, out_dim = _encoder_state(out_dim)
    steps_per_epoch = len(names) // batch
    iteration = (epoch + 1) * steps_per_epoch
    if keys is None:
        keys = list(names[: min(capacity, batch)])
    if observed is None:
        observed = [iteration - 1] * len(keys)
    if fingerprint is None:
        rows = []
        for name in names:
            data, snapshot = snapshot_bytes(tmp_path / name)
            rows.append(
                {
                    "image_key": name,
                    "size_bytes": int(snapshot.size_bytes),
                    "sha256": snapshot.sha256,
                }
            )
        fingerprint = canonical_digest(rows)
    if image_count is None:
        image_count = len(names)
    if size_bytes is None:
        size_bytes = sum((tmp_path / name).stat().st_size for name in names)
    schedule = {
        "steps_per_epoch": steps_per_epoch,
        "total_steps": iteration,
        "warmup_steps": 0,
        "original_max_epochs": epoch + 1,
        "last_lr": 0.001,
        "final_lr": 0.0001,
    }
    schedule.update(schedule_overrides or {})
    import random

    import numpy as np

    queue_state = {
        "version": 1,
        "keys": list(keys),
        "features": torch.zeros(len(keys), 2, out_dim),
        "observed_steps": list(observed),
        "completed_steps": iteration,
        "data_fingerprint": fingerprint,
        "data_image_count": image_count,
        "data_size_bytes": size_bytes,
    }
    payload = {
        "config": {
            "model_name": "vit_tiny_patch16_224",
            "out_dim": out_dim,
            "image_size": 64,
            "ssl_method": "relational",
            "ssl_head": "projection_mlp_2048",
            # A disabled KoLeo regularizer is encoded by *absence*, exactly as a
            # zero-weight relational run writes it.
            "relational_queue_config": {"version": 1, "capacity": capacity},
        },
        "epoch": epoch,
        "iteration": iteration,
        "schedule": schedule,
        "args": {
            "batch_size": batch,
            "max_epochs": epoch + 1,
            "relational_queue_size": capacity,
            "requested_relational_queue_size": None,
        },
        "optimizer": {"param_groups": [{"params": [], "lr": 0.001}], "state": {}},
        "student": state_dict,
        "teacher": state_dict,
        "center": torch.zeros(1, out_dim),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
        "relational_queue": queue_state,
    }
    for key in drop:
        payload.pop(key, None)
    return payload


def test_fifo_data_identity_precedes_optimizer_execution_probe(tmp_path, monkeypatch):
    csv, _ = _write_images(tmp_path)
    events = []
    prepare_data = recipe.prepare_relational_data
    validate_optimizer = recipe._validate_queue_optimizer_precision

    def prepare(*args, **kwargs):
        result = prepare_data(*args, **kwargs)
        events.append("complete_identity")
        return result

    def validate(*args, **kwargs):
        assert events == ["complete_identity"]
        events.append("optimizer_probe")
        return validate_optimizer(*args, **kwargs)

    monkeypatch.setattr(recipe, "prepare_relational_data", prepare)
    monkeypatch.setattr(recipe, "_validate_queue_optimizer_precision", validate)
    recipe.prepare_pretrain_relational_queue(
        _request(tmp_path, csv, relational_queue_size=2, weight_decay=0.04), None,
    )
    assert events == ["complete_identity", "optimizer_probe"]


def test_fifo_direct_omission_stays_legacy_off():
    prep = recipe.prepare_pretrain_relational_queue(
        argparse.Namespace(batch_size=32), None
    )
    assert prep.ssl_method == "relational" and prep.capacity == 0
    assert prep.data is None and prep.requested_capacity is None
    assert prep.effective_batch_size == 32


def test_fifo_dino_explicit_zero_is_rejected():
    args = argparse.Namespace(
        ssl_method="dino",
        batch_size=32,
        relational_queue_size=0,
        relational_queue_size_explicit=True,
    )
    with pytest.raises(ValueError, match="relational"):
        recipe.prepare_pretrain_relational_queue(args, None)


def test_fifo_request_matrix(tmp_path):
    csv, names = _write_images(tmp_path)

    # Fresh relational: omission resolves to the effective zero.
    omitted = recipe.prepare_pretrain_relational_queue(_request(tmp_path, csv), None)
    assert omitted.capacity == 0
    assert omitted.requested_capacity is None
    assert omitted.capacity_source == "default"
    assert omitted.batch_size_source == "default"
    assert omitted.resolved_log()["applicability"] == "disabled"

    explicit_zero = recipe.prepare_pretrain_relational_queue(
        _request(tmp_path, csv, relational_queue_size=0,
                 relational_queue_size_explicit=True),
        None,
    )
    assert explicit_zero.capacity == 0
    assert explicit_zero.requested_capacity == 0
    assert explicit_zero.capacity_source == "explicit"

    positive = recipe.prepare_pretrain_relational_queue(
        _request(tmp_path, csv, relational_queue_size=3,
                 relational_queue_size_explicit=True),
        None,
    )
    assert positive.capacity == 3
    assert positive.capacity_source == "explicit"
    assert positive.data is not None
    assert positive.data.image_keys == tuple(names)
    assert positive.candidate_bound == 2 + min(3, 4 - 2)
    assert recipe.prepare_pretrain_relational_queue(
        _request(tmp_path, csv, relational_queue_size=2,
                 relational_queue_size_explicit=True),
        None,
    ).candidate_bound == 4
    assert positive.resolved_log()["applicability"] == "active"

    # A numeric direct-core request without a marker is an explicit override.
    unmarked = recipe.prepare_pretrain_relational_queue(
        _request(tmp_path, csv, relational_queue_size=1,
                 relational_queue_size_explicit=None),
        None,
    )
    assert unmarked.capacity == 1 and unmarked.capacity_source == "explicit"

    # DINO omission is inactive, never a false active zero.
    dino = recipe.prepare_pretrain_relational_queue(
        _request(tmp_path, csv, ssl_method="dino"), None
    )
    assert dino.capacity is None
    assert dino.capacity_source == "inactive"
    assert dino.resolved_log()["applicability"] == "inactive"
    for value in (0, 4):
        with pytest.raises(ValueError, match="relational"):
            recipe.prepare_pretrain_relational_queue(
                _request(
                    tmp_path,
                    csv,
                    ssl_method="dino",
                    relational_queue_size=value,
                    relational_queue_size_explicit=True,
                ),
                None,
            )

    # Invalid requests are rejected before any preparation.
    for value in (-1, True, 1.5, "3", float("inf")):
        with pytest.raises(ValueError, match="relational-queue-size"):
            recipe.prepare_pretrain_relational_queue(
                _request(tmp_path, csv, relational_queue_size=value,
                         relational_queue_size_explicit=True),
                None,
            )


def test_fifo_disabled_resume_never_enables_history(tmp_path):
    csv, names = _write_images(tmp_path)
    payload = {
        "config": {"model_name": "vit_tiny_patch16_224", "out_dim": 8},
        "epoch": 0,
        "iteration": 2,
        "schedule": {"steps_per_epoch": 2, "total_steps": 2},
        "args": {"batch_size": 2, "max_epochs": 1, "relational_queue_size": 0},
        "optimizer": {"param_groups": [{"params": [], "lr": 0.001}], "state": {}},
        "student": {},
        "teacher": {},
        "center": torch.zeros(1, 8),
    }
    omitted = recipe.prepare_pretrain_relational_queue(
        _request(tmp_path, csv, resume="runs/x/SSL_latest.pth"), payload
    )
    assert omitted.capacity == 0
    assert omitted.capacity_source == "inherited"
    assert omitted.data is None

    explicit_zero = recipe.prepare_pretrain_relational_queue(
        _request(
            tmp_path,
            csv,
            resume="runs/x/SSL_latest.pth",
            relational_queue_size=0,
            relational_queue_size_explicit=True,
        ),
        payload,
    )
    assert explicit_zero.capacity == 0
    assert explicit_zero.capacity_source == "explicit"

    with pytest.raises(ValueError, match="Cannot enable the relational queue"):
        recipe.prepare_pretrain_relational_queue(
            _request(
                tmp_path,
                csv,
                resume="runs/x/SSL_latest.pth",
                relational_queue_size=8,
                relational_queue_size_explicit=True,
            ),
            payload,
        )

    contradictory = {
        **payload,
        "args": {**payload["args"], "relational_queue_size": 8},
    }
    with pytest.raises(ValueError, match="without the queue config"):
        recipe.prepare_pretrain_relational_queue(
            _request(tmp_path, csv, resume="runs/x/SSL_latest.pth"), contradictory
        )


def test_fifo_resume_inherits_and_conflicts(tmp_path):
    csv, names = _write_images(tmp_path)
    payload = _enabled_payload(tmp_path, names, capacity=2, batch=2)
    resume = "runs/x/SSL_epoch_0001.pth"

    inherited = recipe.prepare_pretrain_relational_queue(
        _request(tmp_path, csv, resume=resume, batch_size=None), payload
    )
    assert inherited.capacity == 2
    assert inherited.capacity_source == "inherited"
    assert inherited.effective_batch_size == 2
    assert inherited.batch_size_source == "inherited"

    matching = recipe.prepare_pretrain_relational_queue(
        _request(
            tmp_path,
            csv,
            resume=resume,
            relational_queue_size=2,
            relational_queue_size_explicit=True,
            batch_size_explicit=True,
        ),
        payload,
    )
    assert matching.capacity_source == "explicit"
    assert matching.batch_size_source == "explicit"

    with pytest.raises(ValueError, match="Cannot change --relational-queue-size"):
        recipe.prepare_pretrain_relational_queue(
            _request(
                tmp_path,
                csv,
                resume=resume,
                relational_queue_size=8,
                relational_queue_size_explicit=True,
            ),
            payload,
        )
    with pytest.raises(ValueError, match="Cannot change --batch-size"):
        recipe.prepare_pretrain_relational_queue(
            _request(tmp_path, csv, resume=resume, batch_size=4, batch_size_explicit=True),
            payload,
        )


def test_fifo_resume_rejects_dino_with_queue_state(tmp_path):
    csv, names = _write_images(tmp_path)
    payload = _enabled_payload(tmp_path, names)
    payload["config"] = {
        **payload["config"],
        "ssl_method": "dino",
        "ssl_head": "dino_mlp_2048",
    }
    with pytest.raises(ValueError, match="DINO"):
        recipe.prepare_pretrain_relational_queue(
            _request(tmp_path, csv, resume="runs/x/SSL_latest.pth"), payload
        )


def test_fifo_bundles_must_be_complete(tmp_path):
    csv, names = _write_images(tmp_path)
    payload = _enabled_payload(tmp_path, names)
    without_state = {key: value for key, value in payload.items() if key != "relational_queue"}
    with pytest.raises(ValueError, match="without relational_queue"):
        recipe.prepare_pretrain_relational_queue(
            _request(tmp_path, csv, resume="runs/x/SSL_latest.pth"), without_state
        )
    without_config = {
        **payload,
        "config": {
            key: value
            for key, value in payload["config"].items()
            if key != "relational_queue_config"
        },
    }
    with pytest.raises(ValueError, match="relational_queue_config"):
        recipe.prepare_pretrain_relational_queue(
            _request(tmp_path, csv, resume="runs/x/SSL_latest.pth"), without_config
        )


def test_fifo_resume_rejects_incomplete_or_nonfinite_schedule(tmp_path):
    csv, names = _write_images(tmp_path)
    resume = "runs/x/SSL_epoch_0001.pth"
    request = _request(
        tmp_path, csv, resume=resume, batch_size=2, batch_size_explicit=True
    )

    payload = _enabled_payload(tmp_path, names, capacity=2, batch=2)
    recipe.prepare_pretrain_relational_queue(request, payload)

    for mutate, match in (
        (lambda p: p["schedule"].pop("final_lr"), r"schedule\.final_lr"),
        (lambda p: p["schedule"].pop("last_lr"), r"schedule\.last_lr"),
        (lambda p: p["schedule"].pop("warmup_steps"), r"schedule\.warmup_steps"),
        (
            lambda p: p["schedule"].update({"final_lr": float("nan")}),
            r"schedule final_lr to be finite",
        ),
        (
            lambda p: p["schedule"].update({"last_lr": float("inf")}),
            r"schedule last_lr to be finite",
        ),
        (
            lambda p: p["schedule"].update({"last_lr": -1e-05}),
            "last_lr to be non-negative",
        ),
        # Python-finite but not finite at the float32 schedule precision.
        (
            lambda p: p["schedule"].update({"last_lr": 1e100}),
            "survive the float32 schedule precision",
        ),
        (
            lambda p: p["schedule"].update({"final_lr": 1e100}),
            "survive the float32 schedule precision",
        ),
        (
            lambda p: p["schedule"].update({"total_steps": 1}),
            "beyond the recorded",
        ),
        (
            lambda p: p["schedule"].update({"original_max_epochs": 0}),
            "positive saved schedule.original_max_epochs",
        ),
    ):
        broken = _enabled_payload(tmp_path, names, capacity=2, batch=2)
        mutate(broken)
        with pytest.raises(ValueError, match=match):
            recipe.prepare_pretrain_relational_queue(request, broken)

    # The invocation's own --lr enters the same float32 schedule.
    broken = _enabled_payload(tmp_path, names, capacity=2, batch=2)
    with pytest.raises(ValueError, match=r"--lr to survive the float32"):
        recipe.prepare_pretrain_relational_queue(
            _request(
                tmp_path,
                csv,
                resume=resume,
                batch_size=2,
                batch_size_explicit=True,
                lr=1e100,
            ),
            broken,
        )


def test_fifo_resume_accepts_a_zero_learning_rate_writer_state(tmp_path):
    """--lr 0 is a legal run, so a zero endpoint must be readable."""
    csv, names = _write_images(tmp_path, count=14)
    schedule_args = argparse.Namespace(
        max_epochs=1,
        lr=0.0,
        warmup_epochs=3,
        teacher_momentum=0.9,
        teacher_momentum_end=0.99,
        student_temp=0.1,
        teacher_temp_start=0.04,
        teacher_temp_end=0.07,
    )
    lr_schedule, _, _, _, metadata = recipe._build_pretrain_schedules(
        schedule_args, None, steps_per_epoch=7, global_step=0, completed_epochs=0
    )
    assert metadata["last_lr"] == 0.0 and metadata["final_lr"] == 0.0
    # The real optimizer uses this schedule and keeps finite parameters.
    parameter = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.AdamW([parameter], lr=float(lr_schedule[0]))
    for step in range(7):
        optimizer.param_groups[0]["lr"] = float(lr_schedule[step])
        parameter.grad = torch.ones(3)
        optimizer.step()
    assert torch.isfinite(parameter).all()
    assert float(optimizer.param_groups[0]["lr"]) == 0.0

    request = _request(
        tmp_path,
        csv,
        resume="runs/x/SSL_epoch_0001.pth",
        batch_size=2,
        batch_size_explicit=True,
        max_epochs=1,
        lr=0.0,
    )
    payload = _enabled_payload(
        tmp_path,
        names,
        capacity=2,
        batch=2,
        keys=names[:2],
        observed=[6, 6],
        schedule_overrides=metadata,
    )
    assert recipe.prepare_pretrain_relational_queue(request, payload).capacity == 2

    # Values that collapse toward zero at float32 stay finite schedules.
    for tiny in (1e-44, 1e-46):
        payload = _enabled_payload(
            tmp_path,
            names,
            capacity=2,
            batch=2,
            keys=names[:2],
            observed=[6, 6],
            schedule_overrides={**metadata, "last_lr": tiny, "final_lr": tiny},
        )
        assert (
            recipe.prepare_pretrain_relational_queue(request, payload).capacity == 2
        )


def test_fifo_resume_accepts_the_writers_clamped_warmup(tmp_path):
    """The writer records warmup_steps the scheduler itself clamps."""
    csv, names = _write_images(tmp_path, count=14)
    schedule_args = argparse.Namespace(
        max_epochs=1,
        lr=0.001,
        warmup_epochs=3,
        teacher_momentum=0.9,
        teacher_momentum_end=0.99,
        student_temp=0.1,
        teacher_temp_start=0.04,
        teacher_temp_end=0.07,
    )
    _, _, _, _, metadata = recipe._build_pretrain_schedules(
        schedule_args, None, steps_per_epoch=7, global_step=0, completed_epochs=0
    )
    # 1 epoch of 7 steps with --warmup-epochs 3: recorded 21, clamped to 7.
    assert metadata["warmup_steps"] == 21
    assert metadata["total_steps"] == 7
    payload = _enabled_payload(
        tmp_path,
        names,
        capacity=2,
        batch=2,
        keys=names[:2],
        observed=[6, 6],
        schedule_overrides=metadata,
    )
    extension = _request(
        tmp_path,
        csv,
        resume="runs/x/SSL_epoch_0001.pth",
        batch_size=2,
        batch_size_explicit=True,
        max_epochs=3,
    )
    preparation = recipe.prepare_pretrain_relational_queue(extension, payload)
    assert preparation.capacity == 2 and preparation.active


def test_fifo_resume_rejects_a_schedule_shorter_than_the_plan(tmp_path):
    """A short total_steps would under-size the rebuilt temperature array."""
    csv, names = _write_images(tmp_path, count=14)
    request = _request(
        tmp_path,
        csv,
        resume="runs/x/SSL_epoch_0001.pth",
        batch_size=2,
        batch_size_explicit=True,
        max_epochs=3,
    )
    for total_steps in (7, 8):
        payload = _enabled_payload(
            tmp_path,
            names,
            capacity=2,
            batch=2,
            keys=names[:2],
            observed=[6, 6],
            schedule_overrides={
                "steps_per_epoch": 7,
                "total_steps": total_steps,
                "original_max_epochs": 3,
                "warmup_steps": 21,
                "last_lr": 0.001,
                "final_lr": 1e-05,
            },
        )
        with pytest.raises(ValueError, match="inconsistent saved schedule"):
            recipe.prepare_pretrain_relational_queue(request, payload)
    # The matching rectangle is accepted.
    payload = _enabled_payload(
        tmp_path,
        names,
        capacity=2,
        batch=2,
        keys=names[:2],
        observed=[6, 6],
        schedule_overrides={
            "steps_per_epoch": 7,
            "total_steps": 21,
            "original_max_epochs": 3,
            "warmup_steps": 21,
            "last_lr": 0.001,
            "final_lr": 1e-05,
        },
    )
    assert (
        recipe.prepare_pretrain_relational_queue(request, payload).capacity == 2
    )


def test_fifo_preparation_checks_the_requested_lr(tmp_path):
    """A fresh enabled run uses --lr as the schedule base, so it is checked too."""
    csv, names = _write_images(tmp_path)
    fresh = _request(
        tmp_path,
        csv,
        relational_queue_size=2,
        relational_queue_size_explicit=True,
    )
    with pytest.raises(ValueError, match=r"--lr to survive the float32"):
        recipe.prepare_pretrain_relational_queue(
            argparse.Namespace(**{**vars(fresh), "lr": 1e100}), None
        )
    with pytest.raises(ValueError, match="--lr to be non-negative"):
        recipe.prepare_pretrain_relational_queue(
            argparse.Namespace(**{**vars(fresh), "lr": -1.0}), None
        )
    # Zero stays legal, and a disabled run is not asked about its LR at all.
    assert (
        recipe.prepare_pretrain_relational_queue(
            argparse.Namespace(**{**vars(fresh), "lr": 0.0}), None
        ).active
    )
    # A reused preparation re-checks this invocation's LR.
    preparation = recipe.prepare_pretrain_relational_queue(
        argparse.Namespace(**{**vars(fresh), "lr": 0.0}), None
    )
    with pytest.raises(ValueError, match=r"--lr to survive the float32"):
        recipe.verify_relational_queue_preparation(
            preparation,
            argparse.Namespace(**{**vars(fresh), "lr": 1e100}),
            None,
        )


def test_fifo_center_validation_uses_the_restore_cast():
    """Validation must describe the cast the restore really performs."""
    payload = torch.zeros(1, 8, dtype=torch.float64) + 0.5
    recipe._validate_fifo_center(payload, 8)  # finite float64 is restorable
    with pytest.raises(ValueError, match="not finite at float32"):
        recipe._validate_fifo_center(
            torch.full((1, 8), 1e100, dtype=torch.float64), 8
        )
    with pytest.raises(ValueError, match=r"exact shape \[1, 8\]"):
        recipe._validate_fifo_center(torch.zeros(8, dtype=torch.float64), 8)


def _optimizer_group(**flags):
    return {
        "param_groups": [
            {
                "params": [],
                "lr": 0.001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
                **flags,
            }
        ],
        "state": {},
    }


def _restore_and_step(flags: dict, device: torch.device) -> str:
    """The trainer's real path: default construct, load, backward, step."""
    parameter = torch.nn.Parameter(torch.ones(4, device=device))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    saved = optimizer.state_dict()
    saved["param_groups"][0].update(flags)
    for index in range(len(saved["param_groups"][0]["params"])):
        saved["state"][index] = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.zeros_like(parameter, device="cpu"),
            "exp_avg_sq": torch.zeros_like(parameter, device="cpu"),
        }
    try:
        optimizer.load_state_dict(saved)
        parameter.grad = torch.ones(4, device=device)
        optimizer.step()
    except Exception as exc:  # noqa: BLE001 - the point is that it fails
        return f"{type(exc).__name__}"
    return "ok" if bool(torch.isfinite(parameter).all()) else "nonfinite"


@pytest.mark.parametrize(
    "coefficients, match",
    [
        (
            {"weight_decay": 1e100},
            "optimizer update is unusable",
        ),
        ({"eps": 1e-100}, "optimizer update is unusable"),
        ({"eps": 0.0}, "finite AdamW eps"),
        ({"betas": [1 - 1e-12, 0.999], "differentiable": True}, "optimizer update is unusable"),
        ({"betas": [0.9, 1.0]}, "AdamW betas"),
        ({"betas": [0.9, -0.1]}, "AdamW betas"),
        ({"weight_decay": -0.1}, "finite AdamW weight_decay"),
        ({"eps": float("nan")}, "finite AdamW eps"),
    ],
)
def test_fifo_optimizer_coefficients_follow_the_execution_branch(coefficients, match):
    group = {
        "params": [],
        "lr": 0.001,
        "eps": 1e-8,
        "betas": [0.9, 0.999],
        "weight_decay": 0.0,
        **coefficients,
    }
    with pytest.raises(ValueError, match=match):
        recipe._validate_queue_optimizer_coefficients(
            group, group_index=0, lr_bound=0.001
        )


def _restored_update(flags: dict, coefficients: dict, steps: int = 3) -> bool:
    """Restore a saved state into a default AdamW and run real steps."""
    parameter = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.AdamW([parameter], lr=0.001)
    saved = optimizer.state_dict()
    saved["param_groups"][0].update(
        {
            "eps": 1e-8,
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
            **flags,
            **coefficients,
        }
    )
    saved["state"][0] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.zeros_like(parameter),
        "exp_avg_sq": torch.zeros_like(parameter),
    }
    optimizer.load_state_dict(saved)
    for _ in range(steps):
        parameter.grad = torch.ones(4)
        optimizer.step()
    return bool(torch.isfinite(parameter).all())


@pytest.mark.parametrize(
    "flags, coefficients",
    [
        # Decoupled decay: lr * weight_decay overflows the update precision.
        ({}, {"weight_decay": 1e100}),
        # The differentiable branch computes in float32, so a beta that rounds
        # to one zeroes the bias correction.
        ({"differentiable": True}, {"betas": (1 - 1e-12, 0.999)}),
    ],
)
def test_fifo_optimizer_coefficients_match_the_real_update(flags, coefficients):
    """The guarded values really do leave a non-finite model."""
    base = {"eps": 1e-8, "betas": (0.9, 0.999), "weight_decay": 0.0}
    assert _restored_update(flags, base)
    assert not _restored_update(flags, coefficients)
    with pytest.raises(ValueError):
        recipe._validate_queue_optimizer_coefficients(
            {"params": [], "lr": 0.001, **base, **flags, **coefficients},
            group_index=0,
            lr_bound=0.001,
        )


def test_fifo_optimizer_eps_underflow_fails_the_zero_gradient_probe():
    """A float32-zero eps fails with zero moments and zero gradients."""
    base = {"eps": 1e-100, "betas": (0.9, 0.999), "weight_decay": 0.0}
    with pytest.raises(ValueError, match="optimizer update is unusable"):
        recipe._validate_queue_optimizer_coefficients(
            {"params": [], "lr": 0.001, **base}, group_index=0, lr_bound=0.001
        )


def test_fifo_amsgrad_is_judged_per_group():
    """One AMSGrad group must not impose max_exp_avg_sq on the others."""
    first = torch.nn.Parameter(torch.ones(2))
    second = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW(
        [{"params": [first], "amsgrad": True}, {"params": [second], "amsgrad": False}],
        lr=0.001,
    )
    saved = optimizer.state_dict()
    saved["state"] = {
        0: {
            "step": torch.tensor(1.0),
            "exp_avg": torch.zeros(2),
            "exp_avg_sq": torch.zeros(2),
            "max_exp_avg_sq": torch.zeros(2),
        },
        1: {
            "step": torch.tensor(1.0),
            "exp_avg": torch.zeros(2),
            "exp_avg_sq": torch.zeros(2),
        },
    }
    # Accepted: only the AMSGrad group needs the maximum second moment.
    recipe._validate_fifo_adamw_state(
        optimizer, saved, completed_steps=2, label="pretrain resume"
    )
    # And the same state really does restore and step.
    for parameter in (first, second):
        parameter.grad = torch.ones(2)
    optimizer.load_state_dict(saved)
    for parameter in (first, second):
        parameter.grad = torch.ones(2)
    optimizer.step()
    assert torch.isfinite(first).all() and torch.isfinite(second).all()

    # The AMSGrad group itself must still carry it.
    broken = {
        "param_groups": saved["param_groups"],
        "state": {0: dict(saved["state"][0]), 1: saved["state"][1]},
    }
    broken["state"][0].pop("max_exp_avg_sq")
    with pytest.raises(ValueError, match="max_exp_avg_sq"):
        recipe._validate_fifo_adamw_state(
            optimizer, broken, completed_steps=2, label="pretrain resume"
        )


def test_fifo_optimizer_decay_product_overflow():
    """Each factor can be representable while their product is not."""
    with pytest.raises(ValueError, match="optimizer update is unusable"):
        recipe._validate_queue_optimizer_coefficients(
            {"params": [], "lr": 1e10, "eps": 1e-8, "betas": [0.9, 0.999],
             "weight_decay": 1e30},
            group_index=0,
            lr_bound=1e10,
        )
    # The same weight decay is fine at a modest learning rate.
    recipe._validate_queue_optimizer_coefficients(
        {"params": [], "lr": 0.001, "eps": 1e-8, "betas": [0.9, 0.999],
         "weight_decay": 1e30},
        group_index=0,
        lr_bound=0.001,
    )


@pytest.mark.parametrize(
    "step_spec, flags, match",
    [
        # A tensor step the foreach path rejects (its dtype check).
        (("int64", 0), {"foreach": True}, "rejects"),
        (("float16", 0), {"foreach": True}, "rejects"),
        # Differentiable needs a scalar; a shaped step is CPU-only and 2-D never.
        (("float32", 2), {"differentiable": True}, "shaped step only on CPU"),
        # Only the differentiable branch rejects requires_grad on its counter.
        (("float32", 0, True), {"differentiable": True}, "requires_grad"),
    ],
)
def test_fifo_optimizer_step_format_matches_the_branch(step_spec, flags, match):
    name, dims = step_spec[0], step_spec[1]
    requires_grad = len(step_spec) > 2 and step_spec[2]
    dtype = getattr(torch, name)
    shape = () if dims == 0 else (1,) * dims
    step = torch.ones(shape, dtype=dtype)
    step.requires_grad_(requires_grad)
    with pytest.raises(ValueError, match=match):
        recipe._validate_fifo_step_format(
            step, flags={**{
                "capturable": False,
                "fused": False,
                "foreach": False,
                "differentiable": False,
            }, **flags}, target_device=torch.device("cpu"),
            label="optimizer step for parameter 0",
        )


@pytest.mark.parametrize(
    "flags, runs",
    [
        ({}, True),
        ({"differentiable": True}, True),
        ({"foreach": True}, True),
        ({"fused": True}, True),
        ({"fused": True, "foreach": True}, True),
        ({"fused": True, "capturable": True}, True),
        ({"fused": True, "capturable": True, "foreach": True}, True),
        ({"foreach": True, "differentiable": True}, False),
        ({"fused": True, "differentiable": True}, False),
        ({"capturable": True}, False),
        ({"capturable": True, "foreach": True}, False),
        ({"capturable": True, "differentiable": True}, False),
    ],
)
def test_fifo_optimizer_flags_match_the_restored_optimizer(flags, runs):
    """The rules must agree with construct -> load -> step on the real device."""
    for device in (torch.device("cpu"), torch.device("mps")):
        if device.type == "mps" and not torch.backends.mps.is_available():
            continue
        observed = _restore_and_step(flags, device)
        if runs:
            assert observed == "ok", (device, flags, observed)
            recipe._validate_fifo_optimizer_execution_flags(
                _optimizer_group(**flags), target_device=device
            )
        else:
            assert observed != "ok", (device, flags, observed)
            with pytest.raises(ValueError):
                recipe._validate_fifo_optimizer_execution_flags(
                    _optimizer_group(**flags), target_device=device
                )

    # A CUDA target accepts every single flag as a *rule*: no CUDA hardware was
    # exercised here, so this is not evidence that those steps run.
    for name in ("capturable", "fused", "foreach", "differentiable"):
        recipe._validate_fifo_optimizer_execution_flags(
            _optimizer_group(**{name: True}), target_device=torch.device("cuda")
        )
    # The device-independent conflict stays rejected for CUDA as well.
    with pytest.raises(ValueError, match="differentiable=True"):
        recipe._validate_fifo_optimizer_execution_flags(
            _optimizer_group(foreach=True, differentiable=True),
            target_device=torch.device("cuda"),
        )


def test_fifo_reuse_is_bound_to_the_epoch_target(tmp_path):
    csv, names = _write_images(tmp_path, count=14)
    resume = "runs/x/SSL_epoch_0001.pth"
    payload = _enabled_payload(
        tmp_path,
        names,
        capacity=2,
        batch=2,
        keys=names[:2],
        observed=[6, 6],
        schedule_overrides={
            "steps_per_epoch": 7,
            "total_steps": 7,
            "original_max_epochs": 3,
            "warmup_steps": 21,
            "last_lr": 0.001,
            "final_lr": 1e-05,
        },
    )
    payload["epoch"] = 0
    payload["iteration"] = 7
    payload["args"] = {**payload["args"], "batch_size": 2, "max_epochs": 3}
    extension = _request(
        tmp_path,
        csv,
        resume=resume,
        batch_size=2,
        batch_size_explicit=True,
        max_epochs=4,
    )
    preparation = recipe.prepare_pretrain_relational_queue(extension, payload)
    same_plan = _request(
        tmp_path,
        csv,
        resume=resume,
        batch_size=2,
        batch_size_explicit=True,
        max_epochs=3,
    )
    with pytest.raises(ValueError, match="different request"):
        recipe.verify_relational_queue_preparation(
            preparation, same_plan, payload
        )
    with pytest.raises(ValueError, match="inconsistent saved schedule"):
        recipe._validate_queue_schedule_coverage(
            payload, requested_max_epochs=3
        )


def test_fifo_resume_rejects_conflicting_capacity_representations(tmp_path):
    csv, names = _write_images(tmp_path)
    resume = "runs/x/SSL_epoch_0001.pth"
    request = _request(
        tmp_path, csv, resume=resume, batch_size=2, batch_size_explicit=True
    )
    payload = _enabled_payload(tmp_path, names, capacity=2, batch=2)
    recipe.prepare_pretrain_relational_queue(request, payload)

    for mutate, match in (
        (
            lambda p: p["args"].update({"relational_queue_size": 4}),
            "Queue config/args conflict",
        ),
        (
            lambda p: p["args"].update({"relational_queue_size": 2.0}),
            "non-integer",
        ),
        (
            lambda p: p["args"].update({"relational_queue_size": "2"}),
            "non-integer",
        ),
        (
            lambda p: p["args"].update({"requested_relational_queue_size": 4}),
            "Queue request/config conflict",
        ),
        (
            lambda p: p["args"].update({"requested_relational_queue_size": 2.5}),
            "malformed",
        ),
        (
            lambda p: p["args"].update({"requested_relational_queue_size": "2"}),
            "malformed",
        ),
        (
            lambda p: p["args"].update({"requested_relational_queue_size": -1}),
            "negative",
        ),
    ):
        broken = _enabled_payload(tmp_path, names, capacity=2, batch=2)
        mutate(broken)
        with pytest.raises(ValueError, match=match):
            recipe.prepare_pretrain_relational_queue(request, broken)

    # An omitted request marker stays valid when the bundle is complete.
    omitted_request = _enabled_payload(tmp_path, names, capacity=2, batch=2)
    omitted_request["args"]["requested_relational_queue_size"] = None
    assert (
        recipe.prepare_pretrain_relational_queue(request, omitted_request).capacity
        == 2
    )


def test_fifo_resume_locks_exact_batch_not_only_loader_length(tmp_path):
    csv, names = _write_images(tmp_path, count=100)
    # 100 images: 100 // 30 == 100 // 33 == 3, so an equal loader length must not
    # make a changed batch size valid.
    payload = _enabled_payload(
        tmp_path, names, capacity=4, batch=30, keys=names[:4], observed=[2, 2, 2, 2]
    )
    resume = "runs/x/SSL_epoch_0001.pth"
    with pytest.raises(ValueError, match="Cannot change --batch-size"):
        recipe.prepare_pretrain_relational_queue(
            _request(tmp_path, csv, resume=resume, batch_size=33, batch_size_explicit=True),
            payload,
        )


def test_fifo_resume_rejects_impossible_latest_commit_tail(tmp_path):
    csv, names = _write_images(tmp_path, count=8)
    resume = "runs/x/SSL_epoch_0001.pth"
    # K=4, B=4 at epoch 9 (iteration 10): the newest four records must be
    # stamped 9, and no older record may share that stamp.
    payload = _enabled_payload(
        tmp_path,
        names,
        capacity=4,
        batch=4,
        epoch=9,
        keys=names[:4],
    )
    payload["args"] = {**payload["args"], "max_epochs": 10}
    recipe.prepare_pretrain_relational_queue(
        _request(
            tmp_path,
            csv,
            resume=resume,
            batch_size=4,
            batch_size_explicit=True,
            max_epochs=10,
        ),
        payload,
    )
    stale_tail = _enabled_payload(
        tmp_path,
        names,
        capacity=4,
        batch=4,
        epoch=9,
        keys=names[:4],
        # 8 images at batch 4: iteration 20, so the stale tail is stamped 18.
        observed=[18, 19, 19, 19],
    )
    stale_tail["args"] = {**stale_tail["args"], "max_epochs": 10}
    with pytest.raises(ValueError, match="latest successful batch"):
        recipe.prepare_pretrain_relational_queue(
            _request(
                tmp_path,
                csv,
                resume=resume,
                batch_size=4,
                batch_size_explicit=True,
                max_epochs=10,
            ),
            stale_tail,
        )


def test_fifo_epoch_resume_rejects_truncated_history_with_valid_tail(tmp_path):
    csv, names = _write_images(tmp_path, count=100)
    resume = "runs/x/SSL_epoch_0001.pth"
    # N=100, B=10, K=64: the last ten keys at step 9 satisfy the tail rule, but
    # a complete epoch consumed 100 distinct keys, so occupancy must reach 64.
    truncated = _enabled_payload(
        tmp_path,
        names,
        capacity=64,
        batch=10,
        epoch=9,
        keys=names[:10],
        observed=[99] * 10,
    )
    truncated["args"] = {**truncated["args"], "max_epochs": 10}
    with pytest.raises(ValueError, match="truncated history"):
        recipe.prepare_pretrain_relational_queue(
            _request(
                tmp_path,
                csv,
                resume=resume,
                batch_size=10,
                batch_size_explicit=True,
                max_epochs=10,
            ),
            truncated,
        )
    complete = _enabled_payload(
        tmp_path,
        names,
        capacity=64,
        batch=10,
        epoch=9,
        keys=names[:64],
        observed=[98] * 54 + [99] * 10,
    )
    complete["args"] = {**complete["args"], "max_epochs": 10}
    preparation = recipe.prepare_pretrain_relational_queue(
        _request(
            tmp_path,
            csv,
            resume=resume,
            batch_size=10,
            batch_size_explicit=True,
            max_epochs=10,
        ),
        complete,
    )
    assert preparation.capacity == 64


def test_fifo_temperature_admission_uses_expanded_candidates(tmp_path, monkeypatch):
    csv, names = _write_images(tmp_path, count=8)
    preparation = recipe.prepare_pretrain_relational_queue(
        _request(
            tmp_path,
            csv,
            relational_queue_size=4,
            relational_queue_size_explicit=True,
        ),
        None,
    )
    assert preparation.candidate_bound == 2 + min(4, 8 - 2) == 6

    import numpy as np

    limit_value = float(np.finfo(np.float32).max) * (
        1.0 - 4.0 * float(np.finfo(np.float32).eps)
    )
    # Admitted at batch 4, rejected once the candidates include the history:
    # bound = length / tau, and only the shorter reduction stays finite.
    tiny = 5.0 / limit_value
    with pytest.raises(ValueError, match="batch size 6"):
        recipe._admissible_temperature(
            tiny, preparation.candidate_bound, "teacher_temp_start"
        )
    assert recipe._admissible_temperature(tiny, 4, "teacher_temp_start") == tiny
    assert recipe._admissible_temperature(0.04, preparation.candidate_bound, "x") == 0.04

    # The resolved target recipe admits teacher endpoints against that same
    # expanded bound instead of the parser batch size.
    captured: list[tuple[str, int]] = []
    real = recipe._admissible_temperature

    def spy(value, batch_size, option):
        captured.append((option, int(batch_size)))
        return real(value, batch_size, option)

    monkeypatch.setattr(recipe, "_admissible_temperature", spy)
    args = _request(
        tmp_path,
        csv,
        relational_queue_size=4,
        relational_queue_size_explicit=True,
    )
    recipe._resolve_relational_target_config(
        args, None, candidate_count=preparation.candidate_bound
    )
    assert ("--teacher-temp-start", 6) in captured
    assert ("--teacher-temp-end", 6) in captured

    # Student temperature admission uses the same length in the SSL resolver.
    captured.clear()
    recipe.resolve_pretrain_ssl_config(
        _request(
            tmp_path,
            csv,
            relational_queue_size=4,
            relational_queue_size_explicit=True,
            student_temp=0.1,
        ),
        None,
        relational_queue_preparation=preparation,
    )
    assert ("--student-temp", 6) in captured


def test_fifo_rng_validation_does_not_change_caller_rng():
    import random

    import numpy as np

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    torch.manual_seed(3)
    before_torch = torch.get_rng_state().clone()
    before_python = random.getstate()
    validated = recipe._validate_fifo_rng_state(state, target_device=torch.device("cpu"))
    assert set(validated) == {"python", "numpy", "torch"}
    assert torch.equal(before_torch, torch.get_rng_state())
    assert before_python == random.getstate()

    with pytest.raises(ValueError, match="requires the saved rng_state"):
        recipe._validate_fifo_rng_state(None, target_device=torch.device("cpu"))
    with pytest.raises(ValueError, match=r"rng_state\.numpy"):
        recipe._validate_fifo_rng_state(
            {"python": state["python"], "numpy": None, "torch": state["torch"]},
            target_device=torch.device("cpu"),
        )


def test_fifo_rng_target_matrix(monkeypatch):
    import random

    import numpy as np

    base = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    malformed = {**base, "cuda": [torch.zeros(4, dtype=torch.float32)]}
    with pytest.raises(ValueError, match="malformed rng_state.cuda"):
        recipe._validate_fifo_rng_state(malformed, target_device=torch.device("cpu"))

    empty = {**base, "cuda": []}
    with pytest.raises(ValueError, match="empty rng_state.cuda"):
        recipe._validate_fifo_rng_state(empty, target_device=torch.device("cpu"))

    # A CPU-only checkpoint cannot continue on CUDA, and no substitute stream is
    # seeded for it.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="records no CUDA RNG"):
        recipe._validate_fifo_rng_state(base, target_device=torch.device("cuda"))

    mismatched = {**base, "cuda": [torch.zeros(4, dtype=torch.uint8)]}
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="this host exposes 2"):
        recipe._validate_fifo_rng_state(mismatched, target_device=torch.device("cuda"))

    # A CUDA-shaped payload recovered on a non-CUDA target is structurally
    # checked, dropped from the restore copy and reported.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    valid_cuda = {**base, "cuda": [torch.zeros(4, dtype=torch.uint8)]}
    restored_states: list[dict] = []

    def spy_restore(state):
        restored_states.append(state)

    monkeypatch.setattr(recipe, "_restore_rng_state", spy_restore)
    restored = recipe._validate_fifo_rng_state(
        valid_cuda, target_device=torch.device("cpu")
    )
    assert "cuda" not in restored
    assert "cuda" not in restored_states[-1]
    recipe._validate_fifo_rng_state(valid_cuda, target_device=torch.device("cuda"))
    assert "cuda" in restored_states[-1]


def test_fifo_preparation_rejects_before_any_probe(tmp_path, monkeypatch):
    csv, names = _write_images(tmp_path)
    calls = []

    for module in (
        recipe,
        __import__("otuformer.training.pretrain_loop", fromlist=["x"]),
        __import__("otuformer.training.augmentation", fromlist=["x"]),
    ):
        original = getattr(module, "OTUFormerEncoder", None)
        if original is None:
            continue

        def spy(*args, __original=original, **kwargs):
            calls.append(kwargs.get("out_dim"))
            return __original(*args, **kwargs)

        monkeypatch.setattr(module, "OTUFormerEncoder", spy)

    bad = _request(
        tmp_path,
        csv,
        ssl_method="dino",
        relational_queue_size=0,
        relational_queue_size_explicit=True,
    )
    with pytest.raises(ValueError, match="relational"):
        recipe.prepare_pretrain_relational_queue(bad, None)
    assert calls == []


def test_fifo_handoff_is_consumed_once_and_retry_is_clean(tmp_path):
    csv, _names = _write_images(tmp_path)
    preparation = recipe.prepare_pretrain_relational_queue(
        _request(tmp_path, csv, relational_queue_size=2,
                 relational_queue_size_explicit=True),
        None,
    )
    args = argparse.Namespace(
        relational_queue_preparation=preparation,
        ssl_config_effective=None,
        resume_state=None,
        resume_state_path=None,
    )
    handoff = recipe.take_pretrain_handoff(args)
    assert handoff.relational_queue_preparation is preparation
    assert not hasattr(args, "relational_queue_preparation")
    retry = recipe.take_pretrain_handoff(args)
    assert retry.relational_queue_preparation is None

    # A request change invalidates the reused preparation instead of silently
    # serving the old data.
    changed = _request(
        tmp_path,
        csv,
        relational_queue_size=3,
        relational_queue_size_explicit=True,
    )
    with pytest.raises(ValueError, match="different request"):
        recipe.verify_relational_queue_preparation(preparation, changed, None)


def test_fifo_enabled_recipe_requires_preparation(tmp_path):
    csv, _names = _write_images(tmp_path)
    settings = recipe.resolve_pretrain_settings(
        _request(tmp_path, csv, relational_queue_size=2,
                 relational_queue_size_explicit=True),
        None,
    )
    with pytest.raises(ValueError, match="no prepared selection"):
        recipe._resolve_pretrain_recipe(
            _request(tmp_path, csv, relational_queue_size=2,
                     relational_queue_size_explicit=True),
            None,
            settings=settings,
        )
