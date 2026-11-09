"""DINO training-loop integration: pairing, forwards, freeze, EMA and KoLeo."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from otuformer.training import trainer
from otuformer.training.model import OTUFormerEncoder
import otuformer.training.dino as dino_module
import otuformer.training.loss as loss_module
import otuformer.training.objectives as objectives_module
import otuformer.training.pretrain_recipe as pretrain_recipe_module
import otuformer.training.reporting as reporting_module
import otuformer.training.runtime as runtime_module
import otuformer.training.pretrain_loop as pretrain_loop_module

MODEL = "vit_tiny_patch16_224"
SIZE = 224


def _images(tmp_path: Path, rows: int = 2, duplicates: bool = False) -> tuple[Path, Path]:
    images = tmp_path / "images"
    images.mkdir(parents=True, exist_ok=True)
    for index in range(1 if duplicates else rows):
        Image.new("RGB", (SIZE, SIZE), (index * 40, 10, 90)).save(
            images / f"{index}.png"
        )
    manifest = tmp_path / "data.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image"])
        for index in range(rows):
            writer.writerow([f"{0 if duplicates else index}.png"])
    return images, manifest


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    base = dict(
        train_data=str(tmp_path / "data.csv"),
        input_images_dir=str(tmp_path / "images"),
        out_dir=str(tmp_path / "out"),
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
        koleo_weight=0.0,
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
        resume="",
        log_every_n_steps=1,
        save_every_epochs=100,
        keep_last_checkpoints=1,
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
        device="auto",
        seed=42,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture()
def tiny_run(tmp_path):
    _images(tmp_path)


def _dino_objective_built(monkeypatch, captured: list) -> None:
    """Capture every objective the run builds.

    The trainer constructs it from its own namespace, so the patch has to land
    where the call happens, not on the class definition module.
    """
    real = pretrain_loop_module.DINOObjective

    def spy(*args, **kwargs):
        objective = real(*args, **kwargs)
        captured.append(objective)
        return objective

    monkeypatch.setattr(pretrain_loop_module, "DINOObjective", spy)


def _capture_teacher(monkeypatch, teachers: list) -> None:
    real_update = pretrain_loop_module.update_teacher

    def update_spy(student, teacher, momentum):
        if hasattr(teacher, "backbone"):
            teachers.append(teacher)
        return real_update(student, teacher, momentum)

    monkeypatch.setattr(pretrain_loop_module, "update_teacher", update_spy)


def test_dino_global_pairs_skip_same_crop_and_zero_locals():
    torch.manual_seed(0)
    logits = [torch.randn(3, 5), torch.randn(3, 5)]
    targets = [
        F.one_hot(torch.tensor([0, 0, 0]), 5).float(),
        F.one_hot(torch.tensor([4, 4, 4]), 5).float(),
    ]

    loss = pretrain_loop_module._compute_dino_global_loss(logits, targets, 0.1)
    cross = (
        objectives_module.dino_distillation_loss(logits[0], targets[1], 0.1)
        + objectives_module.dino_distillation_loss(logits[1], targets[0], 0.1)
    ) / 2
    same_crop = (
        objectives_module.dino_distillation_loss(logits[0], targets[0], 0.1)
        + objectives_module.dino_distillation_loss(logits[1], targets[1], 0.1)
    ) / 2
    torch.testing.assert_close(loss, cross)
    assert not torch.allclose(loss, same_crop)

    zero_local = pretrain_loop_module._compute_dino_local_loss([], targets, 0.1)
    assert float(zero_local) == 0.0
    assert zero_local.ndim == 0


@pytest.mark.parametrize(
    ("patch_mode", "masked_forwards"),
    [("none", 0), ("consistency", 0), ("masked-feature", 2), ("ibot", 2)],
)
def test_dino_reuses_unmasked_cls_and_backbone_tokens(
    tmp_path, monkeypatch, patch_mode, masked_forwards
):
    _images(tmp_path)
    events = {"unmasked": [], "masked": [], "koleo_inputs": []}
    real_forward = OTUFormerEncoder.forward_pretrain
    real_koleo = objectives_module.koleo_loss

    def forward_spy(self, *args, **kwargs):
        out = real_forward(self, *args, **kwargs)
        bucket = "masked" if kwargs.get("mask") is not None else "unmasked"
        events[bucket].append(out.tokens.detach().clone())
        return out

    def koleo_spy(features, **kwargs):
        events["koleo_inputs"].append(features.detach().clone())
        return real_koleo(features, **kwargs)

    monkeypatch.setattr(OTUFormerEncoder, "forward_pretrain", forward_spy)
    monkeypatch.setattr(objectives_module, "koleo_loss", koleo_spy)

    pretrain_loop_module.run_pretrain(
        _args(
            tmp_path,
            patch_loss=patch_mode,
            koleo_weight=0.1,
            koleo_weight_explicit=True,
            max_epochs=1,
        )
    )

    # Two global crops per encoder, never a third pass for CLS or KoLeo.
    assert len(events["unmasked"]) == 4  # student + teacher, two globals
    assert len(events["masked"]) == masked_forwards
    assert len(events["koleo_inputs"]) == 2
    for index, captured in enumerate(events["koleo_inputs"]):
        # KoLeo consumes the unmasked student backbone CLS token exactly.
        unmasked = events["unmasked"][index][:, 0]
        torch.testing.assert_close(captured, unmasked)
        if events["masked"]:
            assert not torch.allclose(captured, events["masked"][index][:, 0])


def test_dino_epoch_zero_freeze_then_joint_clip(tmp_path, monkeypatch):
    _images(tmp_path)
    captured: list = []
    _dino_objective_built(monkeypatch, captured)
    snapshots = {"epoch0_direction_unchanged": None, "epoch1_direction_updated": None}
    observed = {
        "prototype_in_joint_norm_and_clip": [],
        "clip_calls": [],
        "direction": [],
    }
    real_norm = pretrain_loop_module._compute_grad_norm_parameters
    real_clip = torch.nn.utils.clip_grad_norm_

    def norm_spy(params):
        direction = captured[0].trainable_parameters()[0]
        observed["prototype_in_joint_norm_and_clip"].append(
            any(parameter is direction for parameter in params)
            and direction.grad is not None
        )
        observed["direction"].append(direction.detach().clone())
        return real_norm(params)

    def clip_spy(params, max_norm, **kwargs):
        direction = captured[0].trainable_parameters()[0]
        observed["clip_calls"].append(direction.grad is None)
        return real_clip(params, max_norm, **kwargs)

    monkeypatch.setattr(pretrain_loop_module, "_compute_grad_norm_parameters", norm_spy)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip_spy)

    values = []
    real_clip_wrapper = None

    real_step = torch.optim.AdamW.step
    steps = {"count": 0}
    after_step = []

    def step_spy(self, *args, **kwargs):
        result = real_step(self, *args, **kwargs)
        steps["count"] += 1
        after_step.append(
            captured[0].student_prototypes.weight_v.detach().clone()
        )
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", step_spy)
    pretrain_loop_module.run_pretrain(_args(tmp_path, max_epochs=2))
    snapshots["epoch0_direction_unchanged"] = torch.equal(
        observed["direction"][0], after_step[0]
    )
    snapshots["epoch1_direction_updated"] = not torch.equal(
        after_step[-1], after_step[0]
    )

    # Absolute epoch 0 cleared the gradient; epoch 1 trains the direction again.
    assert observed["clip_calls"] == [True, False]
    assert observed["prototype_in_joint_norm_and_clip"] == [False, True]
    assert snapshots["epoch0_direction_unchanged"] is True
    assert snapshots["epoch1_direction_updated"] is True
    assert real_clip_wrapper is None


def test_dino_sk_ibot_centers_are_independent(tmp_path, monkeypatch):
    _images(tmp_path)
    captured: list = []
    teachers: list = []
    _dino_objective_built(monkeypatch, captured)
    _capture_teacher(monkeypatch, teachers)
    events = {"legacy_center_updates": 0, "patch_center_updates": 0, "steps": 0}
    real_center = pretrain_loop_module._update_teacher_center
    real_patch_center = pretrain_loop_module._update_patch_center
    real_step = torch.optim.AdamW.step

    def center_spy(center, teacher_global):
        events["legacy_center_updates"] += 1
        return real_center(center, teacher_global)

    def patch_center_spy(center, logits):
        events["patch_center_updates"] += 1
        return real_patch_center(center, logits)

    def step_spy(self, *args, **kwargs):
        events["steps"] += 1
        return real_step(self, *args, **kwargs)

    monkeypatch.setattr(pretrain_loop_module, "_update_teacher_center", center_spy)
    monkeypatch.setattr(pretrain_loop_module, "_update_patch_center", patch_center_spy)
    monkeypatch.setattr(torch.optim.AdamW, "step", step_spy)

    pretrain_loop_module.run_pretrain(
        _args(
            tmp_path,
            dino_centering="sinkhorn-knopp",
            dino_centering_explicit=True,
            patch_loss="ibot",
            koleo_weight=0.1,
            koleo_weight_explicit=True,
            max_epochs=2,
        )
    )

    assert events["legacy_center_updates"] == 0
    assert events["patch_center_updates"] == events["steps"]
    assert torch.count_nonzero(teachers[-1].center) == 0
    assert captured[-1].cls_center is None
    assert "cls_center" not in captured[-1].state_dict()


def test_dino_koleo_sums_crops_without_work_when_disabled(tmp_path, monkeypatch):
    _images(tmp_path)
    calls = []
    real_koleo = objectives_module.koleo_loss

    def koleo_spy(features, **kwargs):
        calls.append(tuple(features.shape))
        return real_koleo(features, **kwargs)

    monkeypatch.setattr(objectives_module, "koleo_loss", koleo_spy)
    pretrain_loop_module.run_pretrain(_args(tmp_path, koleo_weight=0.0))
    assert calls == []

    calls.clear()
    pretrain_loop_module.run_pretrain(
        _args(tmp_path, koleo_weight=0.1, out_dir=str(tmp_path / "out2"))
    )
    # One call per global crop, each over exactly the batch pool.
    assert calls == [(2, 192), (2, 192)]


def test_dino_koleo_duplicate_records_keep_per_crop_pool(tmp_path, monkeypatch):
    _images(tmp_path, rows=2, duplicates=True)
    calls = []
    real_koleo = objectives_module.koleo_loss

    def koleo_spy(features, **kwargs):
        calls.append(tuple(features.shape))
        return real_koleo(features, **kwargs)

    monkeypatch.setattr(objectives_module, "koleo_loss", koleo_spy)
    pretrain_loop_module.run_pretrain(_args(tmp_path, koleo_weight=0.1))

    assert calls == [(2, 192), (2, 192)]


def test_dino_construction_uses_supplied_effective_dim(tmp_path, monkeypatch):
    _images(tmp_path)
    dims = []
    real_init = OTUFormerEncoder.__init__

    def init_spy(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        dims.append(int(self.projector.net[-1].out_features))

    monkeypatch.setattr(OTUFormerEncoder, "__init__", init_spy)
    args = _args(tmp_path, out_dim=256)
    args.ssl_config_effective = {
        "ssl_method": "dino",
        "ssl_head": "dino_mlp_2048",
        "out_dim": 17,
        "dino_prototypes": 11,
        "dino_centering": "ema",
        "koleo_weight": 0.0,
        "teacher_temp_warmup_ratio": 0.7,
        "norm_bias_weight_decay": True,
        "dino_hidden_dim": 2048,
        "dino_mlp_init_std": 0.02,
        "dino_mlp_init_bias": 0.0,
        "dino_weight_norm_api": "explicit",
        "dino_prototype_scale": 1.0,
        "dino_freeze_last_layer_epochs": 1,
        "dino_target_dtype": "float32",
        "dino_center_momentum": 0.9,
        "koleo_normalize_eps": 1e-8,
        "koleo_distance_eps": 1e-8,
        "koleo_log_eps": 1e-8,
    }

    pretrain_loop_module.run_pretrain(args)

    # The first encoder is the preflight probe at the requested width; the
    # student and teacher then use the resolved effective DINO width.
    assert dims == [256, 17, 17]
    assert args.out_dim == 256


def test_dino_initializes_only_projector_before_teacher_sync(tmp_path, monkeypatch):
    _images(tmp_path)
    observed = {}
    real_init = OTUFormerEncoder.__init__
    real_trunc = torch.nn.init.trunc_normal_
    initialised = []
    real_sync = pretrain_loop_module._synchronize_dino_teacher

    def init_spy(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        self.__dict__["_backbone_snapshot"] = {
            name: value.detach().clone()
            for name, value in self.backbone.named_parameters()
        }

    def trunc_spy(tensor, *args, **kwargs):
        initialised.append(tuple(tensor.shape))
        return real_trunc(tensor, *args, **kwargs)

    def sync_spy(student, teacher, dino_objective):
        result = real_sync(student, teacher, dino_objective)
        observed["projector_matches_teacher"] = all(
            torch.equal(a, b)
            for a, b in zip(
                student.projector.state_dict().values(),
                teacher.projector.state_dict().values(),
            )
        )
        observed["objective_synced"] = torch.equal(
            dino_objective.student_prototypes.weight_v,
            dino_objective.teacher_prototypes.weight_v,
        )
        observed["backbone_untouched"] = all(
            torch.equal(value, student.backbone.state_dict()[name])
            for name, value in student.__dict__["_backbone_snapshot"].items()
        )
        return result

    monkeypatch.setattr(OTUFormerEncoder, "__init__", init_spy)
    monkeypatch.setattr(torch.nn.init, "trunc_normal_", trunc_spy)
    monkeypatch.setattr(pretrain_loop_module, "_synchronize_dino_teacher", sync_spy)

    # CPU: this compares parameter snapshots taken at construction with the
    # moved model, which the device policy keeps on CPU.
    pretrain_loop_module.run_pretrain(_args(tmp_path, device="cpu"))

    # Only the projector's linear weights are re-initialized (3 layers), and the
    # teacher copy happens after it.
    assert len(initialised) == 3
    assert observed["projector_matches_teacher"] is True
    assert observed["objective_synced"] is True
    assert observed["backbone_untouched"] is True


def test_dino_targets_and_ema_update_once_after_step(tmp_path, monkeypatch):
    """One ordered update per step: step, encoder EMA, prototype EMA, center.

    The two EMA calls share one function, so they are told apart by the objects
    they receive; a single label would let a removed or reordered prototype EMA
    pass. The whole sequence is compared, not just the first position.
    """
    _images(tmp_path)
    captured: list = []
    # Capture the real class members before ``_dino_objective_built`` replaces
    # the module attribute with a factory, or this spy would wrap that factory.
    from otuformer.training.dino import DINOObjective

    real_teacher_targets = DINOObjective.teacher_targets
    real_update_center = DINOObjective.update_center
    real_step = torch.optim.AdamW.step
    real_update_teacher = pretrain_loop_module.update_teacher
    events = {"target_builds": 0, "center_updates": 0, "steps": 0}
    order: list[str] = []
    _dino_objective_built(monkeypatch, captured)

    def targets_spy(self, teacher_logits, teacher_temp):
        events["target_builds"] += 1
        events["pre_ema_logits"] = [
            tensor.detach().clone() for tensor in teacher_logits
        ]
        return real_teacher_targets(self, teacher_logits, teacher_temp)

    def center_spy(self, teacher_logits):
        events["center_updates"] += 1
        order.append("center")
        events["center_input"] = [tensor.detach().clone() for tensor in teacher_logits]
        return real_update_center(self, teacher_logits)

    def step_spy(self, *args, **kwargs):
        events["steps"] += 1
        order.append("step")
        return real_step(self, *args, **kwargs)

    def teacher_ema_spy(student, teacher, momentum):
        # Identity decides which EMA this is: the objective owns the prototype
        # pair, so a missing or reordered prototype update changes the sequence.
        objective = captured[0]
        if student is objective.student_prototypes:
            order.append("prototype_ema")
        elif getattr(student, "backbone", None) is not None:
            order.append("encoder_ema")
        else:
            order.append("other_ema")
        return real_update_teacher(student, teacher, momentum)

    monkeypatch.setattr(DINOObjective, "teacher_targets", targets_spy)
    monkeypatch.setattr(DINOObjective, "update_center", center_spy)
    monkeypatch.setattr(torch.optim.AdamW, "step", step_spy)
    monkeypatch.setattr(pretrain_loop_module, "update_teacher", teacher_ema_spy)

    pretrain_loop_module.run_pretrain(_args(tmp_path, max_epochs=1, log_every_n_steps=1000))

    assert events["target_builds"] == events["steps"]
    assert events["center_updates"] == events["steps"]
    # The exact per-step sequence, repeated once per optimizer step.
    expected_step = ["step", "encoder_ema", "prototype_ema", "center"]
    assert order == expected_step * events["steps"]
    # The center consumes the same cached pre-EMA logits used for the targets.
    for cached, used in zip(events["pre_ema_logits"], events["center_input"]):
        torch.testing.assert_close(cached, used)
    expected = torch.cat(events["pre_ema_logits"]).mean(dim=0, keepdim=True) * 0.1
    torch.testing.assert_close(captured[-1].cls_center, expected, rtol=1e-5, atol=1e-6)


def test_dino_prototype_scales_remain_one(tmp_path, monkeypatch):
    _images(tmp_path)
    captured: list = []
    _dino_objective_built(monkeypatch, captured)
    pretrain_loop_module.run_pretrain(_args(tmp_path, max_epochs=2))

    state = captured[-1].state_dict()
    for key in ("student_prototypes.weight_g", "teacher_prototypes.weight_g"):
        assert torch.equal(state[key], torch.ones(11, 1, device=state[key].device)), key
    assert "center" not in state  # SK/EMA center lives under cls_center
    assert "cls_center" in state


def test_dino_single_epoch_warns(tmp_path, capsys):
    _images(tmp_path)
    pretrain_loop_module.run_pretrain(_args(tmp_path, max_epochs=1))
    output = capsys.readouterr().out
    assert "prototype direction" in output
    assert "frozen" in output


def test_relational_loop_is_unchanged(tmp_path):
    _images(tmp_path)
    pretrain_loop_module.run_pretrain(
        _args(
            tmp_path,
            ssl_method="relational",
            ssl_method_explicit=False,
            out_dim=8,
            out_dim_explicit=False,
            dino_prototypes_explicit=False,
            dino_centering_explicit=False,
            koleo_weight_explicit=False,
            dino_prototypes=64,
            dino_centering="ema",
            koleo_weight=0.0,
            max_epochs=1,
            patch_loss="consistency",
        )
    )
    saved = torch.load(tmp_path / "out" / "SSL_latest.pth", weights_only=False)

    # Legacy training state is preserved: encoder center and patch objective.
    assert "center" in saved
    assert "patch_objective" in saved
    assert saved["config"]["out_dim"] == 8
    assert torch.count_nonzero(saved["center"]) > 0


def _metric_rows(root: Path) -> list[dict[str, str]]:
    import csv as _csv

    with (root / "out" / "logs" / "instant_metrics.pretrain.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return list(_csv.DictReader(handle))


def test_dino_instant_metrics_rows_end_to_end(tmp_path):
    _images(tmp_path)
    pretrain_loop_module.run_pretrain(
        _args(
            tmp_path,
            max_epochs=1,
            koleo_weight=0.1,
            koleo_weight_explicit=True,
            dino_centering="ema",
        )
    )
    row = _metric_rows(tmp_path)[-1]

    assert row["ssl_method"] == "dino"
    assert row["dino_centering"] == "ema"
    assert float(row["koleo_weight"]) == 0.1
    # DINO has no legacy teacher center or per-iteration cosine statistic.
    assert row["teacher_center_norm"] == ""
    assert row["cosine_similarity"] == ""
    assert float(row["cls_center_norm"]) > 0.0
    assert row["koleo_status"] == "active"
    assert float(row["koleo_eligible_samples"]) == 4
    # A signed KoLeo contribution retires the ordinary loss fraction.
    assert row["patch_loss_fraction_of_total"] == ""
    for name in (
        "cls_assignment_entropy",
        "cls_prototype_perplexity",
        "cls_active_prototype_ratio",
        "cls_max_prototype_occupancy",
    ):
        float(row[name])
    assert float(row["global_loss_weighted"]) == float(row["global_loss"])


def test_relational_instant_metrics_rows_keep_legacy_meaning(tmp_path, capsys):
    _images(tmp_path)
    pretrain_loop_module.run_pretrain(
        _args(
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
            patch_loss="consistency",
        )
    )
    row = _metric_rows(tmp_path)[-1]

    assert row["ssl_method"] == "relational"
    assert row["dino_centering"] == ""
    # KoLeo is a shared regularizer, so the display always carries its effective
    # value, zero included; persistence still omits the four keys at zero weight.
    assert row["koleo_weight"] == "0.0"
    # KoLeo is shared now, so a zero weight reads as "disabled", not "N/A".
    assert row["koleo_status"] == "disabled"
    assert row["cls_center_norm"] == ""
    float(row["teacher_center_norm"])
    float(row["cosine_similarity"])
    assert float(row["patch_loss_fraction_of_total"]) > 0.0

    # The operator-facing SSL config line is printed once, after Parameters and
    # beside the patch config, and its display projection carries nulls for
    # inactive fields without spreading them into the persisted config.
    out = capsys.readouterr().out
    assert '"ssl_method": "relational"' in out
    assert '"dino_prototypes": null' in out
    assert out.count("SSL config:") == 1


def test_dino_singleton_batch_records_koleo_skip_end_to_end(tmp_path):
    _images(tmp_path, rows=4)
    pretrain_loop_module.run_pretrain(
        _args(
            tmp_path,
            batch_size=1,
            koleo_weight=0.1,
            koleo_weight_explicit=True,
            patch_loss="consistency",
        )
    )
    rows = _metric_rows(tmp_path)

    assert rows  # at least one optimizer step ran
    for row in rows:
        # A singleton batch skips KoLeo but keeps numeric zeros and its counts.
        assert row["koleo_status"] == "skipped"
        assert float(row["koleo_loss_raw"]) == 0.0
        assert float(row["koleo_loss_weighted"]) == 0.0
        assert float(row["koleo_eligible_samples"]) == 0
        assert float(row["koleo_skipped_samples"]) == 2
        # Retirement of the fraction column covers skipped batches too.
        assert row["patch_loss_fraction_of_total"] == ""


def test_dino_ce_only_backward_reaches_backbone_projector_and_prototypes(
    tmp_path, monkeypatch
):
    """Real loop, patch/KoLeo/local disabled, absolute epoch 1, no injection."""
    _images(tmp_path)
    captured: list = []
    teachers: list = []
    students: list = []
    _dino_objective_built(monkeypatch, captured)
    _capture_teacher(monkeypatch, teachers)
    stacked_update = pretrain_loop_module.update_teacher

    def update_spy(student, teacher, momentum):
        if hasattr(teacher, "backbone"):
            students.append(student)
        return stacked_update(student, teacher, momentum)

    monkeypatch.setattr(pretrain_loop_module, "update_teacher", update_spy)

    real_norm = pretrain_loop_module._compute_grad_norm_parameters
    calls = {"count": 0}
    observed: dict = {}

    def norm_spy(params):
        calls["count"] += 1
        if calls["count"] == 2 and not observed:
            student, teacher, objective = students[-1], teachers[-1], captured[-1]
            observed["backbone"] = [
                p.grad for p in student.backbone.parameters()
            ]
            observed["projector"] = [
                p.grad for p in student.projector.parameters()
            ]
            observed["prototype"] = objective.trainable_parameters()[0].grad
            observed["teacher_encoder"] = [p.grad for p in teacher.parameters()]
            observed["teacher_prototype"] = [
                p.grad for p in objective.teacher_prototypes.parameters()
            ]
            observed["fixed_scales"] = [
                objective.student_prototypes.weight_g.grad,
                objective.teacher_prototypes.weight_g.grad,
            ]
        return real_norm(params)

    monkeypatch.setattr(pretrain_loop_module, "_compute_grad_norm_parameters", norm_spy)
    pretrain_loop_module.run_pretrain(
        _args(
            tmp_path,
            max_epochs=2,
            local_crops=0,
            koleo_weight=0.0,
            koleo_weight_explicit=True,
            patch_loss="none",
        )
    )

    def usable(grads):
        return grad is not None
    assert any(
        g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
        for g in observed["backbone"]
    )
    assert any(
        g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
        for g in observed["projector"]
    )  # a detached student bottleneck cannot satisfy this
    assert observed["prototype"] is not None
    assert torch.isfinite(observed["prototype"]).all()
    assert observed["prototype"].abs().sum() > 0
    assert all(g is None for g in observed["teacher_encoder"])
    assert all(g is None for g in observed["teacher_prototype"])
    assert all(g is None for g in observed["fixed_scales"])


def test_non_finite_gradient_norm_stops_the_step(tmp_path, monkeypatch):
    """A finite loss is not enough: the backward pass can still overflow."""
    _images(tmp_path)
    monkeypatch.setattr(
        pretrain_loop_module, "_compute_grad_norm_parameters", lambda params: float("inf")
    )
    with pytest.raises(ValueError, match="gradient norm became non-finite"):
        pretrain_loop_module.run_pretrain(_args(tmp_path))
    assert not (tmp_path / "out" / "SSL_latest.pth").exists()


def test_clip_reduction_overflow_stops_the_step(tmp_path, monkeypatch):
    """The check and the clip reduce differently; the clip must refuse itself.

    Per-parameter norms squared in Python float64 stay finite at 1e19, while the
    float32 accumulation inside ``clip_grad_norm_`` overflows, zeroes every
    gradient and would otherwise still let the optimizer and the teacher advance.
    """
    _images(tmp_path)
    real_clip = torch.nn.utils.clip_grad_norm_
    steps = {"count": 0}
    real_step = torch.optim.AdamW.step

    def step_spy(self, *args, **kwargs):
        steps["count"] += 1
        return real_step(self, *args, **kwargs)

    def clip_spy(params, max_norm, **kwargs):
        # The guard above must not already have refused; then the real reduction
        # is the one that overflows.
        for parameter in params:
            if parameter.grad is not None:
                parameter.grad.fill_(1e19)
        return real_clip(params, max_norm, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip_spy)
    monkeypatch.setattr(torch.optim.AdamW, "step", step_spy)

    with pytest.raises(ValueError, match="gradient norm became non-finite"):
        pretrain_loop_module.run_pretrain(_args(tmp_path, max_epochs=1))
    assert steps["count"] == 0
    assert not (tmp_path / "out" / "SSL_latest.pth").exists()
    assert not (tmp_path / "out" / "SSL_epoch_0001.pth").exists()
