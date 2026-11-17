"""Live ranking lifecycle: skipped-step head state, optimizer reach, recovery."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from otuformer.training import pretrain_loop as pretrain_loop_module

SIZE = 64


def _dataset(tmp_path, rows=4):
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    for index in range(rows):
        Image.new("RGB", (64, 64), color=(index * 40, 10, 10)).save(
            images / f"img_{index}.jpg"
        )
    frame = pd.DataFrame({"image": [f"img_{i}.jpg" for i in range(rows)]})
    csv = tmp_path / "data.csv"
    frame.to_csv(csv, index=False)
    return csv, images


def _args(tmp_path, name, **overrides):
    base = dict(
        seed=42, cpus=0, device="cpu", out_dir=str(tmp_path / name),
        model_name="vit_tiny_patch16_224", register_tokens=None, out_dim=16,
        ssl_method="relational", ssl_method_explicit=False,
        koleo_weight=0.0, koleo_weight_explicit=False,
        max_epochs=1, lr=1e-3, weight_decay=0.05, warmup_epochs=0,
        augmentation=None, orientation_policy=None,
        global_crop_size=SIZE, local_crop_size=32, local_crops=1,
        mask_ratio="auto", lambda_local=1.5, lambda_mask=1.0,
        patch_loss="none", masking_strategy="random",
        teacher_momentum=0.995, teacher_momentum_end=0.999,
        student_temp=0.1, teacher_temp_start=0.04, teacher_temp_end=0.07,
        teacher_temp_warmup_ratio=None,
        relational_teacher_views="weak", relational_center_source=None,
        relational_pairing="cross", relational_queue_size=0,
        resume="", init_checkpoint="", init_source=None, init_source_explicit=False,
        log_every_n_steps=1000, save_every_epochs=1, keep_last_checkpoints=1,
        visualize_data="", extract_size=None, metrics_sample_size=10,
        umap_n_neighbors=5, umap_min_dist=0.1, umap_metric="cosine",
        visualize_class_number=5, compute_embedding_metrics=False,
        batch_size=2, num_workers=0,
        coordinate_weight=0.0, coordinate_weight_explicit=False,
        dense_query_points=49, dense_query_points_explicit=False,
        ranking_weight=0.0, ranking_weight_explicit=False,
        ranking_reference_points=16, ranking_reference_points_explicit=False,
    )
    csv, images = _dataset(tmp_path)
    base.update(train_data=str(csv), input_images_dir=str(images))
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("diffsort"),
    reason="the optional ranking package is not installed",
)
def test_ranking_run_persists_heads_and_counts(tmp_path):
    args = _args(
        tmp_path, "rank",
        ranking_weight=0.5, ranking_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
    )
    pretrain_loop_module.run_pretrain(args)
    saved = torch.load(
        Path(args.out_dir) / "SSL_latest.pth", map_location="cpu", weights_only=False
    )
    config = saved["config"]["dense_training_config"]
    assert config["objectives"] == ["ranking"]
    assert config["ranking_config"]["reference_cap"] == 16
    assert config["ranking_config"]["dependency"]["name"] == "diffsort"
    state = saved["dense_training"]
    assert set(state["ranking"]) == {
        "student_head", "teacher_head", "reference_rng", "contributed_steps"
    }
    assert 0 <= state["ranking"]["contributed_steps"] <= state["completed_steps"]
    # Both heads are saved separately and stay distinct.
    student = state["ranking"]["student_head"]
    teacher = state["ranking"]["teacher_head"]
    assert set(student) == set(teacher)
    assert any(
        not torch.equal(student[key], teacher[key]) for key in student
    )


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("diffsort"),
    reason="the optional ranking package is not installed",
)
def test_ranking_head_is_in_the_optimizer_not_the_teacher(tmp_path):
    args = _args(
        tmp_path, "opt",
        ranking_weight=0.3, ranking_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
    )
    captured: dict = {}
    real = pretrain_loop_module._pretrain_optimizer_groups

    def spy(*a, **kw):
        groups = real(*a, **kw)
        captured["ranking"] = kw.get("ranking_objective")
        captured["params"] = [p for g in groups for p in g["params"]]
        return groups

    pretrain_loop_module._pretrain_optimizer_groups = spy
    try:
        pretrain_loop_module.run_pretrain(args)
    finally:
        pretrain_loop_module._pretrain_optimizer_groups = real

    ranking = captured["ranking"]
    assert ranking is not None
    ids = {id(p) for p in captured["params"]}
    assert all(id(p) in ids for p in ranking.student_head.parameters())
    assert not any(id(p) in ids for p in ranking.teacher_head.parameters())


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("diffsort"),
    reason="the optional ranking package is not installed",
)
def test_joint_run_keeps_both_objectives(tmp_path):
    args = _args(
        tmp_path, "joint",
        coordinate_weight=0.4, coordinate_weight_explicit=True,
        ranking_weight=0.4, ranking_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
    )
    pretrain_loop_module.run_pretrain(args)
    saved = torch.load(
        Path(args.out_dir) / "SSL_latest.pth", map_location="cpu", weights_only=False
    )
    assert saved["config"]["dense_training_config"]["objectives"] == [
        "coordinate",
        "ranking",
    ]


def test_dense_resolved_config_is_printed_once_with_sources(tmp_path, capsys):
    import json
    args = _args(tmp_path, "resolved", coordinate_weight=0.4,
                 coordinate_weight_explicit=True, batch_size=4)
    pretrain_loop_module.run_pretrain(args)
    prefix = "Resolved dense config: "
    lines = [line[len(prefix):] for line in capsys.readouterr().out.splitlines()
             if line.startswith(prefix)]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["coordinate_weight"] == {
        "requested": 0.4, "effective": 0.4, "source": "requested", "applicable": True}
    assert record["dense_query_points"]["source"] == "default"
    assert record["ranking_reference_points"]["applicable"] is False
    assert record["ranking_reference_points"]["effective"] is None
    latest = Path(args.out_dir) / "SSL_latest.pth"
    resumed = _args(tmp_path, "resolved", resume=str(latest), max_epochs=2, batch_size=4)
    pretrain_loop_module.run_pretrain(resumed)
    lines = [line[len(prefix):] for line in capsys.readouterr().out.splitlines()
             if line.startswith(prefix)]
    assert len(lines) == 1
    assert json.loads(lines[0])["coordinate_weight"]["source"] == "inherited"


def test_coordinate_only_run_has_no_ranking_state(tmp_path):
    args = _args(
        tmp_path, "coord",
        coordinate_weight=0.4, coordinate_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
    )
    pretrain_loop_module.run_pretrain(args)
    saved = torch.load(
        Path(args.out_dir) / "SSL_latest.pth", map_location="cpu", weights_only=False
    )
    assert "ranking" not in saved["dense_training"]
    assert "ranking_config" not in saved["config"]["dense_training_config"]


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("diffsort"),
    reason="the optional ranking package is not installed",
)
def test_ranking_resume_rejects_conflicting_cap(tmp_path):
    args = _args(
        tmp_path, "cap",
        ranking_weight=0.3, ranking_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
        ranking_reference_points=8, ranking_reference_points_explicit=True,
    )
    pretrain_loop_module.run_pretrain(args)
    latest = Path(args.out_dir) / "SSL_latest.pth"

    resumed = _args(
        tmp_path, "cap", max_epochs=2, resume=str(latest),
        ranking_weight=0.3, ranking_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
        ranking_reference_points=8, ranking_reference_points_explicit=True,
    )
    pretrain_loop_module.run_pretrain(resumed)

    conflicting = _args(
        tmp_path, "cap", max_epochs=2, resume=str(latest),
        ranking_weight=0.3, ranking_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
        ranking_reference_points=12, ranking_reference_points_explicit=True,
    )
    with pytest.raises(ValueError, match="ranking_reference_points"):
        pretrain_loop_module.run_pretrain(conflicting)


def test_ranking_weight_without_package_fails_before_output(tmp_path, monkeypatch):
    import sys

    args = _args(
        tmp_path, "absent",
        ranking_weight=0.3, ranking_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
    )
    monkeypatch.setitem(sys.modules, "diffsort", None)
    out_dir = Path(args.out_dir)
    with pytest.raises(Exception):
        pretrain_loop_module.run_pretrain(args)
    assert not out_dir.exists()


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("diffsort"),
    reason="the optional ranking package is not installed",
)
def test_enabled_checkpoint_is_readable_without_the_optional_package(tmp_path, monkeypatch):
    """Read-only consumers must not need diffsort to read an enabled run."""
    import sys

    from otuformer.embedding.extractor import _load_model

    args = _args(
        tmp_path, "reader",
        ranking_weight=0.3, ranking_weight_explicit=True,
        coordinate_weight=0.3, coordinate_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
    )
    pretrain_loop_module.run_pretrain(args)
    latest = Path(args.out_dir) / "SSL_latest.pth"

    monkeypatch.setitem(sys.modules, "diffsort", None)
    model, _ = _load_model(latest, "vit_tiny_patch16_224", torch.device("cpu"))
    assert model is not None
    saved = torch.load(latest, map_location="cpu", weights_only=False)
    # The training-only heads are not part of the encoder state dict.
    assert not any(key.startswith("student_head") for key in saved["model_state_dict"])


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("diffsort"),
    reason="the optional ranking package is not installed",
)
def test_ranking_run_logs_its_own_raw_loss(tmp_path):
    """Every enabled objective must report its own raw/weighted loss."""
    import csv as _csv

    args = _args(
        tmp_path, "logged",
        ranking_weight=0.3, ranking_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
    )
    pretrain_loop_module.run_pretrain(args)
    path = Path(args.out_dir) / "logs" / "instant_metrics.pretrain.csv"
    header = path.read_text().splitlines()[0].split(",")
    assert "dense_ranking_loss_raw" in header
    assert "dense_ranking_references" in header
    # Ranking-specific population/shortage and eligibility columns, so a run can
    # be audited for contributed ranking signal instead of only Coordinate.
    for column in (
        "dense_ranking_reference_count",
        "dense_ranking_shortage_samples",
        "dense_coordinate_eligible",
    ):
        assert column in header
    rows = list(_csv.DictReader(path.open()))
    assert rows[0]["dense_ranking_status"] in ("observed", "skipped")
    if rows[0]["dense_ranking_status"] == "observed":
        assert float(rows[0]["dense_ranking_loss_raw"]) >= 0.0
    for row in rows:
        for branch, attempts in (("global", 2), ("local", 2)):
            prefix = f"dense_ranking_{branch}"
            assert int(row[f"{prefix}_attempted_pairs"]) == attempts
            assert int(row[f"{prefix}_attempted_samples"]) == attempts * args.batch_size
            assert int(row[f"{prefix}_contributing_pairs"]) + int(row[f"{prefix}_skipped_pairs"]) == attempts
            assert row[f"dense_coordinate_{branch}_attempted_pairs"] == ""


def test_unavailable_dense_statistics_write_blank_not_zero():
    """A missing population is empty in the CSV, never an observed 0 or rho 1."""
    assert pretrain_loop_module._dense_stat_or_blank(None) == ""
    assert pretrain_loop_module._dense_stat_or_blank(0.0) == 0.0
    assert pretrain_loop_module._dense_stat_or_blank(0.5) == 0.5


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("diffsort"),
    reason="the optional ranking package is not installed",
)
def test_ranking_resume_rejects_contribution_count_without_matching_moments(tmp_path):
    """The saved contribution count must match the head's real AdamW state."""
    args = _args(
        tmp_path, "contrib",
        ranking_weight=0.3, ranking_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
        ranking_reference_points=8, ranking_reference_points_explicit=True,
    )
    pretrain_loop_module.run_pretrain(args)
    latest = Path(args.out_dir) / "SSL_latest.pth"
    saved = torch.load(latest, map_location="cpu", weights_only=False)
    contributed = saved["dense_training"]["ranking"]["contributed_steps"]
    assert contributed >= 1

    # A forged zero count over real head moments is not legitimate inactivity.
    forged_zero = dict(saved)
    forged_zero["dense_training"] = {
        **saved["dense_training"],
        "ranking": {
            **saved["dense_training"]["ranking"],
            "contributed_steps": 0,
        },
    }
    forged_zero_path = tmp_path / "forged-zero.pth"
    torch.save(forged_zero, forged_zero_path)
    with pytest.raises(ValueError, match="contributed_steps is 0|moments"):
        pretrain_loop_module.run_pretrain(
            _args(
                tmp_path, "contrib", max_epochs=2, resume=str(forged_zero_path),
                ranking_weight=0.3, ranking_weight_explicit=True,
                dense_query_points=4, dense_query_points_explicit=True,
                ranking_reference_points=8, ranking_reference_points_explicit=True,
            )
        )


def test_enabled_dense_post_step_nonfinite_stops_the_commit(tmp_path, monkeypatch):
    """A normally-returning optimizer that produced NaN must abort the step."""
    original = torch.optim.AdamW.step
    state = {"done": False}

    def corrupting_step(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if not state["done"]:
            with torch.no_grad():
                self.param_groups[0]["params"][0].fill_(float("nan"))
            state["done"] = True
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", corrupting_step)
    args = _args(
        tmp_path, "nonfinite",
        coordinate_weight=0.3, coordinate_weight_explicit=True,
    )
    with pytest.raises(ValueError, match="non-finite"):
        pretrain_loop_module.run_pretrain(args)
    # Nothing was committed: no checkpoint and no completed-epoch state.
    assert not list(Path(args.out_dir).glob("SSL_*.pth"))


def _titled_panel_has_data(axis) -> bool:
    """A panel counts as filled when it or a twinx sibling has a line."""
    if axis.get_lines():
        return True
    for other in axis.figure.axes:
        if other is axis:
            continue
        if axis.get_shared_x_axes().joined(axis, other) and other.get_lines():
            return True
    return False


def test_dense_diagnostics_reach_the_training_curves_pdf(tmp_path):
    """A dense-enabled run must put its diagnostics into the PDF, not only the CSV."""
    import re

    args = _args(
        tmp_path, "pdf",
        coordinate_weight=0.4, coordinate_weight_explicit=True,
    )
    pretrain_loop_module.run_pretrain(args)
    out_dir = Path(args.out_dir)
    pdf_path = out_dir / "logs" / "training_curves_pretrain.pdf"
    assert pdf_path.exists()
    pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf_path.read_bytes()))
    # One existing metrics page plus the appended dense diagnostics page.
    assert pages >= 2, pages

    # The dense page is built from the dense columns, and the title is stable.
    import matplotlib.pyplot as plt
    import pandas as pd
    from otuformer.training import reporting as reporting_module

    logger = reporting_module.InstantMetricsLogger(
        out_dir / "logs" / "instant_metrics.pretrain.csv", dense_enabled=True
    )
    frame = pd.read_csv(out_dir / "logs" / "instant_metrics.pretrain.csv")
    figure = logger._dense_diagnostics_figure(frame, plt)
    assert figure is not None
    titles = {axis.get_title() for axis in figure.axes if axis.get_title()}
    for expected in (
        "Dense raw losses vs existing objectives",
        "Dense weighted contribution",
        "Dense query population",
        "Dense contributing denominators",
        "Dense global denominators",
        "Dense local denominators",
        "Min participating norm",
        "Min interpolation rho",
    ):
        assert expected in titles, sorted(titles)
    plt.close(figure)


def test_all_off_run_keeps_a_single_page_pdf(tmp_path):
    """Without dense columns the PDF keeps exactly its historical page."""
    import re

    args = _args(tmp_path, "plain")
    pretrain_loop_module.run_pretrain(args)
    pdf_path = Path(args.out_dir) / "logs" / "training_curves_pretrain.pdf"
    pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf_path.read_bytes()))
    assert pages == 1, pages


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("diffsort"),
    reason="the optional ranking package is not installed",
)
@pytest.mark.parametrize("tamper", ["step", "nonfinite"])
def test_ranking_resume_rejects_mismatched_head_step_or_moments(tmp_path, tamper):
    """Preflight must reject a corrupt head entry, not defer it to the live check."""
    args = _args(
        tmp_path, f"tamper-{tamper}",
        ranking_weight=0.3, ranking_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
        ranking_reference_points=8, ranking_reference_points_explicit=True,
    )
    pretrain_loop_module.run_pretrain(args)
    latest = Path(args.out_dir) / "SSL_latest.pth"
    saved = torch.load(latest, map_location="cpu", weights_only=False)
    contributed = saved["dense_training"]["ranking"]["contributed_steps"]
    assert contributed >= 1

    # The ranking student head is appended last, so its six parameters own the
    # highest optimizer-state indices.
    system_state = saved["optimizer"]["state"]
    keys = sorted(int(key) for key in system_state)
    head_keys = keys[-6:]
    assert all(
        isinstance(system_state[key]["exp_avg"], torch.Tensor)
        for key in head_keys
    )
    for key in head_keys:
        entry = system_state[key]
        if tamper == "step":
            entry["step"] = torch.tensor(float(999))
        else:
            entry["exp_avg"] = torch.full_like(entry["exp_avg"], float("nan"))
    tampered = tmp_path / f"tampered-{tamper}.pth"
    torch.save(saved, tampered)
    resumed = _args(
        tmp_path, f"tamper-{tamper}", max_epochs=2, resume=str(tampered),
        ranking_weight=0.3, ranking_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
        ranking_reference_points=8, ranking_reference_points_explicit=True,
    )
    # The preflight that runs before any output exists must reject it; a later
    # live-restore rejection would already have touched the run directory.
    from otuformer.training.pretrain_recipe import resolve_pretrain_ssl_config

    with pytest.raises(ValueError, match="ranking head"):
        resolve_pretrain_ssl_config(resumed, saved)
    with pytest.raises(ValueError, match="ranking head"):
        pretrain_loop_module.run_pretrain(resumed)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("diffsort"),
    reason="the optional ranking package is not installed",
)
def test_dino_resume_accepts_a_never_contributed_ranking_head(tmp_path):
    """A head that never contributed is legal state, not missing AdamW state."""
    args = _args(
        tmp_path, "dino-rank",
        ssl_method="dino", ssl_method_explicit=True,
        dino_prototypes=11, dino_prototypes_explicit=True,
        dino_centering="ema", dino_centering_explicit=True,
        relational_queue_size=None,
        ranking_weight=0.3, ranking_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
        ranking_reference_points=8, ranking_reference_points_explicit=True,
    )
    pretrain_loop_module.run_pretrain(args)
    latest = Path(args.out_dir) / "SSL_latest.pth"
    saved = torch.load(latest, map_location="cpu", weights_only=False)
    assert saved["dense_training"]["ranking"]["contributed_steps"] >= 1

    # Forge the legal "ranking never contributed" state: count 0 and no head
    # optimizer entries, with both heads still present.
    system_state = saved["optimizer"]["state"]
    head_keys = sorted(int(key) for key in system_state)[-6:]
    saved["optimizer"]["state"] = {
        key: value for key, value in system_state.items() if int(key) not in head_keys
    }
    saved["dense_training"]["ranking"]["contributed_steps"] = 0
    forged = tmp_path / "dino-never-contributed.pth"
    torch.save(saved, forged)

    resumed = _args(
        tmp_path, "dino-rank", max_epochs=2, resume=str(forged),
        ssl_method="dino", ssl_method_explicit=True,
        dino_prototypes=11, dino_prototypes_explicit=True,
        dino_centering="ema", dino_centering_explicit=True,
        relational_queue_size=None,
        ranking_weight=0.3, ranking_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
        ranking_reference_points=8, ranking_reference_points_explicit=True,
    )
    pretrain_loop_module.run_pretrain(resumed)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("diffsort"),
    reason="the optional ranking package is not installed",
)
def test_substituted_non_recipe_ranking_head_is_rejected_before_output(tmp_path):
    """A head whose widths are not the frozen recipe must fail in preflight."""
    args = _args(
        tmp_path, "headshape",
        ranking_weight=0.3, ranking_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
        ranking_reference_points=8, ranking_reference_points_explicit=True,
    )
    pretrain_loop_module.run_pretrain(args)
    latest = Path(args.out_dir) / "SSL_latest.pth"
    saved = torch.load(latest, map_location="cpu", weights_only=False)

    # Substitute a plausible but non-recipe head in both head state dicts.
    width = saved["dense_training"]["ranking"]["student_head"]["0.weight"].shape[1]
    import torch.nn as nn

    small = nn.Sequential(
        nn.Linear(int(width), 16), nn.GELU(), nn.Linear(16, 16), nn.GELU(),
        nn.Linear(16, 8),
    )
    for key in ("student_head", "teacher_head"):
        saved["dense_training"]["ranking"][key] = {
            name: value.detach().clone() for name, value in small.state_dict().items()
        }
    forged = tmp_path / "forged-head.pth"
    torch.save(saved, forged)

    resumed = _args(
        tmp_path, "headshape", max_epochs=2, resume=str(forged),
        ranking_weight=0.3, ranking_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
        ranking_reference_points=8, ranking_reference_points_explicit=True,
    )
    from otuformer.training.pretrain_recipe import resolve_pretrain_ssl_config

    with pytest.raises(ValueError, match="frozen recipe|does not match"):
        resolve_pretrain_ssl_config(resumed, saved)
    out_dir = Path(resumed.out_dir)
    before = sorted(p.name for p in out_dir.iterdir())
    with pytest.raises(ValueError, match="frozen recipe|does not match"):
        pretrain_loop_module.run_pretrain(resumed)
    # The output directory was not extended by the rejected run.
    assert sorted(p.name for p in out_dir.iterdir()) == before


def test_coordinate_only_run_does_not_report_ranking_skips(tmp_path):
    """A switched-off objective is ``disabled``, not a skipped step, and has no panel."""
    import pandas as pd
    import matplotlib.pyplot as plt
    from otuformer.training import reporting as reporting_module

    args = _args(
        tmp_path, "coordstatus",
        coordinate_weight=0.4, coordinate_weight_explicit=True,
    )
    pretrain_loop_module.run_pretrain(args)
    csv_path = Path(args.out_dir) / "logs" / "instant_metrics.pretrain.csv"
    frame = pd.read_csv(csv_path)
    assert set(frame["dense_ranking_status"].unique()) == {"disabled"}
    assert "skipped" not in set(frame["dense_coordinate_status"].unique())

    logger = reporting_module.InstantMetricsLogger(csv_path, dense_enabled=True)
    figure = logger._dense_diagnostics_figure(frame, plt)
    try:
        titles = {axis.get_title() for axis in figure.axes if axis.get_title()}
        assert "Ranking references" not in titles
        # No objective may leave a titled-but-empty panel behind; in particular a
        # Coordinate-only run must not show a permutation-probability axis.
        for axis in figure.axes:
            if axis.get_title():
                assert _titled_panel_has_data(axis), axis.get_title()
        assert "Min student permutation probability" not in titles
        for axis in figure.axes:
            for line in axis.get_lines():
                assert "Ranking skipped" not in (line.get_label() or "")
        # The existing-objectives curve must not carry the dense contribution.
        for axis in figure.axes:
            for line in axis.get_lines():
                if "Existing objectives" in (line.get_label() or ""):
                    total = frame["loss"].astype(float)
                    dense = (
                        frame["dense_coordinate_loss_weighted"]
                        .astype(float)
                        .fillna(0.0)
                        + frame["dense_ranking_loss_weighted"]
                        .astype(float)
                        .fillna(0.0)
                    )
                    expected = (total - dense).to_numpy()
                    import numpy as np

                    np.testing.assert_allclose(
                        np.asarray(line.get_ydata(), dtype=float), expected
                    )
                    return
        raise AssertionError("the existing-objectives curve is missing")
    finally:
        plt.close(figure)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("diffsort"),
    reason="the optional ranking package is not installed",
)
@pytest.mark.parametrize(
    "damage", ["no_optimizer", "no_center", "no_rng_state", "no_teacher", "missing_key"]
)
def test_dense_resume_requires_the_complete_saved_state(tmp_path, damage):
    """Dense recovery is strict like FIFO: no fresh optimizer/center/encoder."""
    args = _args(
        tmp_path, f"strict-{damage}",
        ranking_weight=0.3, ranking_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
        ranking_reference_points=8, ranking_reference_points_explicit=True,
    )
    pretrain_loop_module.run_pretrain(args)
    latest = Path(args.out_dir) / "SSL_latest.pth"
    saved = torch.load(latest, map_location="cpu", weights_only=False)
    if damage == "no_optimizer":
        saved.pop("optimizer")
    elif damage == "no_center":
        saved.pop("center")
    elif damage == "no_rng_state":
        saved.pop("rng_state")
    elif damage == "no_teacher":
        saved.pop("teacher")
    else:
        # Drop one backbone tensor from the teacher encoder state.
        key = sorted(
            name for name in saved["teacher"] if name.startswith("backbone.")
        )[0]
        del saved["teacher"][key]
    forged = tmp_path / f"strict-{damage}.pth"
    torch.save(saved, forged)

    resumed = _args(
        tmp_path, f"strict-{damage}", max_epochs=2, resume=str(forged),
        ranking_weight=0.3, ranking_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
        ranking_reference_points=8, ranking_reference_points_explicit=True,
    )
    from otuformer.training.pretrain_recipe import resolve_pretrain_ssl_config

    with pytest.raises(ValueError):
        resolve_pretrain_ssl_config(resumed, saved)
    out_dir = Path(resumed.out_dir)
    before = sorted(p.name for p in out_dir.iterdir())
    with pytest.raises(ValueError):
        pretrain_loop_module.run_pretrain(resumed)
    assert sorted(p.name for p in out_dir.iterdir()) == before


def test_all_unavailable_dense_segment_leaves_no_empty_panels(tmp_path):
    """A segment where both objectives were enabled but unavailable."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from otuformer.training import reporting as reporting_module

    path = tmp_path / "instant_metrics.pretrain.csv"
    fields = list(reporting_module._PRETRAIN_DENSE_CURRENT_FIELDS)
    rows = []
    for index in range(3):
        row = {name: "" for name in fields}
        row.update(
            iteration=str(index),
            epoch="0",
            step=str(index),
            loss="1.0",
            lr="0.001",
            dense_coordinate_status="skipped",
            dense_ranking_status="skipped",
            # Real numeric zeros: the counts panels legitimately draw, while the
            # loss/minimum panels have no population and must stay closed.
            dense_coordinate_points="0",
            dense_coordinate_samples="0",
            dense_coordinate_pairs="0",
            dense_coordinate_eligible="0",
            dense_ranking_points="0",
            dense_ranking_samples="0",
            dense_ranking_pairs="0",
        )
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        import csv as csv_module

        writer = csv_module.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    logger = reporting_module.InstantMetricsLogger(path, dense_enabled=True)
    frame = pd.read_csv(path)
    figure = logger._dense_diagnostics_figure(frame, plt)
    try:
        assert figure is not None
        for axis in figure.axes:
            if axis.get_title():
                assert _titled_panel_has_data(axis), axis.get_title()
    finally:
        plt.close(figure)


def test_existing_objectives_curve_is_not_fabricated_without_a_loss_column(tmp_path):
    """No ``loss`` column means no existing-objectives curve at all."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from otuformer.training import reporting as reporting_module

    frame = pd.DataFrame(
        {
            "iteration": [0, 1],
            "dense_coordinate_loss_raw": [0.3, 0.2],
            "dense_coordinate_loss_weighted": [0.3, 0.2],
            "dense_coordinate_status": ["observed", "observed"],
            "dense_ranking_status": ["disabled", "disabled"],
        }
    )
    logger = reporting_module.InstantMetricsLogger(
        tmp_path / "instant_metrics.pretrain.csv", dense_enabled=True
    )
    figure = logger._dense_diagnostics_figure(frame, plt)
    try:
        for axis in figure.axes:
            for line in axis.get_lines():
                assert "Existing objectives" not in (line.get_label() or "")
    finally:
        plt.close(figure)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("diffsort"),
    reason="the optional ranking package is not installed",
)
@pytest.mark.parametrize("damage", ["torch_only", "all_bad"])
def test_dense_resume_rejects_an_unrestorable_rng_state(tmp_path, damage):
    """A non-empty rng_state dict is not evidence that it restores."""
    args = _args(
        tmp_path, f"rng-{damage}",
        ranking_weight=0.3, ranking_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
        ranking_reference_points=8, ranking_reference_points_explicit=True,
    )
    pretrain_loop_module.run_pretrain(args)
    latest = Path(args.out_dir) / "SSL_latest.pth"
    saved = torch.load(latest, map_location="cpu", weights_only=False)
    if damage == "torch_only":
        saved["rng_state"] = {"torch": saved["rng_state"]["torch"]}
    else:
        saved["rng_state"] = {"python": "bad", "numpy": "bad", "torch": "bad"}
    forged = tmp_path / f"rng-{damage}.pth"
    torch.save(saved, forged)

    resumed = _args(
        tmp_path, f"rng-{damage}", max_epochs=2, resume=str(forged),
        ranking_weight=0.3, ranking_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
        ranking_reference_points=8, ranking_reference_points_explicit=True,
    )
    from otuformer.training.pretrain_recipe import resolve_pretrain_ssl_config

    with pytest.raises(ValueError, match="rng_state"):
        resolve_pretrain_ssl_config(resumed, saved)
    out_dir = Path(resumed.out_dir)
    before = sorted(p.name for p in out_dir.iterdir())
    with pytest.raises(ValueError, match="rng_state"):
        pretrain_loop_module.run_pretrain(resumed)
    assert sorted(p.name for p in out_dir.iterdir()) == before


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("diffsort"),
    reason="the optional ranking package is not installed",
)
def test_dense_resume_applies_the_target_device_rng_policy(tmp_path):
    """A CUDA-shaped rng_state on a CPU target must be dropped, not applied."""
    args = _args(
        tmp_path, "rng-device",
        ranking_weight=0.3, ranking_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=True,
        ranking_reference_points=8, ranking_reference_points_explicit=True,
    )
    pretrain_loop_module.run_pretrain(args)
    latest = Path(args.out_dir) / "SSL_latest.pth"
    saved = torch.load(latest, map_location="cpu", weights_only=False)
    from otuformer.training.pretrain_recipe import _verify_restorable_rng_state

    cpu = torch.device("cpu")
    # A malformed CUDA entry is a hard error even on CPU (structure is checked).
    bad_cuda = dict(saved["rng_state"])
    bad_cuda["cuda"] = "bad"
    with pytest.raises(ValueError, match="rng_state"):
        _verify_restorable_rng_state(
            {"rng_state": bad_cuda}, label="dense-enabled", target_device=cpu
        )
    # An empty CUDA sequence is likewise rejected: absence is encoded by omission.
    empty_cuda = dict(saved["rng_state"])
    empty_cuda["cuda"] = []
    with pytest.raises(ValueError, match="rng_state"):
        _verify_restorable_rng_state(
            {"rng_state": empty_cuda}, label="dense-enabled", target_device=cpu
        )
    # A well-formed CUDA-shaped payload on CPU is accepted (structure only).
    shaped = dict(saved["rng_state"])
    shaped["cuda"] = [torch.zeros(16, dtype=torch.uint8)]
    _verify_restorable_rng_state(
        {"rng_state": shaped}, label="dense-enabled", target_device=cpu
    )
    # Make absence explicit even when the test host itself has CUDA streams.
    missing_cuda = dict(saved["rng_state"])
    missing_cuda.pop("cuda", None)
    with pytest.raises(ValueError, match="CUDA"):
        _verify_restorable_rng_state(
            {"rng_state": missing_cuda},
            label="dense-enabled",
            target_device=torch.device("cuda:0"),
        )


@pytest.mark.parametrize(
    ("ssl_method", "coordinate_weight"),
    [("relational", 0.3), ("dino", 0.3), ("dino", 0.0)],
)
def test_actual_resume_uses_target_filtered_rng_copy(
    tmp_path, monkeypatch, ssl_method, coordinate_weight
):
    """Exact CPU recovery must restore the filtered copy, not the saved dict."""
    options = dict(
        ssl_method=ssl_method, ssl_method_explicit=True,
        coordinate_weight=coordinate_weight, coordinate_weight_explicit=True,
        relational_queue_size=None if ssl_method == "dino" else 0,
    )
    if ssl_method == "dino":
        options.update(dino_prototypes=11, dino_centering="ema")
    args = _args(tmp_path, "rng-source", **options)
    pretrain_loop_module.run_pretrain(args)
    saved = torch.load(
        Path(args.out_dir) / "SSL_latest.pth", map_location="cpu", weights_only=False
    )
    saved["rng_state"]["cuda"] = [torch.zeros(16, dtype=torch.uint8)]
    source = tmp_path / "cuda-shaped.pth"
    torch.save(saved, source)
    restored = []

    class StopBeforeResumedUpdate(Exception):
        pass

    def observe_restore(state):
        restored.append(state)
        raise StopBeforeResumedUpdate

    monkeypatch.setattr(pretrain_loop_module, "_restore_rng_state", observe_restore)
    resumed = _args(
        tmp_path, "rng-resumed", max_epochs=2, resume=str(source), **options
    )
    with pytest.raises(StopBeforeResumedUpdate):
        pretrain_loop_module.run_pretrain(resumed)
    assert len(restored) == 1
    assert "cuda" not in restored[0]
    assert {"python", "numpy", "torch"} <= restored[0].keys()
    assert "cuda" in saved["rng_state"]


def test_dense_startup_probe_preserves_generators(tmp_path, monkeypatch):
    pytest.importorskip("diffsort")
    runtimes = []
    before = []
    real_runtime = pretrain_loop_module.new_dense_runtime
    real_unpack = pretrain_loop_module._unpack_pretrain_batch
    unpacked = []
    def unpack(*a, **k):
        result = real_unpack(*a, **k)
        unpacked.append(1)
        return result
    def runtime(*a, **k):
        value = real_runtime(*a, **k)
        runtimes.append(value)
        before.append((value.query_generator.get_state().clone(),
                       value.reference_generator.get_state().clone()))
        return value
    class StopAtFirstTrainingDenseCall(Exception):
        pass
    def terms(settings, runtime, *a, **k):
        assert len(unpacked) == 2  # Startup completed; this is the training batch.
        assert runtime is runtimes[0]
        assert torch.equal(runtime.query_generator.get_state(), before[0][0])
        assert torch.equal(runtime.reference_generator.get_state(), before[0][1])
        assert runtime.ranking_contributed_steps == 0
        raise StopAtFirstTrainingDenseCall
    monkeypatch.setattr(pretrain_loop_module, "new_dense_runtime", runtime)
    monkeypatch.setattr(pretrain_loop_module, "_unpack_pretrain_batch", unpack)
    monkeypatch.setattr(pretrain_loop_module, "_dense_terms", terms)
    args = _args(tmp_path, "startup", ranking_weight=0.3, ranking_weight_explicit=True,
                 dense_query_points=4, dense_query_points_explicit=True, batch_size=4)
    with pytest.raises(StopAtFirstTrainingDenseCall):
        pretrain_loop_module.run_pretrain(args)
    assert len(runtimes) == 1


def test_ranking_contributed_then_skipped_step(tmp_path, monkeypatch):
    pytest.importorskip("diffsort")
    from otuformer.training.neighbor_ranking import NeighborRankingObjective
    real_terms = pretrain_loop_module._dense_terms
    real_step = torch.optim.AdamW.step
    real_ema = NeighborRankingObjective.update_teacher
    calls = []
    steps = []
    emas = []
    def terms(settings, runtime, ranking, *a, **k):
        calls.append(ranking)
        if len(calls) == 1:
            result = real_terms(settings, runtime, ranking, *a, **k)
            assert result[2] is not None and result[2].requires_grad
            return result
        return None, {}, None, {}
    def step(optimizer, *a, **k):
        ranking = calls[-1]
        params = list(ranking.student_head.parameters())
        if steps:
            for p, previous in zip(params, steps[0]):
                assert p.grad is None
                assert torch.equal(p, previous[0])
                assert optimizer.state[p].keys() == previous[1].keys()
                for key, value in optimizer.state[p].items():
                    assert torch.equal(value, previous[1][key])
        result = real_step(optimizer, *a, **k)
        current = [(p.detach().clone(), {k: v.clone() for k, v in optimizer.state[p].items()})
                   for p in params]
        if steps:
            for (param, state), (old, old_state) in zip(current, steps[0]):
                assert torch.equal(param, old)
                assert all(torch.equal(state[key], old_state[key]) for key in state)
        steps.append(current)
        return result
    def ema(self, momentum):
        old = [p.detach().clone() for p in self.teacher_head.parameters()]
        real_ema(self, momentum)
        for teacher, student, previous in zip(self.teacher_head.parameters(),
                                             self.student_head.parameters(), old):
            expected = previous.mul(float(momentum)).add(student.detach(), alpha=1 - float(momentum))
            assert torch.equal(teacher, expected)
        emas.append(1)
    monkeypatch.setattr(pretrain_loop_module, "_dense_terms", terms)
    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    monkeypatch.setattr(NeighborRankingObjective, "update_teacher", ema)
    args = _args(tmp_path, "contribute-skip", ranking_weight=0.3,
                 ranking_weight_explicit=True, dense_query_points=4,
                 dense_query_points_explicit=True)
    pretrain_loop_module.run_pretrain(args)
    saved = torch.load(Path(args.out_dir) / "SSL_latest.pth", weights_only=False)
    assert len(steps) == len(emas) == 2
    assert saved["dense_training"]["ranking"]["contributed_steps"] == 1


def test_prefix_contract_global_and_local_registers(tmp_path, monkeypatch):
    from otuformer.training.model import OTUFormerEncoder
    real_init = OTUFormerEncoder.__init__
    real_grid = pretrain_loop_module._to_feature_grid
    real_terms = pretrain_loop_module._dense_terms
    seen = []
    def init(self, *a, **k):
        k["pretrained"] = False
        real_init(self, *a, **k)
    def grid(tokens, prefix, grid_size):
        result = real_grid(tokens, prefix, grid_size)
        assert tokens.shape[1] == prefix + grid_size[0] * grid_size[1]
        assert torch.equal(result.flatten(1, 2), tokens[:, prefix:])
        seen.append(prefix)
        return result
    def terms(settings, runtime, ranking, student, teacher, *a, **k):
        assert set(student) == {0, 1, 2} and set(teacher) == {0, 1}
        return real_terms(settings, runtime, ranking, student, teacher, *a, **k)
    monkeypatch.setattr(OTUFormerEncoder, "__init__", init)
    monkeypatch.setattr(pretrain_loop_module, "_to_feature_grid", grid)
    monkeypatch.setattr(pretrain_loop_module, "_dense_terms", terms)
    args = _args(tmp_path, "register-prefix", register_tokens=4,
                 coordinate_weight=0.3, coordinate_weight_explicit=True, batch_size=4)
    pretrain_loop_module.run_pretrain(args)
    assert seen.count(5) == 4 and seen.count(0) == 1


@pytest.mark.parametrize("ssl_method", ["relational", "dino"])
@pytest.mark.parametrize("patch_mode", ["none", "consistency", "masked-feature", "ibot"])
@pytest.mark.parametrize("dense_mode", ["off", "coordinate", "ranking", "joint"])
def test_ssl_patch_dense_lifecycle_matrix(
    tmp_path, monkeypatch, ssl_method, patch_mode, dense_mode
):
    """Bounded real one-update save/resume for all 32 supported combinations."""
    if dense_mode in ("ranking", "joint"):
        pytest.importorskip("diffsort", reason="the optional ranking package is not installed")
    from otuformer.training.model import OTUFormerEncoder

    real_init = OTUFormerEncoder.__init__

    def offline_init(self, *args, **kwargs):
        kwargs["pretrained"] = False
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(OTUFormerEncoder, "__init__", offline_init)
    # Plotting is covered separately; this matrix owns real forward/backward/state.
    monkeypatch.setattr(pretrain_loop_module.InstantMetricsLogger, "plot", lambda *a, **k: None)
    options = dict(
        ssl_method=ssl_method, ssl_method_explicit=True,
        patch_loss=patch_mode, batch_size=4, local_crop_size=SIZE,
        relational_queue_size=None if ssl_method == "dino" else 0,
        coordinate_weight=0.2 if dense_mode in ("coordinate", "joint") else 0.0,
        ranking_weight=0.2 if dense_mode in ("ranking", "joint") else 0.0,
        coordinate_weight_explicit=True, ranking_weight_explicit=True,
        dense_query_points=4, dense_query_points_explicit=dense_mode != "off",
        ranking_reference_points=4,
        ranking_reference_points_explicit=dense_mode in ("ranking", "joint"),
        ibot_prototypes=16, ibot_prototypes_explicit=patch_mode == "ibot",
    )
    if ssl_method == "dino":
        options.update(dino_prototypes=11, dino_centering="ema")
    # CPU is deliberate: exact save/resume and patch-position contracts, not device qualification.
    args = _args(tmp_path, "matrix", **options)
    pretrain_loop_module.run_pretrain(args)
    latest = Path(args.out_dir) / "SSL_latest.pth"
    first = torch.load(latest, map_location="cpu", weights_only=False)
    assert first["iteration"] == 1
    resumed = _args(tmp_path, "matrix", max_epochs=2, resume=str(latest), **options)
    pretrain_loop_module.run_pretrain(resumed)
    final = torch.load(latest, map_location="cpu", weights_only=False)
    assert final["iteration"] == 2
    assert final["config"]["ssl_method"] == ssl_method
    assert final["config"]["patch_loss"] == patch_mode
    assert ("dense_training" in final) is (dense_mode != "off")
    for branch in ("student", "teacher"):
        assert all(torch.isfinite(value).all() for value in final[branch].values())
    if dense_mode != "off":
        assert final["dense_training"]["completed_steps"] == 2
