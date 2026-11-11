"""Instant-metrics schema migration and DINO/KoLeo diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import pandas as pd
import pytest
import torch

import otuformer.training.pretrain_loop as pretrain_loop_module
from otuformer.training import trainer
import otuformer.training.objectives as objectives_module
import otuformer.training.reporting as reporting_module

LEGACY_HEADER = list(reporting_module._PRETRAIN_LEGACY_FIELDS)
V070_HEADER = list(reporting_module._PRETRAIN_V070_FIELDS)


def _write(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, dtype=str)


@pytest.mark.parametrize("prior", ["legacy", "v070"])
def test_ssl_metrics_migrate_both_prior_headers(tmp_path, prior):
    path = tmp_path / "instant_metrics.pretrain.csv"
    header = LEGACY_HEADER if prior == "legacy" else V070_HEADER
    first = ["" for _ in header]
    first[:3] = ["0", "0", "0"]
    first[3] = "1.5"
    second = ["" for _ in header]
    second[:3] = ["1", "0", "1"]
    second[3] = "0.5"
    if prior == "v070":
        second[header.index("patch_loss_raw")] = "0.25"
    _write(path, header, [first, second])

    pretrain_loop_module.InstantMetricsLogger(path, mode="pretrain")

    frame = _read(path)
    assert list(frame.columns) == list(reporting_module._PRETRAIN_CURRENT_FIELDS)
    # Historical values and order survive.
    assert frame.loc[0, "loss"] == "1.5"
    assert frame.loc[1, "loss"] == "0.5"
    assert frame.loc[0, "ssl_method"] == ""  # unknown, not relational
    assert frame.loc[1, "ssl_method"] == ""
    for name in reporting_module._PRETRAIN_SSL_SETTING_FIELDS:
        assert (frame[name] == "").all(), name
    if prior == "v070":
        assert frame.loc[1, "patch_loss_raw"] == "0.25"


def test_ssl_metrics_migration_is_idempotent(tmp_path):
    path = tmp_path / "instant_metrics.pretrain.csv"
    _write(path, LEGACY_HEADER, [["0", "0", "0", "1.0"] + [""] * (len(LEGACY_HEADER) - 4)])

    logger = pretrain_loop_module.InstantMetricsLogger(path, mode="pretrain")
    logger.log(iteration=1, epoch=0, step=1, loss=0.5, ssl_method="relational")
    before = path.read_bytes()

    pretrain_loop_module.InstantMetricsLogger(path, mode="pretrain")
    assert path.read_bytes() == before

    frame = _read(path)
    # A relational resume file can hold historical blank identity rows followed
    # by current labeled rows.
    assert frame.loc[0, "ssl_method"] == ""
    assert frame.loc[1, "ssl_method"] == "relational"


def test_ssl_metrics_failed_migration_preserves_bytes(tmp_path, monkeypatch):
    path = tmp_path / "instant_metrics.pretrain.csv"
    _write(path, LEGACY_HEADER, [["0", "0", "0", "1.0"] + [""] * (len(LEGACY_HEADER) - 4)])
    old_bytes = path.read_bytes()

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("os.replace", boom)
    with pytest.raises(OSError):
        pretrain_loop_module.InstantMetricsLogger(path, mode="pretrain")

    assert path.read_bytes() == old_bytes
    assert list(tmp_path.glob("*.tmp")) == []


def test_ssl_metrics_reject_unknown_or_duplicate_header(tmp_path):
    unknown = tmp_path / "unknown.csv"
    _write(unknown, ["iteration", "mystery"], [["0", "1"]])
    with pytest.raises(ValueError, match="Unrecognized|duplicate"):
        pretrain_loop_module.InstantMetricsLogger(unknown, mode="pretrain")

    duplicate = tmp_path / "duplicate.csv"
    _write(duplicate, ["iteration", "iteration"], [["0", "1"]])
    with pytest.raises(ValueError, match="duplicate"):
        pretrain_loop_module.InstantMetricsLogger(duplicate, mode="pretrain")

    malformed = tmp_path / "malformed.csv"
    _write(malformed, LEGACY_HEADER, [["0", "0"]])
    with pytest.raises(ValueError, match="malformed"):
        pretrain_loop_module.InstantMetricsLogger(malformed, mode="pretrain")


def test_epoch_and_finetune_headers_remain_unchanged(tmp_path):
    finetune = pretrain_loop_module.InstantMetricsLogger(
        tmp_path / "instant_metrics.finetune.csv", mode="finetune"
    )
    assert list(finetune.fieldnames) == (
        list(reporting_module._PRETRAIN_BASE_FIELDS) + list(reporting_module._FINETUNE_FIELDS)
    )
    pretrain = pretrain_loop_module.InstantMetricsLogger(
        tmp_path / "instant_metrics.pretrain.csv", mode="pretrain"
    )
    assert list(pretrain.fieldnames) == list(reporting_module._PRETRAIN_CURRENT_FIELDS)
    # The extended schema keeps every frozen v0.7.0 column in order.
    assert list(pretrain.fieldnames)[: len(V070_HEADER)] == V070_HEADER


def _logger(tmp_path: Path) -> tuple[pretrain_loop_module.InstantMetricsLogger, Path]:
    path = tmp_path / "instant_metrics.pretrain.csv"
    return pretrain_loop_module.InstantMetricsLogger(path, mode="pretrain"), path


def _row(path: Path, index: int = -1) -> dict[str, str]:
    frame = _read(path)
    return {name: str(frame.loc[frame.index[index], name]) for name in frame.columns}


def _plot_lines(tmp_path: Path, logger: pretrain_loop_module.InstantMetricsLogger):
    """Run plot() and return {panel title: {label: y values}} for assertions.

    monkeypatch-free: matplotlib is already a hard dependency of plot(), and the
    returned mapping is read straight off the closed figure's axes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    captured: dict[str, dict[str, list[float]]] = {}
    original_savefig = plt.Figure.savefig

    def fake_savefig(self, *args, **kwargs):
        # A twin axis (``ax.twinx()``) carries no title and shares the parent's
        # position, so its lines are attributed to the titled parent panel.
        titled = {
            ax.get_position().bounds: ax.get_title()
            for ax in self.axes
            if ax.get_title()
        }
        for ax in self.axes:
            title = ax.get_title() or titled.get(ax.get_position().bounds)
            if not title:
                continue
            captured.setdefault(title, {}).setdefault("__scale__", ax.get_yscale())
            for line in ax.get_lines():
                label = line.get_label()
                if label.startswith("_"):
                    continue
                captured.setdefault(title, {})[label] = [
                    float(v) for v in line.get_ydata()
                ]
        return original_savefig(self, *args, **kwargs)

    plt.Figure.savefig = fake_savefig
    try:
        logger.plot(tmp_path)
    finally:
        plt.Figure.savefig = original_savefig
    return captured


def test_dino_metrics_use_only_cls_center(tmp_path):
    logger, path = _logger(tmp_path)
    logger.log(
        iteration=0,
        epoch=0,
        step=0,
        teacher_center_norm=None,
        cosine_similarity=None,
        cls_center_norm=1.25,
        ssl_method="dino",
        dino_centering="ema",
        koleo_weight=0.0,
    )
    row = _row(path)
    assert row["teacher_center_norm"] == ""
    assert row["cosine_similarity"] == ""
    assert float(row["cls_center_norm"]) == pytest.approx(1.25)

    # Sinkhorn-Knopp has no active CLS center either.
    logger.log(iteration=1, epoch=0, step=1, ssl_method="dino",
               dino_centering="sinkhorn-knopp", cls_center_norm=None,
               teacher_center_norm=None, cosine_similarity=None)
    assert _row(path)["cls_center_norm"] == ""

    # Relational keeps its D-dimensional teacher center and cosine.
    logger.log(iteration=2, epoch=0, step=2, ssl_method="relational",
               teacher_center_norm=0.5, cosine_similarity=0.75,
               cls_center_norm=None, dino_centering=None, koleo_weight=None)
    relational = _row(path)
    assert float(relational["teacher_center_norm"]) == pytest.approx(0.5)
    assert float(relational["cosine_similarity"]) == pytest.approx(0.75)
    assert relational["dino_centering"] == ""
    assert relational["koleo_weight"] == ""


def test_koleo_status_and_loss_cells(tmp_path):
    logger, path = _logger(tmp_path)

    logger.log(iteration=0, epoch=0, step=0, ssl_method="relational",
               koleo_status="not-applicable", koleo_weight=None,
               koleo_loss_raw=None, koleo_loss_weighted=None)
    relational = _row(path)
    assert relational["koleo_status"] == "not-applicable"
    assert relational["koleo_weight"] == ""
    assert relational["koleo_loss_raw"] == ""
    assert relational["koleo_loss_weighted"] == ""

    logger.log(iteration=1, epoch=0, step=1, ssl_method="dino",
               koleo_status="disabled", koleo_weight=0.0,
               koleo_loss_raw=None, koleo_loss_weighted=None)
    disabled = _row(path)
    assert disabled["koleo_status"] == "disabled"
    assert float(disabled["koleo_weight"]) == 0.0
    assert disabled["koleo_loss_raw"] == ""
    assert disabled["koleo_loss_weighted"] == ""

    logger.log(iteration=2, epoch=0, step=2, ssl_method="dino",
               koleo_status="skipped", koleo_weight=0.1,
               koleo_loss_raw=0.0, koleo_loss_weighted=0.0,
               koleo_eligible_samples=0, koleo_skipped_samples=2)
    skipped = _row(path)
    assert skipped["koleo_status"] == "skipped"
    assert float(skipped["koleo_loss_raw"]) == 0.0
    assert float(skipped["koleo_skipped_samples"]) == 2

    logger.log(iteration=3, epoch=0, step=3, ssl_method="dino",
               koleo_status="active", koleo_weight=0.1,
               koleo_loss_raw=-0.5, koleo_loss_weighted=-0.05,
               koleo_eligible_samples=4, koleo_skipped_samples=0)
    active = _row(path)
    assert active["koleo_status"] == "active"
    assert float(active["koleo_loss_raw"]) == pytest.approx(-0.5)  # sign kept
    assert float(active["koleo_loss_weighted"]) == pytest.approx(-0.05)
    assert float(active["koleo_eligible_samples"]) == 4


def test_weighted_components_reconstruct_total(tmp_path):
    logger, path = _logger(tmp_path)
    global_loss, local, patch, koleo = 1.0, 0.2, 0.3, 0.05
    lambda_local, lambda_mask, koleo_weight = 1.5, 1.0, 0.1
    logger.log(
        iteration=0,
        epoch=0,
        step=0,
        loss=global_loss + lambda_local * local + lambda_mask * patch
        + koleo_weight * koleo,
        global_loss=global_loss,
        local_loss=local,
        mask_loss=patch,
        patch_loss_weighted=lambda_mask * patch,
        global_loss_weighted=global_loss,
        local_loss_weighted=lambda_local * local,
        koleo_loss_raw=koleo,
        koleo_loss_weighted=koleo_weight * koleo,
        koleo_weight=koleo_weight,
        koleo_status="active",
        ssl_method="dino",
    )
    row = _row(path)
    total = float(row["loss"])
    reconstruction = (
        float(row["global_loss_weighted"])
        + float(row["local_loss_weighted"])
        + float(row["patch_loss_weighted"])
        + float(row["koleo_loss_weighted"])
    )
    assert reconstruction == pytest.approx(total, rel=1e-6)
    # No lambda_global exists, so the weighted global equals the raw global.
    assert float(row["global_loss_weighted"]) == pytest.approx(
        float(row["global_loss"])
    )


def test_signed_koleo_does_not_create_patch_fraction(tmp_path):
    logger, path = _logger(tmp_path)
    for total in (1.0, 0.0, -0.4):
        logger.log(
            iteration=int(abs(total) * 10),
            epoch=0,
            step=0,
            loss=total,
            patch_loss_weighted=0.2,
            patch_loss_fraction_of_total=None,
            koleo_weight=0.1,
            koleo_status="active",
        )
    frame = _read(path)
    assert (frame["patch_loss_fraction_of_total"] == "").all()

    # Disabled KoLeo keeps the existing positive-fraction meaning.
    logger.log(iteration=99, epoch=0, step=0, loss=1.0, patch_loss_weighted=0.2,
               patch_loss_fraction_of_total=0.2, koleo_weight=0.0,
               koleo_status="disabled")
    assert float(_read(path).iloc[-1]["patch_loss_fraction_of_total"]) == pytest.approx(0.2)


def _metric_args() -> argparse.Namespace:
    return argparse.Namespace(
        lambda_local=1.5, lambda_mask=1.0, log_every_n_steps=1
    )


def test_koleo_instant_metrics_helper_states():
    ssl_config = {"koleo_weight": 0.1}
    zero = torch.zeros(())
    assert pretrain_loop_module._koleo_instant_metrics(ssl_config, False, zero, [2, 2]) == {
        "koleo_status": "disabled",
        "koleo_loss_raw": None,
        "koleo_loss_weighted": None,
    }
    skipped = pretrain_loop_module._koleo_instant_metrics(ssl_config, True, zero, [1, 1])
    assert skipped["koleo_status"] == "skipped"
    assert skipped["koleo_skipped_samples"] == 2
    active = pretrain_loop_module._koleo_instant_metrics(ssl_config, True, torch.tensor(-0.5), [2, 3])
    assert active["koleo_status"] == "active"
    assert active["koleo_eligible_samples"] == 5
    assert active["koleo_loss_weighted"] == pytest.approx(-0.05)


def test_dino_cls_diagnostics_are_finite_and_entropy_safe():
    targets = [
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
    ]
    metrics = pretrain_loop_module._dino_cls_diagnostics(targets, 3)

    assert metrics["cls_assignment_entropy"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["cls_active_prototype_ratio"] == pytest.approx(1.0)
    assert metrics["cls_max_prototype_occupancy"] == pytest.approx(0.5)
    assert metrics["cls_prototype_perplexity"] > 1.0
    for value in metrics.values():
        assert torch.isfinite(torch.tensor(value))

    # Uniform assignments: maximum entropy, finite perplexity, no NaN.
    uniform = [torch.full((2, 4), 0.25), torch.full((2, 4), 0.25)]
    uniform_metrics = pretrain_loop_module._dino_cls_diagnostics(uniform, 4)
    assert uniform_metrics["cls_assignment_entropy"] > metrics["cls_assignment_entropy"]
    assert uniform_metrics["cls_prototype_perplexity"] == pytest.approx(4.0)
    assert torch.isfinite(torch.tensor(uniform_metrics["cls_prototype_perplexity"]))


def test_optional_metric_plots_keep_historical_rows(tmp_path):
    logger, path = _logger(tmp_path)
    legacy_row = {name: "" for name in reporting_module._PRETRAIN_CURRENT_FIELDS}
    legacy_row.update(iteration="0", epoch="0", step="0", loss="1.0", lr="0.001")
    current_row = dict(legacy_row)
    current_row.update(
        iteration="1",
        loss="0.5",
        ssl_method="dino",
        koleo_loss_raw="-0.25",
        koleo_status="active",
        koleo_weight="0.1",
        cls_center_norm="0.5",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(reporting_module._PRETRAIN_CURRENT_FIELDS)
        )
        writer.writeheader()
        writer.writerow(legacy_row)
        writer.writerow(current_row)

    captured = _plot_lines(tmp_path, logger)
    assert (tmp_path / "training_curves_pretrain.pdf").exists()

    # No panel may invent a series for an all-missing column.
    for title, lines in captured.items():
        for label, values in lines.items():
            if label == "__scale__":
                continue
            assert any(value == value for value in values), (title, label)

    # The signed KoLeo contribution is drawn and keeps its negative value. The
    # historical row has no recorded KoLeo loss, so its cell stays missing
    # rather than being replaced by an invented zero.
    koleo = captured["KoLeo Contribution"]
    observed = koleo["KoLeo loss (raw, signed)"]
    assert math.isnan(observed[0])
    assert observed[1] == pytest.approx(-0.25)

    # A single positive total loss keeps the log axis.
    assert captured["Training Losses"]["Total loss"] == pytest.approx([1.0, 0.5])


def test_plot_keeps_legacy_mask_loss_rows(tmp_path):
    """A mixed history must not drop the pre-patch-schema observations."""
    logger, path = _logger(tmp_path)
    legacy_row = {name: "" for name in reporting_module._PRETRAIN_CURRENT_FIELDS}
    legacy_row.update(iteration="0", epoch="0", step="0", loss="1.0", mask_loss="2.0")
    current_row = dict(legacy_row)
    current_row.update(iteration="1", loss="0.5", mask_loss="1.0", patch_loss_raw="1.0")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(reporting_module._PRETRAIN_CURRENT_FIELDS)
        )
        writer.writeheader()
        writer.writerow(legacy_row)
        writer.writerow(current_row)

    captured = _plot_lines(tmp_path, logger)

    # Row 0 exists only in the legacy column, row 1 only in the new one.
    assert captured["Training Losses"]["Patch loss (raw)"] == pytest.approx([2.0, 1.0])


def test_plot_uses_a_signed_axis_for_non_positive_loss(tmp_path):
    logger, path = _logger(tmp_path)
    for index, loss in enumerate(("0.2", "-0.1")):
        logger.log(iteration=index, epoch=0, step=index, loss=loss,
                   ssl_method="dino", koleo_weight=0.1)

    captured = _plot_lines(tmp_path, logger)

    assert captured["Training Losses"]["Total loss"] == pytest.approx([0.2, -0.1])
    # The axis choice is part of the contract: a log axis would hide the sign.
    assert captured["Training Losses"]["__scale__"] == "symlog"


def test_plot_keeps_a_log_axis_for_positive_losses(tmp_path):
    logger, path = _logger(tmp_path)
    for index, loss in enumerate(("0.2", "0.1")):
        logger.log(iteration=index, epoch=0, step=index, loss=loss)

    captured = _plot_lines(tmp_path, logger)

    assert captured["Training Losses"]["__scale__"] == "log"


def test_plot_reads_a_pure_legacy_mask_loss_file(tmp_path):
    """A file predating the patch schema must still draw the mask curve."""
    logger, path = _logger(tmp_path)
    legacy_row = {name: "" for name in reporting_module._PRETRAIN_CURRENT_FIELDS}
    legacy_row.update(iteration="0", epoch="0", step="0", loss="1.0", mask_loss="2.0")
    second = dict(legacy_row, iteration="1", loss="0.5", mask_loss="1.0")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(reporting_module._PRETRAIN_CURRENT_FIELDS)
        )
        writer.writeheader()
        writer.writerow(legacy_row)
        writer.writerow(second)

    captured = _plot_lines(tmp_path, logger)

    assert captured["Training Losses"]["Patch loss (raw)"] == pytest.approx([2.0, 1.0])


def test_plot_draws_dino_assignment_and_koleo_diagnostics(tmp_path):
    logger, path = _logger(tmp_path)
    for index in range(2):
        logger.log(
            iteration=index,
            epoch=0,
            step=index,
            ssl_method="dino",
            dino_centering="sinkhorn-knopp",
            koleo_weight=0.1,
            teacher_temp=0.04 + 0.01 * index,
            global_loss_weighted=7.5,
            local_loss_weighted=11.2,
            patch_loss_weighted=0.02,
            cls_assignment_entropy=7.2 + index,
            cls_prototype_perplexity=2000.0,
            cls_active_prototype_ratio=0.03,
            cls_max_prototype_occupancy=0.03,
            koleo_loss_raw=-0.4,
            koleo_loss_weighted=-0.04,
            koleo_status="active",
            koleo_eligible_samples=64,
            koleo_skipped_samples=0,
        )

    captured = _plot_lines(tmp_path, logger)

    assert captured["Teacher Temperature & Weighted Losses"]["Teacher temperature"] == (
        pytest.approx([0.04, 0.05])
    )
    assert "local_loss_weighted" in captured["Teacher Temperature & Weighted Losses"]
    cls_panel = captured["DINO CLS Target Health (2 x batch pool)"]
    assert cls_panel["Target entropy (nats)"] == pytest.approx([7.2, 8.2])
    assert cls_panel["Marginal perplexity"] == pytest.approx([2000.0, 2000.0])
    assert cls_panel["Active prototype ratio (K denominator)"] == pytest.approx([0.03, 0.03])
    koleo = captured["KoLeo Contribution"]
    assert koleo["KoLeo loss (raw, signed)"] == pytest.approx([-0.4, -0.4])
    assert koleo["KoLeo loss (weighted, signed)"] == pytest.approx([-0.04, -0.04])
    # Patch-only diagnostics stay in their own panel, absent here.
    assert "iBOT Patch Target Health" not in captured


def test_feature_variation_metrics_survive_dino(tmp_path):
    logger, path = _logger(tmp_path)
    features = torch.randn(8, 6)
    expected_std = float(features.std(dim=0).mean())
    expected_norm = float(features.norm(dim=-1).mean())
    logger.log(
        iteration=0,
        epoch=0,
        step=0,
        ssl_method="dino",
        feature_std=expected_std,
        embedding_norm_mean=expected_norm,
        cls_token_norm_mean=expected_norm,
    )
    row = _row(path)
    assert float(row["feature_std"]) == pytest.approx(expected_std)
    assert float(row["embedding_norm_mean"]) == pytest.approx(expected_norm)
    assert float(row["cls_token_norm_mean"]) == pytest.approx(expected_norm)


def test_singleton_batch_reports_feature_std_as_missing():
    """A batch standard deviation needs two samples; one sample is N/A, not NaN."""
    from otuformer.training.model import OTUFormerEncoder

    encoder = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224",
        out_dim=17,
        pretrained=False,
        img_size=64,
    )
    metrics = pretrain_loop_module._compute_instant_metrics(
        encoder, torch.zeros(1, 3, 64, 64), torch.device("cpu")
    )

    assert metrics["feature_std"] is None
    # The per-sample norms stay defined and finite for one sample.
    assert math.isfinite(metrics["embedding_norm_mean"])
    assert math.isfinite(metrics["cls_token_norm_mean"])

    batched = pretrain_loop_module._compute_instant_metrics(
        encoder, torch.zeros(2, 3, 64, 64), torch.device("cpu")
    )
    assert math.isfinite(batched["feature_std"])


def test_fifo_metrics_resume_current_43_column_log(tmp_path):
    path = tmp_path / "instant.csv"
    old_header = (
        "iteration epoch step loss global_loss local_loss mask_loss lr teacher_temp "
        "grad_norm feature_std embedding_norm_mean cls_token_norm_mean teacher_center_norm "
        "cosine_similarity patch_loss_raw patch_loss_weighted patch_loss_fraction_of_total "
        "requested_mask_ratio selected_patch_ratio actual_mask_ratio_mean actual_mask_ratio_min "
        "actual_mask_ratio_max prototype_perplexity active_prototype_ratio assignment_entropy "
        "max_prototype_occupancy patch_center_norm ssl_method dino_centering koleo_weight "
        "cls_center_norm global_loss_weighted local_loss_weighted cls_assignment_entropy "
        "cls_prototype_perplexity cls_active_prototype_ratio cls_max_prototype_occupancy "
        "koleo_loss_raw koleo_loss_weighted koleo_status koleo_eligible_samples koleo_skipped_samples"
    ).split()
    assert len(old_header) == 43
    _write(path, old_header, [["1"] * len(old_header)])
    pretrain_loop_module.InstantMetricsLogger(path, mode="pretrain").log(
        iteration=2, relational_queue_capacity=128
    )
    rows = list(csv.DictReader(path.open(newline="")))
    assert all(rows[0][name] == "1" for name in old_header)
    assert rows[0]["relational_queue_capacity"] == ""
    assert rows[1]["relational_queue_capacity"] == "128"
    assert rows[1]["loss"] == ""
    assert rows[1]["koleo_status"] == ""
    # The widened header is exactly the current schema, in order.
    assert list(rows[0]) == list(reporting_module._PRETRAIN_CURRENT_FIELDS)


def test_fifo_metrics_unknown_and_duplicate_headers_stay_rejected(tmp_path):
    current = list(reporting_module._PRETRAIN_CURRENT_FIELDS)
    duplicate = tmp_path / "duplicate.csv"
    _write(duplicate, current + [current[0]], [["1"] * (len(current) + 1)])
    original = duplicate.read_bytes()
    with pytest.raises(ValueError, match="duplicate columns"):
        pretrain_loop_module.InstantMetricsLogger(duplicate, mode="pretrain")
    assert duplicate.read_bytes() == original

    unknown = tmp_path / "unknown.csv"
    header = list(current)
    header[5] = "mystery_loss"
    _write(unknown, header, [["1"] * len(header)])
    original = unknown.read_bytes()
    with pytest.raises(ValueError, match="Unrecognized instant-metrics schema"):
        pretrain_loop_module.InstantMetricsLogger(unknown, mode="pretrain")
    assert unknown.read_bytes() == original


def test_fifo_queue_metrics_leave_unavailable_cells_blank():
    metrics = reporting_module._relational_queue_metrics(
        capacity=4,
        occupancy=2,
        eligible=0,
        candidate_count=4,
        ages=(),
        refreshed=0,
        evicted=0,
        teacher_stats=None,
        student_stats=None,
    )
    assert metrics["relational_queue_capacity"] == 4
    assert metrics["relational_queue_occupancy"] == 2
    assert metrics["relational_queue_eligible"] == 0
    assert metrics["relational_candidate_count"] == 4
    assert metrics["relational_history_age_mean"] is None
    assert metrics["relational_history_age_max"] is None
    for name in reporting_module._PRETRAIN_RELATIONAL_QUEUE_FIELDS:
        if name.startswith("relational_teacher") or name.startswith(
            "relational_student"
        ):
            assert metrics[name] is None, name


def test_fifo_queue_metrics_report_ages_and_branch_statistics():
    teacher = {
        "self_mass": torch.tensor(0.5),
        "other_current_mass": torch.tensor(0.25),
        "history_mass": torch.tensor(0.25),
        "entropy": torch.tensor(1.0),
        "nonself_effective": torch.tensor(1.5),
    }
    student = [
        {
            "self_mass": torch.tensor(0.4),
            "other_current_mass": torch.tensor(0.4),
            "history_mass": torch.tensor(0.2),
        },
        {
            "self_mass": torch.tensor(0.3),
            "other_current_mass": torch.tensor(0.5),
            "history_mass": torch.tensor(0.2),
        },
    ]
    metrics = reporting_module._relational_queue_metrics(
        capacity=8,
        occupancy=3,
        eligible=2,
        candidate_count=6,
        ages=(2, 1),
        refreshed=2,
        evicted=1,
        teacher_stats=teacher,
        student_stats=student,
    )
    assert metrics["relational_history_age_mean"] == 1.5
    assert metrics["relational_history_age_max"] == 2
    assert metrics["relational_queue_refreshed"] == 2
    assert metrics["relational_queue_evicted"] == 1
    assert metrics["relational_teacher_self_mass_global"] == pytest.approx(0.5)
    assert metrics["relational_teacher_self_mass_local"] == pytest.approx(0.5)
    assert metrics["relational_student_self_mass_global"] == pytest.approx(0.4)
    assert metrics["relational_student_self_mass_local"] == pytest.approx(0.3)
