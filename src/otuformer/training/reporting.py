"""Instant/epoch metrics CSV schemas, loggers, plots and evaluation."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple
import numpy as np
import torch
import torch.nn as nn
from otuformer.embedding.evaluator import (
    compute_clustering_metrics,
    compute_knn_accuracy,
    compute_linear_probing_metrics,
    compute_map,
    compute_recall_at_k,
    run_umap,
)
from otuformer.embedding.sampling import sorted_sample_indices
from .dense_objective import BRANCH_COUNT_FIELDS
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
    resolve_dataset_image_paths,
)
from otuformer.training.model import (
    PATCH_TARGET_LAYERS,
    ArcFaceEmbeddingHead,
    OTUFormerEncoder,
    PatchObjective,
    native_register_count,
)

_PRETRAIN_BASE_FIELDS = ["iteration", "epoch", "step"]


_PRETRAIN_LEGACY_METRIC_FIELDS = [
    "loss",
    "global_loss",
    "local_loss",
    "mask_loss",
    "lr",
    "teacher_temp",
    "grad_norm",
    "feature_std",
    "embedding_norm_mean",
    "cls_token_norm_mean",
    "teacher_center_norm",
    "cosine_similarity",
]


_PRETRAIN_PATCH_FIELDS = [
    "patch_loss_raw",
    "patch_loss_weighted",
    "patch_loss_fraction_of_total",
    "requested_mask_ratio",
    "selected_patch_ratio",
    "actual_mask_ratio_mean",
    "actual_mask_ratio_min",
    "actual_mask_ratio_max",
    "prototype_perplexity",
    "active_prototype_ratio",
    "assignment_entropy",
    "max_prototype_occupancy",
    "patch_center_norm",
]


_PRETRAIN_LEGACY_FIELDS = _PRETRAIN_BASE_FIELDS + _PRETRAIN_LEGACY_METRIC_FIELDS


_PRETRAIN_V070_FIELDS = _PRETRAIN_LEGACY_FIELDS + _PRETRAIN_PATCH_FIELDS


_PRETRAIN_SSL_SETTING_FIELDS = ["ssl_method", "dino_centering", "koleo_weight"]


_PRETRAIN_SSL_METRIC_FIELDS = [
    "cls_center_norm",
    "global_loss_weighted",
    "local_loss_weighted",
    "cls_assignment_entropy",
    "cls_prototype_perplexity",
    "cls_active_prototype_ratio",
    "cls_max_prototype_occupancy",
    "koleo_loss_raw",
    "koleo_loss_weighted",
    "koleo_status",
    "koleo_eligible_samples",
    "koleo_skipped_samples",
]


# A literal snapshot of the pre-FIFO 43-column instant schema. It is written
# out rather than derived so a simultaneous schema mistake cannot silently widen
# the recognized predecessor and rewrite an unrelated file.
_PRETRAIN_PRE_FIFO_FIELDS = [
    "iteration",
    "epoch",
    "step",
    "loss",
    "global_loss",
    "local_loss",
    "mask_loss",
    "lr",
    "teacher_temp",
    "grad_norm",
    "feature_std",
    "embedding_norm_mean",
    "cls_token_norm_mean",
    "teacher_center_norm",
    "cosine_similarity",
    "patch_loss_raw",
    "patch_loss_weighted",
    "patch_loss_fraction_of_total",
    "requested_mask_ratio",
    "selected_patch_ratio",
    "actual_mask_ratio_mean",
    "actual_mask_ratio_min",
    "actual_mask_ratio_max",
    "prototype_perplexity",
    "active_prototype_ratio",
    "assignment_entropy",
    "max_prototype_occupancy",
    "patch_center_norm",
    "ssl_method",
    "dino_centering",
    "koleo_weight",
    "cls_center_norm",
    "global_loss_weighted",
    "local_loss_weighted",
    "cls_assignment_entropy",
    "cls_prototype_perplexity",
    "cls_active_prototype_ratio",
    "cls_max_prototype_occupancy",
    "koleo_loss_raw",
    "koleo_loss_weighted",
    "koleo_status",
    "koleo_eligible_samples",
    "koleo_skipped_samples",
]

_RELATIONAL_TEACHER_STAT_NAMES = (
    "self_mass",
    "other_current_mass",
    "history_mass",
    "entropy",
    "nonself_effective",
)

_RELATIONAL_STUDENT_STAT_NAMES = (
    "self_mass",
    "other_current_mass",
    "history_mass",
)

_RELATIONAL_BRANCH_SUFFIXES = ("global", "local")


_PRETRAIN_RELATIONAL_QUEUE_FIELDS = (
    [
        "relational_queue_capacity",
        "relational_queue_occupancy",
        "relational_queue_eligible",
        "relational_candidate_count",
        "relational_history_age_mean",
        "relational_history_age_max",
        "relational_queue_refreshed",
        "relational_queue_evicted",
    ]
    + [
        f"relational_teacher_{name}_{suffix}"
        for name in _RELATIONAL_TEACHER_STAT_NAMES
        for suffix in _RELATIONAL_BRANCH_SUFFIXES
    ]
    + [
        f"relational_student_{name}_{suffix}"
        for name in _RELATIONAL_STUDENT_STAT_NAMES
        for suffix in _RELATIONAL_BRANCH_SUFFIXES
    ]
)


if list(_PRETRAIN_PRE_FIFO_FIELDS) != list(
    _PRETRAIN_V070_FIELDS
    + _PRETRAIN_SSL_SETTING_FIELDS
    + _PRETRAIN_SSL_METRIC_FIELDS
):  # pragma: no cover - import-time schema guard
    raise RuntimeError(
        "The pre-FIFO instant schema snapshot no longer matches the composed "
        "pretrain fields."
    )


_PRETRAIN_CURRENT_FIELDS = tuple(_PRETRAIN_PRE_FIFO_FIELDS) + tuple(
    _PRETRAIN_RELATIONAL_QUEUE_FIELDS
)


# Dense objectives append their own fields only for an enabled run; a disabled
# run keeps the previous header exactly.
_PRETRAIN_DENSE_FIELDS = [
    "dense_coordinate_loss_raw",
    "dense_coordinate_loss_weighted",
    "dense_coordinate_status",
    "dense_coordinate_points",
    "dense_coordinate_samples",
    "dense_coordinate_pairs",
    "dense_min_norm",
    "dense_min_rho",
    "dense_ranking_loss_raw",
    "dense_ranking_loss_weighted",
    "dense_ranking_status",
    "dense_ranking_points",
    "dense_ranking_samples",
    "dense_ranking_pairs",
    "dense_ranking_references",
    "dense_ranking_reference_count",
    "dense_ranking_shortage_samples",
    "dense_ranking_min_probability",
    "dense_coordinate_eligible",
    # The ranking objective's own projected-space diagnostics.
    "dense_ranking_min_norm",
    "dense_ranking_min_rho",
    "dense_ranking_projection_min_norm",
    "dense_ranking_reference_min",
    "dense_ranking_reference_population_min",
    "dense_ranking_unavailable_pairs",
]


_PRETRAIN_DENSE_FIELDS += [
    f"dense_{objective}_{branch}_{field}"
    for objective in ("coordinate", "ranking")
    for branch in ("global", "local")
    for field in BRANCH_COUNT_FIELDS
]


#: The dense header written before the ranking projected-space diagnostics were
#: added (86 columns: the eligibility column, no ranking norm/rho). Existing runs
#: with this header must keep resuming, so it stays a recognized predecessor and
#: migrates by filling the two new columns empty.
_PRETRAIN_DENSE_86_FIELDS = tuple(_PRETRAIN_CURRENT_FIELDS) + (
    "dense_coordinate_loss_raw",
    "dense_coordinate_loss_weighted",
    "dense_coordinate_status",
    "dense_coordinate_points",
    "dense_coordinate_samples",
    "dense_coordinate_pairs",
    "dense_min_norm",
    "dense_min_rho",
    "dense_ranking_loss_raw",
    "dense_ranking_loss_weighted",
    "dense_ranking_status",
    "dense_ranking_points",
    "dense_ranking_samples",
    "dense_ranking_pairs",
    "dense_ranking_references",
    "dense_ranking_reference_count",
    "dense_ranking_shortage_samples",
    "dense_ranking_min_probability",
    "dense_coordinate_eligible",
)


#: The dense header written when the ranking projected-space diagnostics were
#: added (88 columns). Every header a previous revision actually wrote stays a
#: supported predecessor, otherwise a legal earlier log blocks its own resume.
_PRETRAIN_DENSE_88_FIELDS = tuple(_PRETRAIN_CURRENT_FIELDS) + (
    "dense_coordinate_loss_raw",
    "dense_coordinate_loss_weighted",
    "dense_coordinate_status",
    "dense_coordinate_points",
    "dense_coordinate_samples",
    "dense_coordinate_pairs",
    "dense_min_norm",
    "dense_min_rho",
    "dense_ranking_loss_raw",
    "dense_ranking_loss_weighted",
    "dense_ranking_status",
    "dense_ranking_points",
    "dense_ranking_samples",
    "dense_ranking_pairs",
    "dense_ranking_references",
    "dense_ranking_reference_count",
    "dense_ranking_shortage_samples",
    "dense_ranking_min_probability",
    "dense_coordinate_eligible",
    "dense_ranking_min_norm",
    "dense_ranking_min_rho",
)


#: The dense header written by the first delivery (before the ranking shortage
#: and eligibility columns were added). It stays a recognized predecessor so an
#: enabled run can still migrate its own earlier rows in place.
_PRETRAIN_DENSE_TASK4_FIELDS = tuple(_PRETRAIN_CURRENT_FIELDS) + (
    "dense_coordinate_loss_raw",
    "dense_coordinate_loss_weighted",
    "dense_coordinate_status",
    "dense_coordinate_points",
    "dense_coordinate_samples",
    "dense_coordinate_pairs",
    "dense_min_norm",
    "dense_min_rho",
    "dense_ranking_loss_raw",
    "dense_ranking_loss_weighted",
    "dense_ranking_status",
    "dense_ranking_points",
    "dense_ranking_samples",
    "dense_ranking_pairs",
    "dense_ranking_references",
    "dense_ranking_min_probability",
)


# Keep predecessor definitions independent of future additions to the live list.
_PRETRAIN_DENSE_92_FIELDS = _PRETRAIN_DENSE_88_FIELDS + (
    "dense_ranking_projection_min_norm",
    "dense_ranking_reference_min",
    "dense_ranking_reference_population_min",
    "dense_ranking_unavailable_pairs",
)


_PRETRAIN_DENSE_CURRENT_FIELDS = tuple(_PRETRAIN_CURRENT_FIELDS) + tuple(
    _PRETRAIN_DENSE_FIELDS
)


_PRETRAIN_PLOT_NUMERIC_FIELDS = (
    _PRETRAIN_LEGACY_METRIC_FIELDS
    + _PRETRAIN_PATCH_FIELDS
    + _PRETRAIN_SSL_METRIC_FIELDS
)


_FINETUNE_FIELDS = [
    "loss",
    "lr",
    "grad_norm",
    "feature_std",
    "embedding_norm_mean",
    "cls_token_norm_mean",
]


@dataclass
class InstantMetricsLogger:
    path: Path
    mode: str = "pretrain"
    dense_enabled: bool = False

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.mode == "pretrain":
            self.fieldnames = list(
                _PRETRAIN_DENSE_CURRENT_FIELDS
                if self.dense_enabled
                else _PRETRAIN_CURRENT_FIELDS
            )
        else:
            self.fieldnames = list(_PRETRAIN_BASE_FIELDS) + list(_FINETUNE_FIELDS)
        if self.path.exists() and self.path.stat().st_size > 0:
            self._validate_or_migrate_existing()
            return
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def _read_existing_csv(self) -> tuple[list[str], list[list[str]]]:
        try:
            with self.path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                try:
                    header = next(reader)
                except StopIteration:
                    raise ValueError(
                        f"instant-metrics schema in {self.path} is empty; refusing "
                        "to append."
                    ) from None
                rows: list[list[str]] = []
                for line_number, row in enumerate(reader, start=2):
                    if len(row) != len(header):
                        raise ValueError(
                            f"instant-metrics schema in {self.path} is malformed: "
                            f"row {line_number} has {len(row)} fields but the header "
                            f"has {len(header)}."
                        )
                    rows.append(row)
        except csv.Error as exc:
            raise ValueError(
                f"instant-metrics schema in {self.path} is malformed CSV: {exc}"
            ) from exc
        return header, rows

    def _validate_or_migrate_existing(self) -> None:
        header, rows = self._read_existing_csv()
        if len(set(header)) != len(header):
            raise ValueError(
                f"instant-metrics schema in {self.path} has duplicate columns; "
                "refusing to append."
            )
        if header == self.fieldnames:
            return
        recognized = (
            list(_PRETRAIN_LEGACY_FIELDS),
            list(_PRETRAIN_V070_FIELDS),
            list(_PRETRAIN_PRE_FIFO_FIELDS),
            list(_PRETRAIN_CURRENT_FIELDS),
        )
        if self.dense_enabled:
            recognized += (
                list(_PRETRAIN_DENSE_TASK4_FIELDS),
                list(_PRETRAIN_DENSE_86_FIELDS),
                list(_PRETRAIN_DENSE_88_FIELDS),
                list(_PRETRAIN_DENSE_92_FIELDS),
                list(_PRETRAIN_DENSE_CURRENT_FIELDS),
            )
        if self.mode == "pretrain" and header in recognized:
            self._migrate_pretrain_schema(header, rows)
            return
        raise ValueError(
            f"Unrecognized instant-metrics schema in {self.path}: {header}. "
            "Refusing to append. Move or remove the file to start a new log."
        )

    def _migrate_pretrain_schema(
        self, header: list[str], rows: list[list[str]]
    ) -> None:
        """Atomically rewrite a recognized prior header to the current schema."""
        import os
        import tempfile

        index = {name: position for position, name in enumerate(header)}
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                newline="",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=self.path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                writer = csv.writer(handle)
                writer.writerow(self.fieldnames)
                for row in rows:
                    writer.writerow(
                        [
                            row[index[name]] if name in index else ""
                            for name in self.fieldnames
                        ]
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
            temp_path = None
        finally:
            # Only this call's own temporary file may be removed: a wildcard
            # sweep would delete an unrelated sidecar that existed beforehand.
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def log(self, **kwargs: Any) -> None:
        row = {k: kwargs.get(k, "") for k in self.fieldnames}
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

    @staticmethod
    def _plottable_series(df_instant, name: str):
        """One numeric column with blanks as missing, or ``None`` if absent.

        An all-missing column returns ``None`` so callers skip it instead of
        drawing an empty legend entry with no data.
        """
        if name not in df_instant.columns:
            return None
        values = df_instant[name]
        return None if values.isna().all() else values

    def _dense_diagnostics_figure(self, df_instant, plt):
        """One optional page of dense-objective diagnostics, or ``None``.

        Only columns that carry at least one measured value are drawn: an
        unavailable statistic is missing, never a fabricated zero. The panels
        keep the CSV column meanings so a reader can move between the two.
        """
        if not self.dense_enabled:
            return None
        import pandas as pd

        def _objective_enabled(status_column: str, *value_columns: str) -> bool:
            """An objective ran if it was never logged ``disabled``."""
            status = df_instant.get(status_column)
            if status is not None:
                tokens = set(status.astype("string").dropna().unique())
                if tokens and tokens <= {"disabled", ""}:
                    return False
            return any(
                self._plottable_series(df_instant, name) is not None
                for name in value_columns
            )

        coordinate_enabled = _objective_enabled(
            "dense_coordinate_status",
            "dense_coordinate_loss_raw",
            "dense_coordinate_points",
        )
        ranking_enabled = _objective_enabled(
            "dense_ranking_status",
            "dense_ranking_loss_raw",
            "dense_ranking_points",
        )

        def series(name: str):
            return self._plottable_series(df_instant, name)

        dense_names = [
            "dense_coordinate_loss_raw",
            "dense_coordinate_loss_weighted",
            "dense_ranking_loss_raw",
            "dense_ranking_loss_weighted",
            "dense_coordinate_points",
            "dense_coordinate_samples",
            "dense_coordinate_pairs",
            "dense_coordinate_eligible",
            "dense_ranking_points",
            "dense_ranking_samples",
            "dense_ranking_pairs",
            "dense_ranking_references",
            "dense_ranking_reference_count",
            "dense_ranking_shortage_samples",
            "dense_min_norm",
            "dense_min_rho",
            "dense_ranking_min_probability",
            "dense_ranking_min_norm",
            "dense_ranking_min_rho",
            "dense_ranking_projection_min_norm",
            "dense_ranking_reference_min",
            "dense_ranking_reference_population_min",
            "dense_ranking_unavailable_pairs",
        ]
        for name in dense_names:
            if name in df_instant.columns:
                df_instant[name] = pd.to_numeric(df_instant[name], errors="coerce")
        if not any(series(name) is not None for name in dense_names):
            return None

        iterations = df_instant["iteration"]
        fig, axes = plt.subplots(3, 4, figsize=(26, 15))
        axes = axes.flatten()
        slot = 0

        def panel(title, ylabel):
            nonlocal slot
            ax = axes[slot]
            slot += 1
            ax.set_title(title)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            return ax

        def close_if_empty(ax, drew):
            """Turn a panel off when no series was drawn into it.

            An objective that never ran must not leave a titled, empty axis in
            the report; the caller's own ``drew`` flag decides.
            """
            if not drew:
                ax.axis("off")
                ax.set_title("")
            return drew

        def draw(ax, name, label, color=None, twin=None):
            values = series(name)
            if values is None:
                return False
            target = ax if twin is None else twin
            target.plot(
                iterations,
                values,
                label=label,
                **({"color": color} if color else {}),
            )
            return True

        def twinx_axis(ax, color, label):
            twin = ax.twinx()
            twin.set_ylabel(label, color=color)
            twin.tick_params(axis="y", labelcolor=color)
            return twin

        def legend(ax, *twins):
            """Legend across the panel and its twin axes; skip an empty one."""
            handles, labels = ax.get_legend_handles_labels()
            for twin in twins:
                twin_handles, twin_labels = twin.get_legend_handles_labels()
                handles += twin_handles
                labels += twin_labels
            if labels:
                ax.legend(handles, labels, fontsize=8)

        # 1. Raw losses beside the loss the existing objectives alone produce.
        #    ``loss`` in the CSV already contains the weighted dense terms, so it
        #    must not be labelled "existing": subtracting them recovers the
        #    existing-only value exactly (total = existing + weighted dense).
        ax = panel("Dense raw losses vs existing objectives", "Raw loss")
        drew = draw(ax, "dense_coordinate_loss_raw", "Coordinate raw", "tab:blue")
        drew = draw(ax, "dense_ranking_loss_raw", "Ranking raw", "tab:orange") or drew
        twin = twinx_axis(ax, "tab:gray", "Existing objectives loss")
        # Only when a real total-loss column exists: without it, subtracting the
        # dense terms from nothing would fabricate a negative "existing" curve.
        existing_only = df_instant.get("loss")
        if existing_only is not None:
            existing_only = pd.to_numeric(existing_only, errors="coerce")
            if existing_only.isna().all():
                existing_only = None
        if existing_only is not None:
            for name in (
                "dense_coordinate_loss_weighted",
                "dense_ranking_loss_weighted",
            ):
                column = df_instant.get(name)
                if column is not None:
                    filled = pd.to_numeric(column, errors="coerce").fillna(0.0)
                    existing_only = existing_only - filled
        if existing_only is not None:
            twin.plot(
                iterations,
                existing_only,
                label="Existing objectives (total minus dense)",
                color="tab:gray",
            )
            drew = True
        close_if_empty(ax, drew)
        if drew:
            legend(ax, twin)

        # 2. Weighted contributions of the two new terms.
        ax = panel("Dense weighted contribution", "Weighted loss")
        drew = draw(
            ax, "dense_coordinate_loss_weighted", "Coordinate weighted", "tab:blue"
        )
        drew = (
            draw(ax, "dense_ranking_loss_weighted", "Ranking weighted", "tab:orange")
            or drew
        )
        close_if_empty(ax, drew)
        if drew:
            legend(ax)

        # 3. Query populations, pre-cap eligible and post-cap selected.
        ax = panel("Dense query population", "Queries")
        drew = draw(ax, "dense_coordinate_eligible", "Coordinate eligible", "tab:blue")
        drew = (
            draw(ax, "dense_coordinate_points", "Coordinate selected", "tab:cyan")
            or drew
        )
        drew = draw(ax, "dense_ranking_points", "Ranking selected", "tab:orange") or drew
        close_if_empty(ax, drew)
        if drew:
            legend(ax)

        # 4. Contributing denominators, so a skip is visible as a dip to 0.
        ax = panel("Dense contributing denominators", "Count")
        drew = draw(
            ax, "dense_coordinate_samples", "Coordinate samples", "tab:blue"
        )
        drew = (
            draw(ax, "dense_coordinate_pairs", "Coordinate pairs", "tab:cyan") or drew
        )
        drew = draw(ax, "dense_ranking_samples", "Ranking samples", "tab:orange") or drew
        drew = draw(ax, "dense_ranking_pairs", "Ranking pairs", "tab:red") or drew
        close_if_empty(ax, drew)
        if drew:
            legend(ax)

        for branch in ("global", "local"):
            ax = panel(f"Dense {branch} denominators", "Count")
            drew = False
            for objective in ("coordinate", "ranking"):
                for field in ("attempted_samples", "contributing_samples", "contributing_pairs"):
                    drew = draw(
                        ax, f"dense_{objective}_{branch}_{field}",
                        f"{objective.title()} {field.replace('_', ' ')}",
                    ) or drew
            close_if_empty(ax, drew)
            if drew:
                legend(ax)

        # 5. Reference population and the count actually used. Ranking-only: a
        #    Coordinate-only run must not show an empty reference panel.
        references = None
        if ranking_enabled:
            references = panel("Ranking references", "References")
            drew = draw(
                references, "dense_ranking_references", "Pool population (max)", "tab:purple"
            )
            drew = (
                draw(
                    references,
                    "dense_ranking_reference_count",
                    "Reference count (max)",
                    "tab:green",
                )
                or drew
            )
            close_if_empty(references, drew)
            if drew:
                legend(references)

        # 6. Shortages and unavailable populations. ``disabled`` is not a skip:
        #    an objective that was switched off must not accumulate a skip count.
        if ranking_enabled or coordinate_enabled:
            ax = panel("Ranking shortages and skips", "Samples / pairs")
            drew = draw(
                ax, "dense_ranking_shortage_samples", "Shortage samples", "tab:red"
            )
            for name, label in (
                ("dense_coordinate_status", "Coordinate skipped"),
                ("dense_ranking_status", "Ranking skipped"),
            ):
                if name not in df_instant.columns:
                    continue
                mask = df_instant[name].astype("string") == "skipped"
                if not bool(mask.any()):
                    continue
                ax.plot(
                    iterations,
                    mask.astype(int).cumsum(),
                    label=label + " (cumulative)",
                    linestyle=":",
                )
                drew = True
            close_if_empty(ax, drew)
            if drew:
                legend(ax)

        # 7-9. Numerical guards in separate panels: their scales differ.
        ax = panel("Min participating norm", "Norm (log)")
        drew = draw(ax, "dense_min_norm", "Min norm", "tab:blue")
        close_if_empty(ax, drew)
        if drew:
            ax.set_yscale("log")
            legend(ax)

        ax = panel("Min interpolation rho", "rho")
        drew = draw(ax, "dense_min_rho", "Min rho", "tab:green")
        close_if_empty(ax, drew)
        if drew:
            all_rho = series("dense_min_rho")
            if all_rho is not None and (all_rho <= 0).any():
                ax.set_ylim(bottom=0.0)
            legend(ax)

        ax = panel("Ranking projected space (norm / rho)", "Value")
        drew = draw(
            ax,
            "dense_ranking_projection_min_norm",
            "Ranking pre-L2 projection norm",
            "tab:purple",
        )
        drew = (
            draw(ax, "dense_ranking_min_norm", "Ranking interpolated min norm", "tab:blue")
            or drew
        )
        drew = (
            draw(ax, "dense_ranking_min_rho", "Ranking min rho", "tab:green") or drew
        )
        close_if_empty(ax, drew)
        if drew:
            legend(ax)

        ax = panel("Min student permutation probability", "P (log)")
        drew = draw(ax, "dense_ranking_min_probability", "Min P", "tab:orange")
        close_if_empty(ax, drew)
        if drew:
            ax.set_yscale("log")
            legend(ax)

        for index in range(slot, len(axes)):
            axes[index].axis("off")
        fig.suptitle("Dense diagnostics (Coordinate and NeCo-style ranking)")
        fig.tight_layout()
        return fig

    def plot(self, out_dir: Path, metrics_path: Path | None = None) -> None:
        if not self.path.exists():
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import pandas as pd
        except ImportError:
            print("[Warning] matplotlib/pandas missing -> skip training curves.")
            return

        df_instant = pd.read_csv(self.path)
        if df_instant.empty:
            return
        # Optional metric cells can be blank (historical rows, N/A under the
        # other method). They become missing values, never fabricated zeros, and
        # an all-missing series is skipped instead of drawn.
        for name in _PRETRAIN_PLOT_NUMERIC_FIELDS:
            if name in df_instant.columns:
                df_instant[name] = pd.to_numeric(df_instant[name], errors="coerce")

        def series(name: str):
            return self._plottable_series(df_instant, name)

        df_metrics = None
        if metrics_path is not None and metrics_path.exists():
            df_metrics = pd.read_csv(metrics_path)
            if "epoch" in df_metrics.columns:
                df_metrics = df_metrics[df_metrics["epoch"].notna()]

        # The panel count grew with the DINO/KoLeo diagnostics, so the grid is
        # sized for them; unused cells are switched off at the end.
        fig, axes = plt.subplots(4, 4, figsize=(26, 20))
        axes = axes.flatten()
        plot_idx = 0

        # Mixed history: rows written before the patch schema have no
        # ``patch_loss_raw`` and only ``mask_loss``. Fall back per row so earlier
        # observations are not dropped once a later row fills the new column, and
        # keep working from a pure legacy file that has no raw column at all.
        patch_series = self._plottable_series(df_instant, "patch_loss_raw")
        legacy_mask = self._plottable_series(df_instant, "mask_loss")
        if patch_series is not None:
            patch_series = (
                patch_series.fillna(legacy_mask)
                if legacy_mask is not None
                else patch_series
            )
        else:
            patch_series = legacy_mask
        if patch_series is not None and patch_series.isna().all():
            patch_series = None
        ax = axes[plot_idx]
        loss_lines = {
            "Total loss": self._plottable_series(df_instant, "loss"),
            "Global loss": self._plottable_series(df_instant, "global_loss"),
            "Local loss": self._plottable_series(df_instant, "local_loss"),
            "Patch loss (raw)": patch_series,
        }
        for label, values in loss_lines.items():
            if values is None:
                continue
            ax.plot(
                df_instant["iteration"],
                values,
                label=label,
                linewidth=1.5 if label == "Total loss" else 1.0,
                alpha=1.0 if label == "Total loss" else 0.8,
            )
        # KoLeo is signed, so a total loss can legitimately reach non-positive
        # values. A log axis cannot show those; keep it only when every drawn
        # value is strictly positive.
        drawn = [
            value
            for values in loss_lines.values()
            if values is not None
            for value in values.dropna()
        ]
        if drawn and all(value > 0 for value in drawn):
            ax.set_yscale("log")
            ax.set_ylabel("Loss (log scale)")
        elif drawn:
            # symlog keeps small positive values readable without clipping or
            # hiding the sign of a negative contribution.
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_ylabel("Loss (symlog scale; signed values kept)")
        ax.set_xlabel("Iteration")
        ax.set_title("Training Losses")
        if drawn:
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        plot_idx += 1

        ax = axes[plot_idx]
        has_cos = series("cosine_similarity") is not None
        has_grad = series("grad_norm") is not None
        second_panel_has_data = False
        if has_cos and has_grad:
            ax_twin = ax.twinx()
            ax.plot(
                df_instant["iteration"],
                df_instant["cosine_similarity"],
                label="Cosine Similarity",
                color="tab:blue",
                linewidth=1.5,
            )
            ax.set_ylabel("Cosine Similarity", color="tab:blue")
            ax.tick_params(axis="y", labelcolor="tab:blue")
            ax_twin.plot(
                df_instant["iteration"],
                df_instant["grad_norm"],
                label="Grad Norm",
                color="tab:orange",
                linewidth=1.5,
                alpha=0.8,
            )
            ax_twin.set_ylabel("Gradient Norm", color="tab:orange")
            ax_twin.tick_params(axis="y", labelcolor="tab:orange")
            ax.set_title("Cosine Similarity & Grad Norm")
            second_panel_has_data = True
        elif has_cos:
            ax.plot(
                df_instant["iteration"],
                df_instant["cosine_similarity"],
                label="Cosine Similarity",
                color="tab:blue",
            )
            ax.set_title("Cosine Similarity")
            second_panel_has_data = True
        elif has_grad:
            ax.plot(
                df_instant["iteration"],
                df_instant["grad_norm"],
                label="Grad Norm",
                color="tab:orange",
            )
            ax.set_title("Gradient Norm")
            second_panel_has_data = True
        if second_panel_has_data:
            ax.set_xlabel("Iteration")
            if has_cos and has_grad:
                lines_l, labels_l = ax.get_legend_handles_labels()
                lines_r, labels_r = ax_twin.get_legend_handles_labels()
                ax.legend(
                    lines_l + lines_r,
                    labels_l + labels_r,
                    fontsize=8,
                    loc="best",
                )
            else:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        else:
            ax.axis("off")

        ax = axes[plot_idx]
        has_left = any(
            series(c) is not None for c in ["feature_std", "embedding_norm_mean"]
        )
        has_cls = series("cls_token_norm_mean") is not None
        if has_left:
            if series("feature_std") is not None:
                ax.plot(
                    df_instant["iteration"],
                    df_instant["feature_std"],
                    label="Feature Std",
                    color="tab:green",
                )
            if series("embedding_norm_mean") is not None:
                ax.plot(
                    df_instant["iteration"],
                    df_instant["embedding_norm_mean"],
                    label="Embedding Norm",
                    color="tab:cyan",
                )
        if has_cls:
            ax_twin = ax.twinx()
            ax_twin.plot(
                df_instant["iteration"],
                df_instant["cls_token_norm_mean"],
                label="CLS Token Norm",
                color="tab:purple",
                alpha=0.8,
            )
            ax_twin.set_ylabel("CLS Token Norm", color="tab:purple")
            ax_twin.tick_params(axis="y", labelcolor="tab:purple")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Feature Std / Embedding Norm")
        ax.set_title("Feature Statistics")
        if has_cls or has_left:
            lines_l, labels_l = ax.get_legend_handles_labels()
            lines_r, labels_r = (
                ax_twin.get_legend_handles_labels() if has_cls else ([], [])
            )
            if labels_l or labels_r:
                ax.legend(
                    lines_l + lines_r, labels_l + labels_r, fontsize=8, loc="best"
                )
        ax.grid(True, alpha=0.3)
        plot_idx += 1

        ax = axes[plot_idx]
        fourth_panel_has_data = False
        has_teacher_center = series("teacher_center_norm") is not None
        has_cls_center = series("cls_center_norm") is not None
        if has_teacher_center:
            ax.plot(
                df_instant["iteration"],
                df_instant["teacher_center_norm"],
                label="Teacher center (D, relational)",
            )
            fourth_panel_has_data = True
        if has_cls_center:
            ax.plot(
                df_instant["iteration"],
                df_instant["cls_center_norm"],
                label="CLS center (K, DINO)",
                color="tab:green",
            )
            fourth_panel_has_data = True
        has_lr = series("lr") is not None
        if has_lr:
            ax_twin = ax.twinx()
            ax_twin.plot(
                df_instant["iteration"],
                df_instant["lr"],
                label="Learning Rate",
                color="tab:red",
            )
            # An all-zero LR (for example a frozen-optimizer test run) has no
            # positive values, and matplotlib warns when log-scaling it.
            if (df_instant["lr"] > 0).any():
                ax_twin.set_yscale("log")
            ax_twin.set_ylabel("Learning Rate", color="tab:red")
            ax_twin.tick_params(axis="y", labelcolor="tab:red")
            fourth_panel_has_data = True
        if has_teacher_center or has_cls_center:
            ax.set_ylabel("Center Norm")
        if fourth_panel_has_data:
            ax.set_xlabel("Iteration")
            if has_lr and (has_teacher_center or has_cls_center):
                ax.set_title("Center Norms & Learning Rate")
                lines_l, labels_l = ax.get_legend_handles_labels()
                lines_r, labels_r = ax_twin.get_legend_handles_labels()
                ax.legend(
                    lines_l + lines_r,
                    labels_l + labels_r,
                    fontsize=8,
                    loc="best",
                )
            elif has_teacher_center or has_cls_center:
                ax.set_title("Center Norms")
                ax.legend(fontsize=8)
            else:
                ax.set_title("Learning Rate")
                lines_r, labels_r = ax_twin.get_legend_handles_labels()
                ax.legend(lines_r, labels_r, fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        else:
            ax.axis("off")

        # Teacher temperature, weighted contributions and the assignment
        # diagnostics that reveal target sharpness. Without these the DINO runs
        # have no per-iteration view of what the CLS objective is actually doing.
        weighted = [
            name
            for name in (
                "global_loss_weighted",
                "local_loss_weighted",
                "patch_loss_weighted",
                "koleo_loss_weighted",
            )
            if series(name) is not None
        ]
        teacher_temp_series = series("teacher_temp")
        if teacher_temp_series is not None or weighted:
            ax = axes[plot_idx]
            left_drawn = False
            if teacher_temp_series is not None:
                ax.plot(
                    df_instant["iteration"],
                    teacher_temp_series,
                    label="Teacher temperature",
                    color="tab:red",
                )
                left_drawn = True
                ax.set_ylabel("Teacher temperature", color="tab:red")
                ax.tick_params(axis="y", labelcolor="tab:red")
            if weighted:
                ax_twin = ax.twinx()
                for name in weighted:
                    ax_twin.plot(
                        df_instant["iteration"],
                        series(name),
                        label=name,
                        alpha=0.8,
                    )
                ax_twin.set_ylabel("Weighted contribution")
                lines_l, labels_l = ax.get_legend_handles_labels()
                lines_r, labels_r = ax_twin.get_legend_handles_labels()
                ax.legend(
                    lines_l + lines_r, labels_l + labels_r, fontsize=7, loc="best"
                )
            elif left_drawn:
                ax.legend(fontsize=8)
            ax.set_xlabel("Iteration")
            ax.set_title("Teacher Temperature & Weighted Losses")
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # DINO CLS assignment health. The statistics pool is the cached teacher
        # targets of both global crops (2 x batch), not the whole dataset, and
        # the active-prototype ratio divides by K.
        cls_entropy = series("cls_assignment_entropy")
        cls_occupancy = series("cls_active_prototype_ratio")
        if cls_entropy is not None or cls_occupancy is not None:
            ax = axes[plot_idx]
            drawn = False
            if cls_entropy is not None:
                ax.plot(
                    df_instant["iteration"],
                    cls_entropy,
                    label="Target entropy (nats)",
                    color="tab:blue",
                )
                drawn = True
            cls_perplexity = series("cls_prototype_perplexity")
            if cls_perplexity is not None:
                ax.plot(
                    df_instant["iteration"],
                    cls_perplexity,
                    label="Marginal perplexity",
                    color="tab:cyan",
                    alpha=0.8,
                )
                drawn = True
            ax.set_ylabel("Entropy (nats) / Perplexity", color="tab:blue")
            if cls_occupancy is not None:
                ax_twin = ax.twinx()
                ax_twin.plot(
                    df_instant["iteration"],
                    cls_occupancy,
                    label="Active prototype ratio (K denominator)",
                    color="tab:green",
                    alpha=0.8,
                )
                cls_occupancy_max = series("cls_max_prototype_occupancy")
                if cls_occupancy_max is not None:
                    ax_twin.plot(
                        df_instant["iteration"],
                        cls_occupancy_max,
                        label="Max occupancy (batch)",
                        color="tab:olive",
                        alpha=0.8,
                    )
                ax_twin.set_ylabel("Ratio", color="tab:green")
                lines_l, labels_l = ax.get_legend_handles_labels()
                lines_r, labels_r = ax_twin.get_legend_handles_labels()
                ax.legend(
                    lines_l + lines_r, labels_l + labels_r, fontsize=7, loc="best"
                )
            elif drawn:
                ax.legend(fontsize=7)
            ax.set_xlabel("Iteration")
            ax.set_title("DINO CLS Target Health (2 x batch pool)")
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # iBOT patch diagnostics use a different statistics pool, so they get
        # their own panel instead of sharing the CLS numbers.
        patch_entropy = series("assignment_entropy")
        patch_occupancy = series("active_prototype_ratio")
        if patch_entropy is not None or patch_occupancy is not None:
            ax = axes[plot_idx]
            drawn = False
            if patch_entropy is not None:
                ax.plot(
                    df_instant["iteration"],
                    patch_entropy,
                    label="Patch target entropy (nats)",
                    color="tab:blue",
                )
                drawn = True
            patch_perplexity = series("prototype_perplexity")
            if patch_perplexity is not None:
                ax.plot(
                    df_instant["iteration"],
                    patch_perplexity,
                    label="Patch marginal perplexity",
                    color="tab:cyan",
                    alpha=0.8,
                )
                drawn = True
            ax.set_ylabel("Entropy (nats) / Perplexity", color="tab:blue")
            if patch_occupancy is not None:
                ax_twin = ax.twinx()
                ax_twin.plot(
                    df_instant["iteration"],
                    patch_occupancy,
                    label="Active prototype ratio",
                    color="tab:green",
                    alpha=0.8,
                )
                patch_occupancy_max = series("max_prototype_occupancy")
                if patch_occupancy_max is not None:
                    ax_twin.plot(
                        df_instant["iteration"],
                        patch_occupancy_max,
                        label="Max occupancy",
                        color="tab:olive",
                        alpha=0.8,
                    )
                ax_twin.set_ylabel("Ratio", color="tab:green")
                lines_l, labels_l = ax.get_legend_handles_labels()
                lines_r, labels_r = ax_twin.get_legend_handles_labels()
                ax.legend(
                    lines_l + lines_r, labels_l + labels_r, fontsize=7, loc="best"
                )
            elif drawn:
                ax.legend(fontsize=7)
            ax.set_xlabel("Iteration")
            ax.set_title("iBOT Patch Target Health")
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # Signed KoLeo contribution; absent when the regularizer is disabled.
        koleo_raw = series("koleo_loss_raw")
        koleo_weighted = series("koleo_loss_weighted")
        if koleo_raw is not None or koleo_weighted is not None:
            ax = axes[plot_idx]
            if koleo_raw is not None:
                ax.plot(
                    df_instant["iteration"],
                    koleo_raw,
                    label="KoLeo loss (raw, signed)",
                    color="tab:purple",
                )
            if koleo_weighted is not None:
                ax.plot(
                    df_instant["iteration"],
                    koleo_weighted,
                    label="KoLeo loss (weighted, signed)",
                    color="tab:pink",
                    alpha=0.8,
                )
            skips = series("koleo_skipped_samples")
            if skips is not None and (skips > 0).any():
                # A skipped batch has no observed loss, so its count is drawn on
                # a second axis rather than read as a loss value.
                ax_twin = ax.twinx()
                ax_twin.plot(
                    df_instant["iteration"],
                    skips,
                    label="Skipped samples",
                    color="tab:gray",
                    linestyle="--",
                )
                ax_twin.set_ylabel("Skipped samples")
                lines_l, labels_l = ax.get_legend_handles_labels()
                lines_r, labels_r = ax_twin.get_legend_handles_labels()
                ax.legend(
                    lines_l + lines_r, labels_l + labels_r, fontsize=7, loc="best"
                )
            else:
                ax.legend(fontsize=7)
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("KoLeo loss (signed)")
            ax.set_title("KoLeo Contribution")
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        if df_metrics is not None and not df_metrics.empty:
            for cols, title in [
                (["Recall@1", "Recall@5", "Recall@10"], "Recall@K"),
                (["kNN_Acc_k1", "kNN_Acc_k5", "kNN_Acc_k20"], "kNN Accuracy"),
                (["NMI", "ARI"], "Clustering"),
                (["mAP", "Linear_Probing_Acc", "Linear_Probing_Balanced_Acc"], "Retrieval/Probe"),
                (["Silhouette_Score", "Purity"], "Structure Quality"),
            ]:
                if plot_idx >= len(axes):
                    break
                ax = axes[plot_idx]
                use_twin = title in {"Structure Quality", "Clustering"}
                ax_right = ax.twinx() if use_twin else None
                # Detect which splits are actually present to avoid empty split labels
                splits_present = [
                    s
                    for s in ["train", "test"]
                    if not df_metrics[df_metrics.get("split", "") == s].empty
                ]
                for split, marker in [("train", "o"), ("test", "s")]:
                    if split not in splits_present:
                        continue
                    sub = df_metrics[df_metrics.get("split", "") == split]
                    multi_split = len(splits_present) > 1
                    for col in cols:
                        if col in sub.columns:
                            numeric = pd.to_numeric(sub[col], errors="coerce")
                            if numeric.notna().any():
                                if use_twin and title == "Structure Quality":
                                    target_ax = (
                                        ax_right if col == "Silhouette_Score" else ax
                                    )
                                elif use_twin and title == "Clustering":
                                    target_ax = ax_right if col == "ARI" else ax
                                else:
                                    target_ax = ax
                                # Only append split suffix when multiple splits are present
                                lbl = f"{col} ({split})" if multi_split else col
                                color = None
                                if use_twin and title == "Structure Quality":
                                    color = (
                                        "tab:red"
                                        if col == "Silhouette_Score"
                                        else "tab:blue"
                                    )
                                elif use_twin and title == "Clustering":
                                    color = "tab:red" if col == "ARI" else "tab:blue"
                                target_ax.plot(
                                    sub["epoch"],
                                    numeric,
                                    marker=marker,
                                    label=lbl,
                                    color=color,
                                )
                ax.set_xlabel("Epoch")
                ax.set_title(title)
                if use_twin and title == "Structure Quality":
                    ax.set_ylabel("Purity", color="tab:blue")
                    ax.tick_params(axis="y", labelcolor="tab:blue")
                    ax_right.set_ylabel("Silhouette Score", color="tab:red")
                    ax_right.tick_params(axis="y", labelcolor="tab:red")
                    lines_l, labels_l = ax.get_legend_handles_labels()
                    lines_r, labels_r = ax_right.get_legend_handles_labels()
                    ax.legend(lines_l + lines_r, labels_l + labels_r, fontsize=7)
                elif use_twin and title == "Clustering":
                    ax.set_ylabel("NMI", color="tab:blue")
                    ax.tick_params(axis="y", labelcolor="tab:blue")
                    ax_right.set_ylabel("ARI", color="tab:red")
                    ax_right.tick_params(axis="y", labelcolor="tab:red")
                    lines_l, labels_l = ax.get_legend_handles_labels()
                    lines_r, labels_r = ax_right.get_legend_handles_labels()
                    ax.legend(lines_l + lines_r, labels_l + labels_r, fontsize=7)
                else:
                    ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
                plot_idx += 1

        for i in range(plot_idx, len(axes)):
            axes[i].axis("off")

        fig.tight_layout()
        out_path = out_dir / f"training_curves_{self.mode}.pdf"
        pages = [fig]
        dense_fig = self._dense_diagnostics_figure(df_instant, plt)
        if dense_fig is not None:
            # A dense-enabled run appends one diagnostics page to the same PDF.
            # Writing the CSV alone never produces these panels.
            pages.append(dense_fig)
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(out_path) as pdf:
            for page in pages:
                pdf.savefig(page, dpi=300, bbox_inches="tight")
        for page in pages:
            plt.close(page)
        print(f"[Info] Saved training curves to {out_path}")


_ENHANCED_BASE_FIELDS = ["epoch", "split"]


_ENHANCED_V070_METRIC_FIELDS = [
    "NMI",
    "ARI",
    "Recall@1",
    "Recall@5",
    "Recall@10",
    "kNN_Acc_k1",
    "kNN_Acc_k5",
    "kNN_Acc_k20",
    "Linear_Probing_Acc",
    "mAP",
    "Silhouette_Score",
    "Purity",
]


_ENHANCED_V070_FIELDS = _ENHANCED_BASE_FIELDS + _ENHANCED_V070_METRIC_FIELDS


_ENHANCED_METRIC_FIELDS = [
    "NMI",
    "ARI",
    "Recall@1",
    "Recall@5",
    "Recall@10",
    "kNN_Acc_k1",
    "kNN_Acc_k5",
    "kNN_Acc_k20",
    "Linear_Probing_Acc",
    "Linear_Probing_Balanced_Acc",
    "mAP",
    "Silhouette_Score",
    "Purity",
]


@dataclass
class EnhancedMetricsLogger:
    """Append-only embedding-metric log with a recognized v0.7.0 migration."""

    path: Path
    mode: str = "pretrain"

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = _ENHANCED_BASE_FIELDS + _ENHANCED_METRIC_FIELDS
        if self.path.exists() and self.path.stat().st_size > 0:
            self._validate_or_migrate_existing()
            return
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def _read_existing_csv(self) -> tuple[list[str], list[list[str]]]:
        try:
            with self.path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                try:
                    header = next(reader)
                except StopIteration:
                    raise ValueError(
                        f"enhanced-metrics schema in {self.path} is empty; refusing "
                        "to append."
                    ) from None
                rows: list[list[str]] = []
                for line_number, row in enumerate(reader, start=2):
                    if len(row) != len(header):
                        raise ValueError(
                            f"enhanced-metrics schema in {self.path} is malformed: "
                            f"row {line_number} has {len(row)} fields but the header "
                            f"has {len(header)}."
                        )
                    rows.append(row)
        except csv.Error as exc:
            raise ValueError(
                f"enhanced-metrics schema in {self.path} is malformed CSV: {exc}"
            ) from exc
        return header, rows

    def _validate_or_migrate_existing(self) -> None:
        header, rows = self._read_existing_csv()
        if len(set(header)) != len(header):
            raise ValueError(
                f"enhanced-metrics schema in {self.path} has duplicate columns; "
                "refusing to append."
            )
        if header == self.fieldnames:
            return
        if header == _ENHANCED_V070_FIELDS:
            self._migrate_v070_schema(rows)
            return
        raise ValueError(
            f"Unrecognized enhanced-metrics schema in {self.path}: {header}. "
            "Refusing to append. Move or remove the file to start a new log."
        )

    def _migrate_v070_schema(self, rows: list[list[str]]) -> None:
        """Atomically add an empty balanced-probe column to a v0.7.0 log."""
        import os
        import tempfile

        index = {name: position for position, name in enumerate(_ENHANCED_V070_FIELDS)}
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                newline="",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=self.path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                writer = csv.writer(handle)
                writer.writerow(self.fieldnames)
                for row in rows:
                    writer.writerow(
                        [
                            row[index[name]] if name in index else ""
                            for name in self.fieldnames
                        ]
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
            temp_path = None
        finally:
            # Only this call's own temporary file may be removed: a wildcard
            # sweep would delete an unrelated sidecar that existed beforehand.
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def log(self, epoch: int, split: str, metrics: dict[str, Any]) -> None:
        row = {"epoch": epoch, "split": split}
        for k in self.fieldnames[2:]:
            row[k] = metrics.get(k, "")
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def _extract_tokens(model: OTUFormerEncoder, images: torch.Tensor) -> torch.Tensor:
    feats = model.backbone.forward_features(images)
    if isinstance(feats, dict):
        return feats["x"]
    return feats


def _compute_grad_norm(model: nn.Module) -> float:
    return _compute_grad_norm_parameters(model.parameters())


def _compute_grad_norm_parameters(params, *, numeric: bool = True) -> float:
    """A finite check, not the reduction ``clip_grad_norm_`` performs.

    Per-parameter norms squared in Python floats stay finite where the float32
    accumulation inside the clipping call overflows, so a finite value here does
    not license the clip; the clip refuses its own non-finite total.

    ``numeric=False`` guards only: it returns 0.0 for a finite set instead of the
    value, so the per-tensor norms make one batched host transfer rather than one
    synchronization each. A non-finite set still returns a non-finite float, which
    the caller's ``isfinite`` check rejects.
    """
    grads = [p.grad.data for p in params if p.grad is not None]
    if not grads:
        return 0.0
    try:
        norms = torch._foreach_norm(grads, 2)
    except (RuntimeError, NotImplementedError):
        # Backends without the batched kernel keep the per-tensor path.
        norms = [grad.norm(2) for grad in grads]
    stacked = torch.stack(norms)
    if not numeric:
        return 0.0 if bool(torch.isfinite(stacked).all()) else math.inf
    values = stacked.tolist()
    return float(sum(value * value for value in values) ** 0.5)


@torch.no_grad()
def _ibot_diagnostics(
    teacher_probs: torch.Tensor, patch_center: torch.Tensor
) -> dict[str, float]:
    """Per-step iBOT stability diagnostics from teacher probabilities."""
    mean_assignment = teacher_probs.mean(dim=0)
    mean_entropy = -(
        mean_assignment * torch.log(mean_assignment.clamp_min(1e-12))
    ).sum()
    hard = teacher_probs.argmax(dim=-1)
    counts = torch.bincount(
        hard, minlength=teacher_probs.shape[1]
    ).float()
    per_position_entropy = -(
        teacher_probs * torch.log(teacher_probs.clamp_min(1e-12))
    ).sum(dim=-1).mean()
    return {
        "prototype_perplexity": float(torch.exp(mean_entropy)),
        "active_prototype_ratio": float((counts > 0).sum())
        / teacher_probs.shape[1],
        "assignment_entropy": float(per_position_entropy),
        "max_prototype_occupancy": float(counts.max())
        / teacher_probs.shape[0],
        "patch_center_norm": float(patch_center.norm()),
    }


def _mask_ratio_diagnostics(
    masks: list[torch.Tensor], patch_count: int
) -> dict[str, float]:
    """Per-sample actual masked fraction across the current step's views."""
    fractions = torch.cat(
        [mask.float().sum(dim=1) / patch_count for mask in masks]
    )
    return {
        "actual_mask_ratio_mean": float(fractions.mean()),
        "actual_mask_ratio_min": float(fractions.min()),
        "actual_mask_ratio_max": float(fractions.max()),
    }


@torch.no_grad()
def _relational_queue_metrics(
    *,
    capacity: int,
    occupancy: int,
    eligible: int,
    candidate_count: int,
    ages: tuple[int, ...],
    refreshed: int | None,
    evicted: int | None,
    teacher_stats: dict[str, torch.Tensor] | None,
    student_stats: list[dict[str, torch.Tensor]] | None,
    has_local: bool = True,
) -> dict[str, object]:
    """Instant FIFO coverage/update/statistics fields for one log step.

    Occupancy, eligibility and ages describe the *pre-step* candidate snapshot;
    the refresh/eviction counts are the successful post-step commit. Unavailable
    measurements stay blank instead of falling back to zero, and no similarity or
    loss is recomputed here.
    """
    metrics: dict[str, object] = {
        "relational_queue_capacity": int(capacity),
        "relational_queue_occupancy": int(occupancy),
        "relational_queue_eligible": int(eligible),
        "relational_candidate_count": int(candidate_count),
        "relational_history_age_mean": (
            float(sum(ages)) / len(ages) if ages else None
        ),
        "relational_history_age_max": (int(max(ages)) if ages else None),
        "relational_queue_refreshed": (
            None if refreshed is None else int(refreshed)
        ),
        "relational_queue_evicted": None if evicted is None else int(evicted),
    }
    # The teacher target is one per step, so its ``global`` columns always
    # apply; a run without local views never computed a local branch and must
    # not report the global numbers as local observations.
    for suffix in _RELATIONAL_BRANCH_SUFFIXES:
        if suffix == "local" and not has_local:
            for name in _RELATIONAL_TEACHER_STAT_NAMES:
                metrics[f"relational_teacher_{name}_{suffix}"] = None
            continue
        for name in _RELATIONAL_TEACHER_STAT_NAMES:
            value = (teacher_stats or {}).get(name)
            metrics[f"relational_teacher_{name}_{suffix}"] = (
                None if value is None else float(value.item())
            )
    by_branch = list(student_stats or [])
    for index, suffix in enumerate(_RELATIONAL_BRANCH_SUFFIXES):
        branch = by_branch[index] if index < len(by_branch) else {}
        for name in _RELATIONAL_STUDENT_STAT_NAMES:
            value = branch.get(name)
            metrics[f"relational_student_{name}_{suffix}"] = (
                None if value is None else float(value.item())
            )
    return metrics


def _compute_instant_metrics(
    model: OTUFormerEncoder,
    sample_view: torch.Tensor,
    device: torch.device,
) -> dict[str, float | None]:
    tokens = _extract_tokens(model, sample_view.to(device))
    cls_token_raw = tokens[:, 0]
    proj = model.projector(cls_token_raw)
    # A batch standard deviation needs at least two samples; a singleton batch is
    # a legitimate input, and reporting NaN as an observed value would be wrong.
    feature_std = (
        float(cls_token_raw.std(dim=0).mean().item())
        if cls_token_raw.shape[0] >= 2
        else None
    )
    return {
        "feature_std": feature_std,
        "embedding_norm_mean": float(proj.norm(dim=1).mean().item()),
        "cls_token_norm_mean": float(cls_token_raw.norm(dim=1).mean().item()),
    }


@dataclass(frozen=True)
class PreparedEvaluationManifest:
    """One invocation's evaluation rows, resolved and capped before decoding.

    The ordered rows, paths and labels are frozen together at startup, so a CSV
    edited mid-run cannot change the live evaluation population.
    """

    csv_path: Path
    paths: tuple[Path, ...]
    labels: np.ndarray | None


def prepare_evaluation_manifest(
    csv_path: Path,
    root_dir: Path,
    *,
    max_samples: int,
    seed: int,
    image_paths: tuple[Path, ...] | list[Path] | None = None,
    refs: tuple[str, ...] | None = None,
) -> PreparedEvaluationManifest:
    """Resolve, validate and cap the evaluation rows before any image is opened.

    Every resolved path must already be an existing file: a missing entry
    outside the sample is a startup error, not a decode failure at save time.
    ``image_paths`` is an already identity-checked selection for the same CSV and
    root (an ordinary pretrain run reuses its training resolution); the cap and
    the file checks still apply to it. ``refs`` must accompany it: the labels are
    read from the CSV again, so the ordered references are compared with the
    ones the paths were resolved from and a reordered rewrite is rejected instead
    of pairing correct labels with the wrong images.
    """
    import pandas as pd

    csv_path = Path(csv_path)
    labels: np.ndarray | None = None
    if image_paths is None:
        frame = pd.read_csv(csv_path)
        if "image" not in frame.columns:
            raise ValueError(f"CSV {csv_path} missing required 'image' column")
        paths = resolve_dataset_image_paths(
            csv_path, Path(root_dir), refs=[str(value) for value in frame["image"]]
        )
        if "label" in frame.columns:
            labels = frame["label"].astype(str).to_numpy()
    else:
        if refs is None:
            raise ValueError(
                "refs are required with image_paths: they prove the CSV rows "
                "still match the resolved selection."
            )
        paths = [Path(path) for path in image_paths]
        labels = _labels_for_rows(csv_path, len(paths), expected_refs=refs)
    if not paths:
        raise ValueError(f"CSV {csv_path} selected no evaluation rows")
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} evaluation image(s) do not resolve to an existing "
            f"file; first: {missing[0]}"
        )
    if 0 < max_samples < len(paths):
        # Same seeded, sorted selection without replacement the legacy
        # post-decode cap used; applied here it avoids the decode entirely.
        idx = sorted_sample_indices(len(paths), max_samples, seed)
        paths = [paths[int(index)] for index in idx]
        labels = None if labels is None else labels[idx]
    return PreparedEvaluationManifest(
        csv_path=csv_path,
        paths=tuple(Path(path) for path in paths),
        labels=labels,
    )


def _labels_for_rows(
    csv_path: Path, rows: int, *, expected_refs: tuple[str, ...]
) -> np.ndarray | None:
    """Labels of an already resolved CSV, read once for the snapshot.

    The ordered references must still match the ones the paths came from: a
    same-length rewrite that reorders rows would otherwise pair labels with the
    wrong images.
    """
    import pandas as pd

    frame = pd.read_csv(csv_path)
    if "image" not in frame.columns:
        raise ValueError(f"CSV {csv_path} missing required 'image' column")
    refs = tuple(str(value) for value in frame["image"])
    if refs != tuple(expected_refs):
        raise ValueError(
            f"CSV {csv_path} changed while it was being prepared: the row order "
            "differs from the resolved selection."
        )
    if "label" not in frame.columns:
        return None
    labels = frame["label"].astype(str).to_numpy()
    if len(labels) != rows:
        raise ValueError(
            f"CSV {csv_path} changed while it was being prepared: {len(labels)} "
            f"label rows for {rows} resolved paths."
        )
    return labels


@torch.no_grad()
def _compute_embeddings_from_csv(
    model: OTUFormerEncoder,
    csv_path: Path,
    root_dir: Path,
    image_size: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    paths: tuple[Path, ...] | list[Path] | None = None,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    import pandas as pd
    from PIL import Image

    if paths is None:
        df = pd.read_csv(csv_path)
        if "image" not in df.columns:
            raise ValueError(f"CSV {csv_path} missing required 'image' column")

        refs = [str(v) for v in df["image"]]
        missing_direct = [
            ref
            for ref in refs
            if _supports_recursive_lookup(ref) and not (root_dir / Path(ref)).exists()
        ]
        by_relative, by_name = ({}, {})
        if missing_direct:
            by_relative, by_name = _build_recursive_index(root_dir)
        image_paths = [
            _resolve_image_path(root_dir, ref, by_relative, by_name) for ref in refs
        ]
        labels = df["label"].astype(str).to_numpy() if "label" in df.columns else None
    else:
        # A prepared manifest already resolved and capped the rows; rereading the
        # CSV here would let a mid-run edit change the evaluation population.
        image_paths = [Path(path) for path in paths]

    tf = center_crop_eval_transform(image_size)
    all_embs: list[np.ndarray] = []
    model.eval()
    _ = num_workers
    for start in range(0, len(image_paths), max(1, batch_size)):
        batch_paths = image_paths[start : start + max(1, batch_size)]
        imgs = []
        for p in batch_paths:
            with Image.open(p) as im:
                img = im.convert("RGB")
            imgs.append(tf(img))
        batch = torch.stack(imgs, dim=0).to(device)
        # Use raw CLS token (backbone output) for evaluation, matching ref default
        # (ref uses --use_projector_output flag which defaults to False)
        tokens = _extract_tokens(model, batch)
        emb = tokens[:, 0].cpu().numpy()
        all_embs.append(emb)
    embs = (
        np.concatenate(all_embs, axis=0)
        if all_embs
        else np.zeros((0, 1), dtype=np.float32)
    )
    return embs, labels


def _compute_all_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray | None,
    compute_linear_probe: bool,
) -> dict[str, Any]:
    fields = {
        "NMI": "",
        "ARI": "",
        "Recall@1": "",
        "Recall@5": "",
        "Recall@10": "",
        "kNN_Acc_k1": "",
        "kNN_Acc_k5": "",
        "kNN_Acc_k20": "",
        "Linear_Probing_Acc": "",
        "Linear_Probing_Balanced_Acc": "",
        "mAP": "",
        "Silhouette_Score": "",
        "Purity": "",
    }
    if (
        labels is None
        or len(embeddings) == 0
        or len(np.unique(labels)) < 2
    ):
        return fields

    try:
        fields.update(compute_recall_at_k(embeddings, labels, k_values=[1, 5, 10]))
        fields.update(compute_knn_accuracy(embeddings, labels, k_values=[1, 5, 20]))
        fields["mAP"] = compute_map(embeddings, labels)
        clustering = compute_clustering_metrics(embeddings, labels)
        fields["NMI"] = clustering.get("NMI", "")
        fields["ARI"] = clustering.get("ARI", "")
        fields["Silhouette_Score"] = clustering.get(
            "Silhouette_Score", clustering.get("Silhouette", "")
        )
        fields["Purity"] = clustering.get("Purity", "")
        if compute_linear_probe:
            probing = compute_linear_probing_metrics(embeddings, labels)
            fields["Linear_Probing_Acc"] = probing["Linear_Probing_Acc"]
            fields["Linear_Probing_Balanced_Acc"] = probing[
                "Linear_Probing_Balanced_Acc"
            ]
    except Exception as exc:
        print(f"[Warning] Error computing metrics: {exc}")
    return {key: "" if value is None else value for key, value in fields.items()}


def _compute_and_log_all_metrics(
    args: argparse.Namespace,
    model: OTUFormerEncoder,
    device: torch.device,
    epoch: int,
    logs_dir: Path,
    metrics_logger: EnhancedMetricsLogger,
    eval_image_size: int,
    force_linear_probe: bool = False,
    manifest: PreparedEvaluationManifest | None = None,
) -> None:
    visualize_csv = getattr(args, "visualize_data", "") or getattr(
        args, "train_data", ""
    )
    compute_lp = force_linear_probe or (((epoch + 1) % 10) == 0)

    if visualize_csv:
        try:
            if manifest is not None:
                feats, labels = _compute_embeddings_from_csv(
                    model=model,
                    csv_path=manifest.csv_path,
                    root_dir=Path(args.input_images_dir),
                    image_size=eval_image_size,
                    device=device,
                    batch_size=max(1, args.batch_size),
                    num_workers=max(0, args.num_workers),
                    paths=manifest.paths,
                    labels=manifest.labels,
                )
                feats_eval, labels_eval = feats, labels
            else:
                feats, labels = _compute_embeddings_from_csv(
                    model=model,
                    csv_path=Path(visualize_csv),
                    root_dir=Path(args.input_images_dir),
                    image_size=eval_image_size,
                    device=device,
                    batch_size=max(1, args.batch_size),
                    num_workers=max(0, args.num_workers),
                )
                feats_eval, labels_eval = feats, labels
                if 0 < args.metrics_sample_size < len(feats):
                    idx = sorted_sample_indices(
                        len(feats), args.metrics_sample_size, args.seed
                    )
                    feats_eval = feats[idx]
                    labels_eval = None if labels is None else labels[idx]
            metrics = _compute_all_metrics(
                feats_eval, labels_eval, compute_linear_probe=compute_lp
            )
            metrics_logger.log(epoch + 1, "train", metrics)
            print(f"[Metrics] Epoch {epoch + 1}:")
            if labels_eval is None:
                print("  [Info] Skipping supervised embedding metrics: no labels provided.")
            elif len(np.unique(labels_eval)) < 2:
                print(
                    "  [Info] Skipping supervised embedding metrics: "
                    "fewer than two label classes after sampling."
                )
            for k, v in metrics.items():
                if v != "":
                    print(f"  {k}: {float(v):.4f}")
            if len(feats_eval) >= 10:
                out_path = logs_dir / f"umap.train.epoch_{epoch + 1:04d}.pdf"
                run_umap(
                    feats_eval,
                    labels_eval,
                    out_path,
                    n_components=2,
                    n_neighbors=args.umap_n_neighbors,
                    min_dist=args.umap_min_dist,
                    metric=args.umap_metric,
                    max_classes=args.visualize_class_number,
                    title=f"UMAP Train - Epoch {epoch + 1}",
                )
                print(f"[Info] Saved UMAP plot to {out_path}")
            else:
                print("[Info] Skipping UMAP: fewer than 10 samples after sampling.")
        except Exception as exc:
            print(
                f"[Warning] Failed to compute embedding metrics at epoch {epoch + 1}: {exc}"
            )
