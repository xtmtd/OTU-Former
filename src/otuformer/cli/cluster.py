"""cluster command stub with full options."""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Literal

import click
import typer

app = typer.Typer(
    help=(
        "Cluster embeddings into morphOTUs via UPGMA hierarchical clustering.\n\n"
        "Builds a distance matrix from embeddings, constructs a UPGMA dendrogram,\n"
        "and partitions it at multiple distance cutoffs. Supports PCA whitening,\n"
        "local scaling, and bootstrap support estimation.\n\n"
        "Quick example:\n\n"
        "  otuformer cluster --embeddings runs/extract/embeddings.csv\n"
        "  otuformer cluster --embeddings embeddings.csv --pca-whitening true --local-scaling true --num-replicates 100\n"
    )
)


def _distance_stats_frame(dist_matrix, name: str):
    import numpy as np
    import pandas as pd

    values = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    row = {
        "name": name,
        "n_samples": int(dist_matrix.shape[0]),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }
    return pd.DataFrame([row])


def _intra_class_distance_stats(
    dist_matrix,
    ids: list[str],
    id_to_label: dict[str, str],
):
    import numpy as np
    import pandas as pd

    idx_and_label = [(i, id_to_label[s]) for i, s in enumerate(ids) if s in id_to_label]
    if not idx_and_label:
        return pd.DataFrame(
            columns=[
                "class",
                "n_samples",
                "n_pairs",
                "mean",
                "median",
                "std",
                "min",
                "max",
                "q25",
                "q75",
            ]
        )

    rows = []
    labels = sorted({label for _, label in idx_and_label})
    for label in labels:
        label_idx = np.array([i for i, l in idx_and_label if l == label], dtype=int)
        n_samples = int(len(label_idx))
        if n_samples < 2:
            rows.append(
                {
                    "class": label,
                    "n_samples": n_samples,
                    "n_pairs": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "q25": np.nan,
                    "q75": np.nan,
                }
            )
            continue
        class_dist = dist_matrix[np.ix_(label_idx, label_idx)]
        intra = class_dist[np.triu_indices_from(class_dist, k=1)]
        rows.append(
            {
                "class": label,
                "n_samples": n_samples,
                "n_pairs": int(len(intra)),
                "mean": float(np.mean(intra)),
                "median": float(np.median(intra)),
                "std": float(np.std(intra)),
                "min": float(np.min(intra)),
                "max": float(np.max(intra)),
                "q25": float(np.percentile(intra, 25)),
                "q75": float(np.percentile(intra, 75)),
            }
        )
    return pd.DataFrame(rows)


def _plot_distance_distributions(dist_matrix, label: str, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    values = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=60)
    ax.set_title(f"Distance distribution ({label})")
    ax.set_xlabel("distance")
    ax.set_ylabel("frequency")
    fig.tight_layout()
    fig.savefig(out_dir / f"distance_hist_{label}.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sorted_vals = np.sort(values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cdf)
    ax.set_title(f"Cumulative distribution ({label})")
    ax.set_xlabel("distance")
    ax.set_ylabel("cumulative frequency")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"distance_cum_{label}.pdf")
    plt.close(fig)

    positive = values[values > 0]
    if len(positive) > 0:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(positive, bins=60)
        ax.set_yscale("log")
        ax.set_title(f"Distance distribution ({label}, log scale)")
        ax.set_xlabel("distance")
        ax.set_ylabel("frequency")
        fig.tight_layout()
        fig.savefig(out_dir / f"distance_hist_{label}_log.pdf")
        plt.close(fig)


def _plot_partition_scan(scan_df, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(scan_df["threshold"], scan_df["clusters"], marker="o")
    ax.set_xlabel("Distance cutoff")
    ax.set_ylabel("Number of clusters")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _contiguous_runs(indices):
    if len(indices) == 0:
        return
    block = [indices[0]]
    for idx in indices[1:]:
        if idx == block[-1] + 1:
            block.append(idx)
        else:
            yield block
            block = [idx]
    yield block


def _compute_clade_heights(z, ids: list[str]):
    n = len(ids)
    current_index = n
    clusters = {i: {ids[i]} for i in range(n)}
    heights = {}
    for row in z:
        a, b, height, _ = row
        a, b = int(a), int(b)
        clade = clusters[a] | clusters[b]
        if 1 < len(clade) < n:
            heights[frozenset(clade)] = height
        clusters[current_index] = clade
        current_index += 1
    return heights


def _compute_clade_y_positions(z, ids: list[str], leaf_order, y_lookup):
    n = len(ids)
    node_y = {i: y_lookup[i] for i in range(n)}
    clusters = {i: {ids[i]} for i in range(n)}
    clade_y = {}
    current = n
    for a, b, _, _ in z:
        a, b = int(a), int(b)
        clade = clusters[a] | clusters[b]
        y = 0.5 * (node_y[a] + node_y[b])
        if 1 < len(clade) < n:
            clade_y[frozenset(clade)] = y
        node_y[current] = y
        clusters[current] = clade
        current += 1
    return clade_y


def _annotate_supports(
    ax, support_dict, clade_heights, clade_y_positions, min_support=50.0
):
    if not support_dict:
        return
    x_min, x_max = ax.get_xlim()
    x_offset = 0.01 * (x_max - x_min)
    for clade, support in support_dict.items():
        if support < min_support:
            continue
        if clade not in clade_heights or clade not in clade_y_positions:
            continue
        x = clade_heights[clade]
        y = clade_y_positions[clade]
        ax.text(
            x + x_offset,
            y,
            f"{support:.0f}",
            ha="left",
            va="center",
            fontsize=7,
            color="darkred",
            clip_on=False,
        )


def _plot_partition_panel(
    ax_names,
    ax_parts,
    ordered_ids,
    partitions,
    leaf_order,
    tip_y_positions,
    bar_width=0.75,
    tip_text_x=0.98,
    tip_text_ha="right",
):
    import numpy as np
    import seaborn as sns
    from matplotlib.patches import Rectangle
    from matplotlib import transforms

    n_samples = len(ordered_ids)
    y_positions = [tip_y_positions[name] for name in ordered_ids]
    y_step = (y_positions[1] - y_positions[0]) if n_samples > 1 else 10.0
    top_label_y = min(y_positions) - 0.7 * abs(y_step)

    ax_names.axis("off")
    ax_names.set_xlim(0, 1)
    tip_transform = transforms.blended_transform_factory(
        ax_names.transAxes, ax_names.transData
    )
    for tip in ordered_ids:
        ax_names.text(
            tip_text_x,
            tip_y_positions[tip],
            tip,
            ha=tip_text_ha,
            va="center",
            fontsize=9,
            transform=tip_transform,
        )

    n_parts = len(partitions)
    if n_parts == 0:
        return

    padding = max(0.05, 0.04 * bar_width)
    ax_parts.set_xlim(-bar_width / 2 - padding, (n_parts - 1) + bar_width / 2 + padding)
    ax_parts.set_xticks([])
    ax_parts.set_yticks([])
    for spine in ax_parts.spines.values():
        spine.set_visible(False)

    bar_colors = sns.color_palette("tab20", max(20, n_parts * 2))
    for col_idx, (th, labels) in enumerate(partitions.items()):
        ordered_labels = np.asarray(labels)[leaf_order]
        cluster_ids = list(dict.fromkeys(ordered_labels))
        colors = {
            cid: bar_colors[i % len(bar_colors)] for i, cid in enumerate(cluster_ids)
        }
        cluster_sizes = {cid: int((ordered_labels == cid).sum()) for cid in cluster_ids}

        for cid in cluster_ids:
            idxs = np.where(ordered_labels == cid)[0]
            for block in _contiguous_runs(idxs):
                y0 = y_positions[block[0]] - abs(y_step) / 2.0
                height = abs(y_step) * len(block)
                rect = Rectangle(
                    (col_idx - bar_width / 2, y0),
                    bar_width,
                    height,
                    facecolor=colors[cid],
                    edgecolor="white",
                    linewidth=0.5,
                )
                ax_parts.add_patch(rect)
                ax_parts.text(
                    col_idx,
                    y0 + height / 2,
                    str(cluster_sizes[cid]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if cluster_sizes[cid] > 1 else "black",
                )

        ax_parts.text(
            col_idx,
            top_label_y,
            f"{len(np.unique(labels))}\n{th:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def _plot_corrected_cluster_panel(
    ax_corr,
    ordered_ids,
    tip_y_positions,
    corrected_labels,
    bar_width=0.08,
):
    import seaborn as sns
    from matplotlib.patches import Rectangle

    if len(ordered_ids) == 0:
        ax_corr.axis("off")
        return

    y_positions = [tip_y_positions[name] for name in ordered_ids]
    y_step = (y_positions[1] - y_positions[0]) if len(y_positions) > 1 else 10.0
    abs_step = abs(y_step)

    cluster_sequence = [corrected_labels.get(name, "") for name in ordered_ids]
    unique_clusters = list(dict.fromkeys(cluster_sequence))
    palette = sns.color_palette("tab20", max(20, len(unique_clusters)))
    color_map = {
        cid: palette[i % len(palette)] for i, cid in enumerate(unique_clusters)
    }

    ax_corr.set_xlim(0.0, 1.0)
    ax_corr.set_xticks([])
    ax_corr.set_yticks([])
    for spine in ax_corr.spines.values():
        spine.set_visible(False)

    bar_w = max(0.02, float(bar_width))
    bar_x = 0.0
    label_x = min(0.98, bar_x + bar_w + 0.03)

    run_start = 0
    for idx in range(1, len(ordered_ids) + 1):
        is_end = idx == len(ordered_ids)
        current_cid = corrected_labels.get(ordered_ids[run_start], "")
        next_cid = corrected_labels.get(ordered_ids[idx], "") if not is_end else None
        if not is_end and next_cid == current_cid:
            continue

        first_id = ordered_ids[run_start]
        last_id = ordered_ids[idx - 1]
        y0 = tip_y_positions[first_id] - abs_step / 2.0
        height = abs_step * (idx - run_start)
        rect = Rectangle(
            (bar_x, y0),
            bar_w,
            height,
            facecolor=color_map.get(current_cid, (0.8, 0.8, 0.8)),
            edgecolor="white",
            linewidth=0.3,
        )
        ax_corr.add_patch(rect)

        label_y = 0.5 * (tip_y_positions[first_id] + tip_y_positions[last_id])
        ax_corr.text(
            label_x,
            label_y,
            current_cid,
            ha="left",
            va="center",
            fontsize=8,
        )
        ax_corr.text(
            bar_x + bar_w / 2.0,
            label_y,
            str(idx - run_start),
            ha="center",
            va="center",
            fontsize=8,
            color="white" if (idx - run_start) > 1 else "black",
        )

        if not is_end:
            y_sep = tip_y_positions[ordered_ids[idx]] - abs_step / 2.0
            ax_corr.plot(
                [bar_x, bar_x + bar_w], [y_sep, y_sep], color="white", linewidth=0.8
            )
        run_start = idx


def _plot_upgma_partition_tree_panel(
    z,
    ids: list[str],
    partitions,
    out_path: Path,
    support_dict=None,
    bootstrap_cutoff=50.0,
    corrected_labels=None,
    corrected_bar_width=0.08,
    figure_width: float | None = None,
):
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from scipy.cluster import hierarchy
    import numpy as np

    n_samples = len(ids)
    max_tip_chars = max((len(str(v)) for v in ids), default=8)
    max_corr_chars = max(
        (len(str(v)) for v in (corrected_labels.values() if corrected_labels else [])),
        default=6,
    )

    n_parts = len(partitions) if partitions else 1
    base_partition_width = max(1.2, n_parts * 0.75 * 0.35)
    partition_width = base_partition_width * 0.8

    tip_text_x = 1.0
    tip_fontsize = 9
    corr_fontsize = 8
    base_fig_width = float(figure_width) if figure_width else 14.0
    fig_height = max(6, 0.25 * n_samples)

    def _measure_text_widths_in(texts, font_sizes):
        fig = plt.figure(figsize=(2, 2))
        artists = [
            fig.text(0.0, 0.0, text or "", fontsize=size, alpha=0.0)
            for text, size in zip(texts, font_sizes)
        ]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        widths = [
            artist.get_window_extent(renderer=renderer).width / fig.dpi
            for artist in artists
        ]
        plt.close(fig)
        return widths

    if corrected_labels:
        max_tip = max((str(v) for v in ids), default="")
        max_corr = max((str(v) for v in corrected_labels.values()), default="")
        tip_w_in, tip_char_in, corr_w_in, _ = _measure_text_widths_in(
            [max_tip, "M", max_corr, "M"],
            [tip_fontsize, tip_fontsize, corr_fontsize, corr_fontsize],
        )

        gap_in = 1.5 * tip_char_in
        names_width_in = max(1.4, (tip_w_in + gap_in) / max(tip_text_x, 0.9))

        bar_w = max(0.02, float(corrected_bar_width))
        bar_x = 0.0
        label_x = min(0.98, bar_x + bar_w + 0.03)
        corr_width_in = max(0.55, (corr_w_in + 0.08) / max(0.2, (1.0 - label_x)))

        width_ratios = [names_width_in, corr_width_in, partition_width, 2.8]
        avg_axis = sum(width_ratios) / len(width_ratios)
        wspace = max(0.0, min(0.1, gap_in / max(avg_axis, 1e-6)))
        fig_width = max(base_fig_width, sum(width_ratios))
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = gridspec.GridSpec(1, 4, width_ratios=width_ratios, wspace=wspace)
        ax_tree = fig.add_subplot(gs[0, 3])
    else:
        fig = plt.figure(figsize=(base_fig_width, fig_height))
        width_ratios = [1.4, partition_width, 2.8]
        gs = gridspec.GridSpec(1, 3, width_ratios=width_ratios, wspace=0.05)
        ax_tree = fig.add_subplot(gs[0, 2])

    dendro = hierarchy.dendrogram(
        z,
        labels=ids,
        orientation="right",
        color_threshold=0,
        no_labels=True,
        ax=ax_tree,
        above_threshold_color="C0",
    )

    leaf_order = dendro["leaves"]
    ordered_ids = [ids[i] for i in leaf_order]
    x_min, x_max = ax_tree.get_xlim()
    ax_tree.set_xlim(x_min, x_max + 0.08 * (x_max - x_min))

    y_step = 10
    y_positions = (y_step / 2) + y_step * np.arange(len(ordered_ids))
    y_lookup = {leaf_order[i]: y_positions[i] for i in range(len(leaf_order))}
    tip_y_positions = {ids[idx]: y_lookup[idx] for idx in leaf_order}

    if corrected_labels:
        ax_names = fig.add_subplot(gs[0, 0], sharey=ax_tree)
        ax_corr = fig.add_subplot(gs[0, 1], sharey=ax_tree)
        ax_parts = fig.add_subplot(gs[0, 2], sharey=ax_tree)
    else:
        ax_names = fig.add_subplot(gs[0, 0], sharey=ax_tree)
        ax_corr = None
        ax_parts = fig.add_subplot(gs[0, 1], sharey=ax_tree)
    if partitions:
        _plot_partition_panel(
            ax_names,
            ax_parts,
            ordered_ids,
            partitions,
            leaf_order,
            tip_y_positions,
            tip_text_x=tip_text_x,
            tip_text_ha="right",
        )
    else:
        ax_names.axis("off")
        ax_parts.axis("off")

    if corrected_labels and ax_corr is not None:
        _plot_corrected_cluster_panel(
            ax_corr,
            ordered_ids,
            tip_y_positions,
            corrected_labels,
            bar_width=corrected_bar_width,
        )

    ymin = y_positions[-1] + y_step
    ymax = y_positions[0] - y_step
    axes = [ax_names, ax_parts, ax_tree]
    if ax_corr is not None:
        axes.insert(1, ax_corr)
    for ax in axes:
        ax.set_ylim(ymin, ymax)

    ax_tree.set_xlabel("Distance")
    ax_tree.set_yticks([])
    for spine in ax_tree.spines.values():
        spine.set_visible(False)
    ax_tree.spines["bottom"].set_visible(True)

    if support_dict:
        clade_heights = _compute_clade_heights(z, ids)
        clade_y = _compute_clade_y_positions(z, ids, leaf_order, y_lookup)
        _annotate_supports(
            ax_tree, support_dict, clade_heights, clade_y, min_support=bootstrap_cutoff
        )

    fig.subplots_adjust(top=0.97, right=0.97, left=0.02)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_metrics_dashboard(metrics_df, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    axes[0, 0].plot(metrics_df["cutoff"], metrics_df["n_OTUs"], "o-", color="#1f77b4")
    if "n_true_species" in metrics_df.columns:
        axes[0, 0].axhline(
            y=metrics_df["n_true_species"].iloc[0],
            color="red",
            linestyle="--",
            label="True species",
        )
        axes[0, 0].legend()
    axes[0, 0].set_xlabel("Distance cutoff")
    axes[0, 0].set_ylabel("Number of OTUs")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(
        metrics_df["cutoff"], metrics_df["ARI"], "o-", color="#2ca02c", label="ARI"
    )
    axes[0, 1].plot(
        metrics_df["cutoff"], metrics_df["NMI"], "s-", color="#ff7f0e", label="NMI"
    )
    axes[0, 1].plot(
        metrics_df["cutoff"], metrics_df["AMI"], "^-", color="#d62728", label="AMI"
    )
    axes[0, 1].set_xlabel("Distance cutoff")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_title("Clustering Agreement")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(
        metrics_df["cutoff"], metrics_df["BCubed_fscore"], "o-", color="#9467bd"
    )
    axes[0, 2].set_xlabel("Distance cutoff")
    axes[0, 2].set_ylabel("BCubed F-score")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(
        metrics_df["cutoff"], metrics_df["splitting_index"], "o-", color="#8c564b"
    )
    axes[1, 0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[1, 0].set_xlabel("Distance cutoff")
    axes[1, 0].set_ylabel("Splitting index")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(
        metrics_df["cutoff"], metrics_df["lumping_index"], "o-", color="#e377c2"
    )
    axes[1, 1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[1, 1].set_xlabel("Distance cutoff")
    axes[1, 1].set_ylabel("Lumping index")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(
        metrics_df["cutoff"],
        metrics_df["silhouette_species"],
        "o-",
        color="#7f7f7f",
        label="Species-level",
    )
    axes[1, 2].plot(
        metrics_df["cutoff"],
        metrics_df["silhouette_cluster"],
        "s-",
        color="#bcbd22",
        label="Cluster-level",
    )
    axes[1, 2].set_xlabel("Distance cutoff")
    axes[1, 2].set_ylabel("Silhouette score")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    axes[2, 0].plot(
        metrics_df["cutoff"],
        metrics_df["homogeneity"],
        "o-",
        color="#17becf",
        label="Homogeneity",
    )
    axes[2, 0].plot(
        metrics_df["cutoff"],
        metrics_df["completeness"],
        "s-",
        color="#ff9896",
        label="Completeness",
    )
    axes[2, 0].plot(
        metrics_df["cutoff"],
        metrics_df["v_measure"],
        "^-",
        color="#98df8a",
        label="V-measure",
    )
    axes[2, 0].set_xlabel("Distance cutoff")
    axes[2, 0].set_ylabel("Score")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(metrics_df["n_OTUs"], metrics_df["WSS"], "o-", color="#c5b0d5")
    axes[2, 1].set_xlabel("Number of OTUs")
    axes[2, 1].set_ylabel("WSS")
    axes[2, 1].set_title("WSS Elbow")
    axes[2, 1].grid(True, alpha=0.3)

    axes[2, 2].plot(
        metrics_df["cutoff"], metrics_df["purity"], "o-", color="#1f77b4", linewidth=2
    )
    axes[2, 2].set_xlabel("Distance cutoff")
    axes[2, 2].set_ylabel("Cluster Purity")
    axes[2, 2].set_title("Cluster Purity")
    axes[2, 2].grid(True, alpha=0.3)
    axes[2, 2].set_ylim([0, 1.05])

    fig.suptitle("Partition Quality Metrics Dashboard", fontsize=16, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _compute_monophyly_proportion_upgma(
    z, ids: list[str], true_labels: list[str]
) -> float | None:
    from scipy.cluster import hierarchy

    if len(ids) != len(true_labels):
        return None

    id_to_label = {ids[i]: true_labels[i] for i in range(len(ids))}
    unique_species = set(true_labels)
    if not unique_species:
        return None

    root = hierarchy.to_tree(z, rd=False)
    all_clades = []

    def _collect(node):
        if node.left is None and node.right is None:
            return {ids[node.id]}
        left = _collect(node.left)
        right = _collect(node.right)
        clade = left | right
        all_clades.append(clade)
        return clade

    _collect(root)

    monophyletic = 0
    for species in unique_species:
        species_tips = {
            tip_id for tip_id, label in id_to_label.items() if label == species
        }
        if len(species_tips) <= 1:
            monophyletic += 1
            continue
        if any(clade == species_tips for clade in all_clades):
            monophyletic += 1

    return monophyletic / len(unique_species)


def _parse_bool_option(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise typer.BadParameter(
        f"Invalid value for --{name}: {value!r}. Use true or false."
    )


def _format_user_command(ctx: typer.Context, params: dict[str, object]) -> str:
    parts = ["otuformer", "cluster"]
    for key, value in params.items():
        source = ctx.get_parameter_source(key)
        if source is not click.core.ParameterSource.COMMANDLINE:
            continue
        option = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            parts.extend([option, str(value).lower()])
            continue
        if value in (None, ""):
            continue
        parts.extend([option, str(value)])
    return " ".join(parts)


def _l2_normalize_rows(x):
    import numpy as np

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


@app.callback(invoke_without_command=True)
def cluster(
    ctx: typer.Context,
    embeddings: Path = typer.Option(..., "--embeddings", help="Embeddings CSV."),
    out_dir: Path = typer.Option(
        Path("runs/cluster"),
        "--out-dir",
        help="Output directory. Existing contents are cleared before each run.",
    ),
    distance: str = typer.Option(
        "cosine",
        "--distance",
        help="Distance metric for pairwise matrix: cosine or euclidean.",
    ),
    prefix: str = typer.Option(
        "OTU", "--prefix", help="Cluster prefix used in partition table labels."
    ),
    pca_whitening: str = typer.Option(
        "false",
        "--pca-whitening",
        help="Enable PCA whitening before distance computation (true|false).",
    ),
    pca_components: int = typer.Option(256, "--pca-components", help="PCA components."),
    local_scaling: str = typer.Option(
        "false",
        "--local-scaling",
        help="Enable Mutual-Proximity style local scaling on distance matrix (true|false).",
    ),
    local_k: int = typer.Option(
        0, "--local-k", help="Fixed k for local scaling (0=auto)."
    ),
    local_k_strategy: Literal["adaptive", "sqrt", "log", "fixed"] = typer.Option(
        "adaptive",
        "--local-k-strategy",
        help="Auto-k strategy when --local-k=0: adaptive | sqrt | log | fixed.",
    ),
    cutoff_min: float = typer.Option(0.05, "--cutoff-min", help="Minimum cutoff."),
    cutoff_max: float = typer.Option(
        1.0, "--cutoff-max", help="Maximum cutoff (inclusive)."
    ),
    cutoff_step: float = typer.Option(
        0.05,
        "--cutoff-step",
        help="Cutoff step for linear scan when --custom-cutoffs is not provided.",
    ),
    custom_cutoffs: str | None = typer.Option(
        None,
        "--custom-cutoffs",
        help="Comma-separated cutoffs (overrides min/max/step scan).",
    ),
    support_mode: Literal["subsample", "bootstrap"] = typer.Option(
        "subsample",
        "--support-mode",
        help="Support estimation mode: subsample (without replacement) or bootstrap (with replacement).",
    ),
    num_replicates: int = typer.Option(
        0,
        "--num-replicates",
        help="Number of support estimation replicates (0 disables).",
    ),
    subsample_ratio: float = typer.Option(
        0.8,
        "--subsample-ratio",
        help="Feature fraction for subsample mode (ignored in bootstrap mode).",
    ),
    support_display_cutoff: float = typer.Option(
        50.0,
        "--support-display-cutoff",
        help="Display threshold for support labels on tree visualizations.",
    ),
    save_bootstrap_trees: str = typer.Option(
        "false",
        "--save-bootstrap-trees",
        help="Save all bootstrap replicate trees to UPGMA/bootstrap_trees.nwk (true|false).",
    ),
    save_distances: str = typer.Option(
        "false",
        "--save-distances",
        help="Save full pairwise distance matrix to distance_statistics/distance_matrix.csv (true|false).",
    ),
    max_distance_pairs: int = typer.Option(
        1000000, "--max-distance-pairs", help="Max pairs to store."
    ),
    label_csv: Path | None = typer.Option(
        None,
        "--label-csv",
        "--labels",
        help="Optional label CSV for partition metrics (supports id/label or image/label).",
    ),
    metrics_sample_size: int = typer.Option(
        10000, "--metrics-sample-size", help="Max samples for metrics."
    ),
    cpus: int = typer.Option(8, "--cpus", help="CPU threads."),
    random_state: int = typer.Option(42, "--random-state", help="Random seed."),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    import numpy as np

    from otuformer.delineation.distance import (
        apply_local_scaling,
        apply_pca_whitening,
        auto_select_k,
        compute_cosine_distances,
        compute_euclidean_distances,
    )
    import pandas as pd

    from otuformer.delineation.partition import (
        build_partitions_from_linkage,
        compute_partition_metrics,
        export_partition_tables,
    )
    from otuformer.delineation.tree import (
        build_upgma,
        compute_bootstrap_support,
        upgma_to_newick,
    )
    from otuformer.utils.io import read_csv
    from otuformer.utils.logging import TeeLogger

    pca_whitening_enabled = _parse_bool_option("pca-whitening", pca_whitening)
    local_scaling_enabled = _parse_bool_option("local-scaling", local_scaling)
    save_bootstrap_trees_enabled = _parse_bool_option(
        "save-bootstrap-trees", save_bootstrap_trees
    )
    save_distances_enabled = _parse_bool_option("save-distances", save_distances)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = TeeLogger(out_dir / "logs" / "cluster.log")
    original_stderr = sys.stderr
    sys.stdout = tee
    sys.stderr = tee
    try:
        params = {
            "embeddings": str(embeddings),
            "out_dir": str(out_dir),
            "distance": distance,
            "prefix": prefix,
            "pca_whitening": pca_whitening_enabled,
            "pca_components": pca_components,
            "local_scaling": local_scaling_enabled,
            "local_k": local_k,
            "local_k_strategy": local_k_strategy,
            "cutoff_min": cutoff_min,
            "cutoff_max": cutoff_max,
            "cutoff_step": cutoff_step,
            "custom_cutoffs": custom_cutoffs or "",
            "support_mode": support_mode,
            "num_replicates": num_replicates,
            "subsample_ratio": subsample_ratio,
            "support_display_cutoff": support_display_cutoff,
            "save_bootstrap_trees": save_bootstrap_trees_enabled,
            "save_distances": save_distances_enabled,
            "max_distance_pairs": max_distance_pairs,
            "label_csv": str(label_csv) if label_csv is not None else "",
            "metrics_sample_size": metrics_sample_size,
            "cpus": cpus,
            "random_state": random_state,
        }
        print(f"Command: {_format_user_command(ctx, params)}")
        print("Parameters:")
        print(json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True))
        print("-" * 80)

        emb_df = read_csv(embeddings)
        print("[1/7] Loaded embeddings")
        ids = emb_df["id"].astype(str).tolist()
        id_to_sample = None
        if "sample" in emb_df.columns:
            id_to_sample = {
                row_id: str(row_sample)
                for row_id, row_sample in zip(
                    emb_df["id"].astype(str), emb_df["sample"].astype(str)
                )
            }
        dim_cols = [c for c in emb_df.columns if c.startswith("dim_")]
        if not dim_cols:
            raise typer.BadParameter(
                "No embedding dimensions found. Expected columns prefixed with 'dim_'."
            )
        x = emb_df[dim_cols].to_numpy()

        n = len(ids)
        n_pairs = n * (n - 1) // 2
        if n_pairs > max_distance_pairs:
            raise typer.BadParameter(
                f"Pair count {n_pairs} exceeds --max-distance-pairs={max_distance_pairs}. "
                "Increase --max-distance-pairs or reduce input size."
            )

        if pca_whitening_enabled:
            n_comp = min(pca_components, x.shape[1], x.shape[0])
            x, _ = apply_pca_whitening(x, n_components=n_comp)
            print("[2/7] PCA whitening applied")
        else:
            print("[2/7] SKIP PCA whitening")

        if distance == "euclidean":
            x_for_distance = _l2_normalize_rows(x)
            d = compute_euclidean_distances(x_for_distance)
            print(
                "[3/7] Distance matrix computed (euclidean on L2-normalized embeddings)"
            )
        else:
            d = compute_cosine_distances(x)
            print("[3/7] Distance matrix computed")

        if local_scaling_enabled:
            k = (
                local_k
                if local_k > 0
                else auto_select_k(len(ids), strategy=local_k_strategy)
            )
            d, _ = apply_local_scaling(d, k=k)
            print(f"[4/7] Local scaling applied (k={k})")
        else:
            print("[4/7] SKIP local scaling")

        distance_stats_dir = out_dir / "distance_statistics"
        upgma_dir = out_dir / "UPGMA"
        partitions_dir = upgma_dir / "partitions"
        tables_dir = partitions_dir / "tables"
        distance_stats_dir.mkdir(parents=True, exist_ok=True)
        upgma_dir.mkdir(parents=True, exist_ok=True)
        partitions_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        distance_name = f"raw_{distance.lower()}"
        _distance_stats_frame(d, distance_name).to_csv(
            distance_stats_dir / "distance_stats.csv", index=False
        )
        _plot_distance_distributions(d, distance_name, distance_stats_dir)
        print("[5/7] Distance statistics and plots saved")

        if save_distances_enabled:
            pd.DataFrame(d, index=ids, columns=ids).to_csv(
                distance_stats_dir / "distance_matrix.csv"
            )

        z = build_upgma(d)
        tree_name = f"UPGMA_{distance.capitalize()}.nwk"
        upgma_to_newick(z, ids, upgma_dir / tree_name)
        print("[6/7] UPGMA tree built")

        true_labels_arr = None
        ids_for_metrics = None
        labels_for_metrics = None
        id_to_label = None
        x_for_metrics = None
        d_for_metrics = d
        z_metrics = z
        if label_csv is not None:
            lbl_df = read_csv(label_csv)
            if "label" not in lbl_df.columns:
                raise typer.BadParameter("--label-csv must contain a 'label' column.")
            if "id" in lbl_df.columns:
                key_col = "id"
            elif "image" in lbl_df.columns:
                key_col = "image"
            else:
                key_candidates = [c for c in lbl_df.columns if c != "label"]
                if not key_candidates:
                    raise typer.BadParameter(
                        "--label-csv must contain either 'id' or 'image' plus 'label'."
                    )
                key_col = key_candidates[0]

            labels_aligned = lbl_df[[key_col, "label"]].copy()
            labels_aligned.rename(columns={key_col: "id"}, inplace=True)
            labels_aligned["id"] = labels_aligned["id"].astype(str)
            id_to_label = dict(zip(labels_aligned["id"], labels_aligned["label"]))
            merged = emb_df[["id"]].merge(labels_aligned, on="id", how="inner")
            if len(merged) > 0:
                id_to_row = {v: i for i, v in enumerate(ids)}
                merged_idx = [id_to_row[i] for i in merged["id"].astype(str)]
                true_labels_arr = merged["label"].to_numpy()
                ids_for_metrics = merged["id"].astype(str).tolist()
                labels_for_metrics = merged["label"].astype(str).tolist()
                x_for_metrics = x[merged_idx]
                d_for_metrics = d[np.ix_(merged_idx, merged_idx)]
                z_metrics = build_upgma(d_for_metrics)

                if (
                    metrics_sample_size > 0
                    and len(true_labels_arr) > metrics_sample_size
                ):
                    rng = np.random.default_rng(random_state)
                    sample_idx = np.sort(
                        rng.choice(
                            len(true_labels_arr),
                            size=metrics_sample_size,
                            replace=False,
                        )
                    )
                    true_labels_arr = true_labels_arr[sample_idx]
                    x_for_metrics = x_for_metrics[sample_idx]
                    d_for_metrics = d_for_metrics[np.ix_(sample_idx, sample_idx)]
                    z_metrics = build_upgma(d_for_metrics)
            else:
                typer.echo(
                    "[Warning] No overlapping IDs between embeddings and --label-csv."
                )

        if custom_cutoffs:
            cutoffs = [float(v.strip()) for v in custom_cutoffs.split(",") if v.strip()]
        else:
            cutoffs = np.arange(cutoff_min, cutoff_max + 1e-9, cutoff_step)
            cutoffs = np.round(cutoffs, 6).tolist()

        partitions = build_partitions_from_linkage(z, cutoffs)
        partition_scan_rows = [
            {"threshold": float(th), "clusters": int(len(np.unique(labels)))}
            for th, labels in partitions.items()
        ]
        partition_scan_df = pd.DataFrame(partition_scan_rows)
        partition_scan_df.to_csv(partitions_dir / "partition_scan.csv", index=False)
        _plot_partition_scan(partition_scan_df, partitions_dir / "partition_scan.pdf")

        export_partition_tables(
            ids,
            partitions,
            tables_dir,
            prefix=prefix,
            id_to_sample=id_to_sample,
        )
        support = {}
        if num_replicates > 0:
            if support_mode == "subsample" and subsample_ratio >= 1.0:
                print(
                    "  [WARNING] --subsample-ratio >= 1 uses all embedding dimensions; "
                    "bootstrap supports can become trivially high (often 100)."
                )
            if support_mode == "bootstrap":
                print("  Support mode: bootstrap (with replacement over dimensions)")
            else:
                print("  Support mode: subsample (without replacement over dimensions)")
            print(f"  Running {num_replicates} support replicates...")
            support = compute_bootstrap_support(
                x,
                ids,
                z,
                distance=distance,
                support_mode=support_mode,
                n_replicates=num_replicates,
                subsample_ratio=subsample_ratio,
                random_state=random_state,
                save_trees_path=(
                    upgma_dir / "bootstrap_trees.nwk"
                    if save_bootstrap_trees_enabled
                    else None
                ),
                n_jobs=cpus,
            )
            upgma_to_newick(
                z,
                ids,
                upgma_dir / f"UPGMA_{distance.capitalize()}_bootstrap.nwk",
                support_dict=support,
            )

        _plot_upgma_partition_tree_panel(
            z,
            ids,
            partitions,
            partitions_dir / "UPGMA_tree_partitions.pdf",
            support_dict=support,
            bootstrap_cutoff=support_display_cutoff,
        )
        print("[7/7] Partition tables and tree panels saved")

        if true_labels_arr is not None:
            monophyly_prop = _compute_monophyly_proportion_upgma(
                z_metrics,
                ids_for_metrics,
                labels_for_metrics,
            )
            parts_for_metrics = build_partitions_from_linkage(z_metrics, cutoffs)
            metrics = compute_partition_metrics(
                parts_for_metrics,
                true_labels_arr,
                x=x_for_metrics,
                monophyly_proportion=monophyly_prop,
            )
            metrics_rows = [{"cutoff": th, **vals} for th, vals in metrics.items()]
            metrics_df = pd.DataFrame(metrics_rows).sort_values("cutoff")
            metrics_df.to_csv(upgma_dir / "metrics.csv", index=False)
            _plot_metrics_dashboard(metrics_df, upgma_dir / "metrics_dashboard.pdf")

            if id_to_label is not None:
                intra_df = _intra_class_distance_stats(d, ids, id_to_label)
                if not intra_df.empty:
                    intra_df.to_csv(
                        distance_stats_dir
                        / "pairwise_distance_summary_intra-class_raw.csv",
                        index=False,
                    )
            typer.echo(f"Partition metrics saved to: {upgma_dir / 'metrics.csv'}")

        typer.echo(f"Cluster outputs written to: {out_dir}")
    except Exception:
        traceback.print_exc(file=tee)
        raise
    finally:
        sys.stdout = tee.terminal
        sys.stderr = original_stderr
        tee.close()
