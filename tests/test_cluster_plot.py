import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from otuformer.cli.cluster import _plot_corrected_cluster_panel


def test_corrected_cluster_panel_draws_run_counts_and_thicker_bars():
    fig, ax = plt.subplots(figsize=(4, 6))
    ordered_ids = ["a", "b", "c", "d", "e"]
    tip_y_positions = {"a": 5, "b": 15, "c": 25, "d": 35, "e": 45}
    corrected_labels = {
        "a": "OTU_1",
        "b": "OTU_1",
        "c": "OTU_2",
        "d": "OTU_2",
        "e": "OTU_2",
    }

    _plot_corrected_cluster_panel(ax, ordered_ids, tip_y_positions, corrected_labels)

    widths = [p.get_width() for p in ax.patches]
    assert len(widths) == 2
    assert all(w >= 0.06 for w in widths)

    text_values = {t.get_text() for t in ax.texts}
    assert "2" in text_values
    assert "3" in text_values

    plt.close(fig)


def test_corrected_cluster_panel_respects_custom_bar_width():
    fig, ax = plt.subplots(figsize=(4, 6))
    ordered_ids = ["a", "b"]
    tip_y_positions = {"a": 5, "b": 15}
    corrected_labels = {"a": "OTU_1", "b": "OTU_2"}

    _plot_corrected_cluster_panel(
        ax,
        ordered_ids,
        tip_y_positions,
        corrected_labels,
        bar_width=0.12,
    )

    widths = [p.get_width() for p in ax.patches]
    assert widths == [0.12, 0.12]


def test_corrected_cluster_panel_anchors_bars_at_left_edge():
    fig, ax = plt.subplots(figsize=(4, 6))
    ordered_ids = ["a", "b"]
    tip_y_positions = {"a": 5, "b": 15}
    corrected_labels = {"a": "OTU_1", "b": "OTU_2"}

    _plot_corrected_cluster_panel(ax, ordered_ids, tip_y_positions, corrected_labels)

    xs = [p.get_x() for p in ax.patches]
    assert all(x == 0.0 for x in xs)

    plt.close(fig)
