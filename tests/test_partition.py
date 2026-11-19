import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from otuformer.delineation.partition import (
    build_partitions_from_linkage,
    compute_partition_metrics,
    export_partition_tables,
    two_stage_threshold_scan,
)


def simple_linkage():
    d = np.array(
        [
            [0.0, 0.1, 0.8, 0.9],
            [0.1, 0.0, 0.7, 0.8],
            [0.8, 0.7, 0.0, 0.1],
            [0.9, 0.8, 0.1, 0.0],
        ]
    )
    return hierarchy.linkage(squareform(d, checks=False), method="average"), [
        "A",
        "B",
        "C",
        "D",
    ]


def test_partitions_returned():
    z, ids = simple_linkage()
    parts = build_partitions_from_linkage(z, [0.3, 0.5])
    assert len(parts) == 2
    for _, labels in parts.items():
        assert len(labels) == len(ids)


def test_export_creates_files(tmp_path):
    z, ids = simple_linkage()
    parts = build_partitions_from_linkage(z, [0.5])
    export_partition_tables(ids, parts, tmp_path, prefix="OTU")
    files = list(tmp_path.glob("*.csv"))
    assert len(files) >= 2


def test_partition_prefix_applied(tmp_path):
    z, ids = simple_linkage()
    parts = build_partitions_from_linkage(z, [0.5])
    export_partition_tables(ids, parts, tmp_path, prefix="Dun2024")
    summary = pd.read_csv(next(tmp_path.glob("*summary*")))
    assert all(summary["cluster"].str.startswith("Dun2024"))


def test_partition_metrics_keys():
    z, _ = simple_linkage()
    labels_true = np.array([0, 0, 1, 1])
    parts = build_partitions_from_linkage(z, [0.5])
    metrics = compute_partition_metrics(parts, labels_true)
    assert 0.5 in metrics
    row = metrics[0.5]
    for k in ["NMI", "ARI", "AMI", "BCubed_F", "V_measure"]:
        assert k in row


def test_two_stage_threshold_scan_shapes():
    z, _ = simple_linkage()
    labels_true = np.array([0, 0, 1, 1])
    coarse, fine = two_stage_threshold_scan(
        z,
        labels_true=labels_true,
        coarse_min=0.05,
        coarse_max=1.0,
        coarse_step=0.05,
        fine_step=0.01,
        window_expand=0.15,
    )
    assert len(coarse) > 0
    assert len(fine) > 0


# --- Fixed-output-tree monophyly -----------------------------------------------------

_MONOPHYLY_IDS = [f"e{i}" for i in range(6)]
_MONOPHYLY_LABELS = ["A", "A", "B", "B", "C", "D"]
_MONOPHYLY_DISTANCES = np.array(
    [
        [0.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        [0.9, 0.0, 0.85, 0.75, 0.65, 0.55],
        [0.8, 0.85, 0.0, 0.2, 0.9, 0.95],
        [0.7, 0.75, 0.2, 0.0, 0.88, 0.93],
        [0.6, 0.65, 0.9, 0.88, 0.0, 0.1],
        [0.5, 0.55, 0.95, 0.93, 0.1, 0.0],
    ]
)


def _monophyly_mapping():
    return dict(zip(_MONOPHYLY_IDS, _MONOPHYLY_LABELS))


def test_monophyly_is_measured_on_the_fixed_output_tree():
    from otuformer.cli.cluster import _compute_monophyly_proportion_upgma
    from otuformer.delineation.tree import build_upgma

    fixed = _compute_monophyly_proportion_upgma(
        build_upgma(_MONOPHYLY_DISTANCES), _MONOPHYLY_IDS, _monophyly_mapping()
    )
    assert fixed == 0.75  # B is not a clade on this topology

    # Characterization of the pre-fix call: the sampled linkage was scored with
    # the unsampled id/label list, so leaf positions addressed other specimens.
    # 0.5 is that old wrong value, not expected behavior.
    sample = [2, 4, 5]
    sampled = _compute_monophyly_proportion_upgma(
        build_upgma(_MONOPHYLY_DISTANCES[np.ix_(sample, sample)]),
        _MONOPHYLY_IDS,
        _monophyly_mapping(),
    )
    assert sampled == 0.5


def test_monophyly_scores_only_labeled_tips_and_singletons():
    from otuformer.cli.cluster import _compute_monophyly_proportion_upgma
    from otuformer.delineation.tree import build_upgma

    mapping = {key: value for key, value in _monophyly_mapping().items() if key != "e2"}
    # e2's unlabeled tip does not break B's labeled pair e3 alone: B becomes a
    # singleton (monophyletic) and the denominator counts represented species.
    partial = _compute_monophyly_proportion_upgma(
        build_upgma(_MONOPHYLY_DISTANCES), _MONOPHYLY_IDS, mapping
    )
    assert partial == 0.75

    # No labels at all: there is no score.
    assert (
        _compute_monophyly_proportion_upgma(
            build_upgma(_MONOPHYLY_DISTANCES), _MONOPHYLY_IDS, {}
        )
        is None
    )


def test_monophyly_ignores_a_rebuilt_labeled_tree_topology():
    from otuformer.cli.cluster import _compute_monophyly_proportion_upgma
    from otuformer.delineation.tree import build_upgma

    labeled = [0, 1, 2, 3]
    subset = _MONOPHYLY_DISTANCES[np.ix_(labeled, labeled)]
    # The rebuilt labeled-only tree resolves differently (A stops being a
    # clade), so scoring it would report another value. The fixed output tree is
    # the one that is scored.
    assert _compute_monophyly_proportion_upgma(
        build_upgma(subset), [_MONOPHYLY_IDS[i] for i in labeled], _monophyly_mapping()
    ) == 0.5
    assert _compute_monophyly_proportion_upgma(
        build_upgma(_MONOPHYLY_DISTANCES), _MONOPHYLY_IDS, _monophyly_mapping()
    ) == 0.75
