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

# Six unit vectors whose cosine distance matrix separates two labeled species in the
# output tree but not in a tree rebuilt from the labeled rows alone.
_MONOPHYLY_EMBEDDINGS = np.array(
    [
        [-0.458690, 0.873981, -0.160498],
        [-0.320069, 0.932894, -0.165121],
        [-0.261424, -0.897887, 0.354197],
        [0.246504, 0.508374, 0.825101],
        [0.375613, 0.027986, 0.926354],
        [-0.855399, -0.081946, 0.511447],
    ]
)
_MONOPHYLY_IDS = [f"e{index}" for index in range(6)]
_MONOPHYLY_LABELS = {"e0": "x", "e1": "x", "e2": "y", "e3": "y"}
_MONOPHYLY_LABELED_ROWS = [0, 1, 2, 3]


def _monophyly_distances():
    from otuformer.delineation.distance import compute_cosine_distances

    return compute_cosine_distances(_MONOPHYLY_EMBEDDINGS.copy())


def _labeled_only_monophyly(z, ids, labels):
    """The pre-fix rule, reimplemented here as an independent reference.

    It scores whatever topology it is handed while reading leaf positions with the
    caller's full id list, so a tree built only from the labeled rows answers a
    different question from the output tree.
    """
    id_to_label = dict(zip(ids, labels))
    species = set(labels)
    clades = []

    def collect(node):
        if node.left is None and node.right is None:
            return {ids[node.id]}
        clade = collect(node.left) | collect(node.right)
        clades.append(clade)
        return clade

    collect(hierarchy.to_tree(z, rd=False))
    hits = 0
    for name in species:
        tips = {tip for tip, label in id_to_label.items() if label == name}
        if len(tips) <= 1 or any(clade == tips for clade in clades):
            hits += 1
    return hits / len(species)


def test_monophyly_is_scored_on_the_output_tree_not_a_labeled_tree():
    from otuformer.delineation.partition import compute_monophyly_proportion

    distances = _monophyly_distances()
    output_tree = hierarchy.linkage(
        squareform(distances, checks=False), method="average"
    )
    assert compute_monophyly_proportion(
        output_tree, _MONOPHYLY_IDS, _MONOPHYLY_LABELS
    ) == 1.0

    # Characterization of the pre-fix basis: a tree rebuilt from the labeled rows
    # alone answers a different question and returns a different value.
    submatrix = distances[np.ix_(_MONOPHYLY_LABELED_ROWS, _MONOPHYLY_LABELED_ROWS)]
    labeled_tree = hierarchy.linkage(squareform(submatrix, checks=False), method="average")
    labeled_ids = [_MONOPHYLY_IDS[index] for index in _MONOPHYLY_LABELED_ROWS]
    assert (
        _labeled_only_monophyly(labeled_tree, labeled_ids, ["x", "x", "y", "y"])
        == 0.5
    )


def test_monophyly_treats_unlabeled_tips_as_neutral_and_counts_singletons():
    from otuformer.delineation.partition import compute_monophyly_proportion

    # Two x tips are joined through a clade that also holds the unlabeled e4: the
    # unlabeled tip must neither join nor break the species.
    distances = np.array(
        [
            [0.0, 0.1, 0.9, 0.9, 0.15],
            [0.1, 0.0, 0.9, 0.9, 0.12],
            [0.9, 0.9, 0.0, 0.2, 0.95],
            [0.9, 0.9, 0.2, 0.0, 0.95],
            [0.15, 0.12, 0.95, 0.95, 0.0],
        ]
    )
    tree = hierarchy.linkage(squareform(distances, checks=False), method="average")
    ids = ["a", "b", "c", "d", "u"]
    labels = {"a": "x", "b": "x", "c": "y", "d": "y"}

    assert compute_monophyly_proportion(tree, ids, labels) == 1.0
    # e4 is labeled as an x tip here: the unlabeled-tip neutrality disappears and
    # the species only counts if its labeled tips are a clade.
    assert compute_monophyly_proportion(tree, ids, {**labels, "u": "y"}) == 0.5
    assert compute_monophyly_proportion(tree, ids, {}) is None


def test_select_labeled_cohort_matches_the_seeded_sorted_cap():
    from otuformer.delineation.partition import select_labeled_cohort

    label_table = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "zz"],
            "label": ["x", "x", "y", "y", "z"],
        }
    )
    ids = ["a", "b", "c", "d", "e"]

    rows, labels = select_labeled_cohort(
        ids, label_table, max_samples=2, seed=42
    )
    expected = np.sort(np.random.default_rng(42).choice(4, size=2, replace=False))
    assert rows.tolist() == expected.tolist()
    assert labels.tolist() == [["x", "x", "y", "y"][index] for index in expected]

    # A nonpositive cap, or one at least as large as the cohort, keeps every row.
    rows, labels = select_labeled_cohort(ids, label_table, max_samples=0, seed=42)
    assert rows.tolist() == [0, 1, 2, 3]
    assert labels.tolist() == ["x", "x", "y", "y"]
    rows, _ = select_labeled_cohort(ids, label_table, max_samples=4, seed=42)
    assert rows.tolist() == [0, 1, 2, 3]
