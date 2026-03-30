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
