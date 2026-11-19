import numpy as np

from otuformer.delineation.distance import (
    apply_local_scaling,
    apply_pca_whitening,
    compute_cosine_distances,
    compute_euclidean_distances,
    within_group_summary,
)


def random_embeddings(n: int = 20, dim: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float64)


def test_cosine_distances_shape():
    x = random_embeddings()
    d = compute_cosine_distances(x)
    assert d.shape == (20, 20)


def test_cosine_distances_symmetric():
    x = random_embeddings()
    d = compute_cosine_distances(x)
    assert np.allclose(d, d.T, atol=1e-6)


def test_cosine_distances_zero_diagonal():
    x = random_embeddings()
    d = compute_cosine_distances(x)
    assert np.allclose(np.diag(d), 0.0)


def test_cosine_distances_range():
    x = random_embeddings()
    d = compute_cosine_distances(x)
    assert d.min() >= -1e-6
    assert d.max() <= 2.0 + 1e-6


def test_euclidean_distances_shape():
    x = random_embeddings()
    d = compute_euclidean_distances(x)
    assert d.shape == (20, 20)
    assert np.allclose(np.diag(d), 0.0)
    assert np.allclose(d, d.T, atol=1e-6)


def test_local_scaling_changes_distances():
    x = random_embeddings(n=30)
    d = compute_cosine_distances(x)
    d_scaled, _ = apply_local_scaling(d, k=5)
    assert not np.allclose(d, d_scaled)
    assert np.allclose(np.diag(d_scaled), 0.0)


def test_pca_whitening_reduces_dims():
    x = random_embeddings(n=50, dim=64)
    x_pca, _ = apply_pca_whitening(x, n_components=16)
    assert x_pca.shape == (50, 16)


def test_within_group_summary_matches_both_commands():
    """The same-named within-group columns mean one statistic in both commands."""
    import pandas as pd

    from otuformer.cli import cluster as cluster_module
    from otuformer.delineation.annotate import summarize_intra_class_distances

    ids = ["a", "b", "c", "d"]
    mapping = {"a": "one", "b": "one", "c": "two", "d": "two"}
    matrix = np.array(
        [
            [0.0, 0.2, 0.9, 0.8],
            [0.2, 0.0, 0.7, 0.6],
            [0.9, 0.7, 0.0, 0.3],
            [0.8, 0.6, 0.3, 0.0],
        ]
    )

    cluster_frame = cluster_module._intra_class_distance_stats(matrix, ids, mapping)
    annotate_frame = summarize_intra_class_distances(matrix, ids, mapping)
    shared = [
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
    pd.testing.assert_frame_equal(
        cluster_frame[shared].reset_index(drop=True),
        annotate_frame[shared].reset_index(drop=True),
    )
    # The singleton convention is a NaN row with zero pairs in both commands.
    singleton = dict(mapping, d="three")
    pd.testing.assert_frame_equal(
        cluster_module._intra_class_distance_stats(matrix, ids, singleton)[shared]
        .reset_index(drop=True),
        summarize_intra_class_distances(matrix, ids, singleton)[shared]
        .reset_index(drop=True),
    )
    assert within_group_summary(matrix[:2, :2])["n_pairs"] == 1
