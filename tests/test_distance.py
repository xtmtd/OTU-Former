import numpy as np

from otuformer.delineation.distance import (
    apply_local_scaling,
    apply_pca_whitening,
    compute_cosine_distances,
    compute_euclidean_distances,
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
