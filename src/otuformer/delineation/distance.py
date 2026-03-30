"""Pairwise distance computation helpers for OTU delineation."""

from __future__ import annotations

import math

import numpy as np
from sklearn.decomposition import PCA


def _safe_l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def compute_cosine_distances(x: np.ndarray, chunk_size: int = 5000) -> np.ndarray:
    """Pairwise cosine distances in [0, 2]."""
    n = x.shape[0]
    if n < 10000:
        x_norm = _safe_l2_normalize(x.astype(np.float64))
        d = 1.0 - x_norm @ x_norm.T
    else:
        x_norm = _safe_l2_normalize(x.astype(np.float32))
        d = np.zeros((n, n), dtype=np.float32)
        n_chunks = (n + chunk_size - 1) // chunk_size
        for i in range(n_chunks):
            si, ei = i * chunk_size, min((i + 1) * chunk_size, n)
            for j in range(i, n_chunks):
                sj, ej = j * chunk_size, min((j + 1) * chunk_size, n)
                block = 1.0 - x_norm[si:ei] @ x_norm[sj:ej].T
                d[si:ei, sj:ej] = block
                if i != j:
                    d[sj:ej, si:ei] = block.T
        d = d.astype(np.float64)

    np.fill_diagonal(d, 0.0)
    d = (d + d.T) / 2.0
    return np.clip(d, 0.0, 2.0)


def compute_euclidean_distances(x: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import cdist

    d = cdist(x.astype(np.float64), x.astype(np.float64), metric="euclidean")
    np.fill_diagonal(d, 0.0)
    return (d + d.T) / 2.0


def apply_pca_whitening(
    x: np.ndarray,
    n_components: int,
    random_state: int = 42,
) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components, whiten=True, random_state=random_state)
    x_pca = pca.fit_transform(x)
    return x_pca, pca


def apply_local_scaling(
    dist_matrix: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    knn_dists = np.partition(dist_matrix, k, axis=1)[:, k]
    scale = np.sqrt(np.outer(knn_dists, knn_dists))
    scale = np.maximum(scale, 1e-10)
    scaled = dist_matrix / scale
    np.fill_diagonal(scaled, 0.0)
    scaled = (scaled + scaled.T) / 2.0
    return scaled, knn_dists


def auto_select_k(n_samples: int, strategy: str = "adaptive") -> int:
    if strategy == "fixed":
        k = 7
    elif strategy == "sqrt":
        k = int(math.sqrt(n_samples))
    elif strategy == "log":
        k = max(5, int(5 * math.log2(n_samples)))
    else:
        if n_samples < 100:
            k = 7
        elif n_samples < 500:
            k = int(math.sqrt(n_samples))
        else:
            k = int(5 * math.log2(n_samples))
        k = max(5, min(k, 100))
    return min(k, n_samples - 1)
