import numpy as np

from otuformer.embedding.evaluator import (
    compute_clustering_metrics,
    compute_knn_accuracy,
    compute_map,
    compute_metric_learning_diagnostics,
    compute_recall_at_k,
)


def make_data(n: int = 50, dim: int = 32, n_classes: int = 5):
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(n_classes), n // n_classes)
    embeddings = rng.standard_normal((n, dim))
    for c in range(n_classes):
        mask = labels == c
        embeddings[mask] += rng.standard_normal(dim) * 3
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings, labels


def test_knn_accuracy_returns_dict():
    embs, labels = make_data()
    result = compute_knn_accuracy(embs, labels, k_values=[1, 5])
    assert "kNN_Acc_k1" in result
    assert "kNN_Acc_k5" in result
    assert 0.0 <= result["kNN_Acc_k1"] <= 1.0


def test_recall_at_k_returns_dict():
    embs, labels = make_data()
    result = compute_recall_at_k(embs, labels, k_values=[1, 5, 10])
    assert "Recall@1" in result
    assert 0.0 <= result["Recall@1"] <= 1.0


def test_map_range():
    embs, labels = make_data()
    score = compute_map(embs, labels)
    assert 0.0 <= score <= 1.0


def test_clustering_metrics_keys():
    embs, labels = make_data()
    result = compute_clustering_metrics(embs, labels)
    for key in ["NMI", "ARI", "AMI", "Silhouette", "Purity"]:
        assert key in result


def test_metric_learning_diagnostics_keys():
    embs, labels = make_data()
    result = compute_metric_learning_diagnostics(embs, labels)
    for key in [
        "Intra_Class_Var",
        "Inter_Class_Dist",
        "Embedding_Norm_Mean",
        "Embedding_Norm_Std",
    ]:
        assert key in result
