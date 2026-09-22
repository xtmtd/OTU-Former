import sys
import types
import warnings

import numpy as np
import pytest

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


def test_knn_accuracy_skips_unsupported_k_without_warning():
    embs = np.eye(8, dtype=float)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_knn_accuracy(embs, labels, k_values=[1, 20])

    assert isinstance(result["kNN_Acc_k1"], float)
    assert result["kNN_Acc_k20"] is None
    assert not caught


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
    for key in ["NMI", "ARI", "AMI", "Silhouette_Score", "Purity"]:
        assert isinstance(result[key], float)


def test_knn_uses_five_fold_shuffled_stratified_cv(monkeypatch):
    import sklearn.model_selection as skms

    captured = {}

    def fake_cross_val_score(estimator, X, y, cv=None, **kwargs):
        captured["cv"] = cv
        return np.array([0.5])

    monkeypatch.setattr(skms, "cross_val_score", fake_cross_val_score)
    embs = np.random.default_rng(0).standard_normal((20, 8))
    labels = np.array([0] * 10 + [1] * 10)

    compute_knn_accuracy(embs, labels, k_values=[1])

    cv = captured["cv"]
    assert isinstance(cv, skms.StratifiedKFold)
    assert cv.n_splits == 5
    assert cv.shuffle is True
    assert cv.random_state == 42


def test_linear_probing_metrics_use_five_fold_shuffled_stratified_cv(monkeypatch):
    import sklearn.model_selection as skms

    from otuformer.embedding.evaluator import compute_linear_probing_metrics

    captured = {}

    def fake_cross_validate(estimator, X, y, cv=None, scoring=None, **kwargs):
        captured["cv"] = cv
        captured["scoring"] = scoring
        return {
            "test_accuracy": np.array([0.8]),
            "test_balanced_accuracy": np.array([0.7]),
        }

    monkeypatch.setattr(skms, "cross_validate", fake_cross_validate)
    embs = np.random.default_rng(0).standard_normal((20, 8))
    labels = np.array([0] * 10 + [1] * 10)

    result = compute_linear_probing_metrics(embs, labels)

    cv = captured["cv"]
    assert isinstance(cv, skms.StratifiedKFold)
    assert cv.n_splits == 5
    assert cv.shuffle is True
    assert cv.random_state == 42
    assert captured["scoring"] == {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
    }
    assert result["Linear_Probing_Acc"] == pytest.approx(0.8)
    assert result["Linear_Probing_Balanced_Acc"] == pytest.approx(0.7)


def test_singleton_class_returns_unavailable_not_zero():
    from otuformer.embedding.evaluator import compute_linear_probing_metrics

    embs = np.random.default_rng(0).standard_normal((6, 4))
    labels = np.array([0, 0, 0, 0, 0, 1])

    knn = compute_knn_accuracy(embs, labels, k_values=[1, 5, 20])
    assert knn == {"kNN_Acc_k1": None, "kNN_Acc_k5": None, "kNN_Acc_k20": None}

    probe = compute_linear_probing_metrics(embs, labels)
    assert probe["Linear_Probing_Acc"] is None
    assert probe["Linear_Probing_Balanced_Acc"] is None


def test_small_training_fold_makes_high_k_unavailable():
    embs = np.random.default_rng(1).standard_normal((10, 6))
    labels = np.array([0] * 5 + [1] * 5)

    result = compute_knn_accuracy(embs, labels, k_values=[1, 5, 20])

    assert isinstance(result["kNN_Acc_k1"], float)
    assert result["kNN_Acc_k20"] is None


def test_linear_probing_metrics_exposes_ordinary_and_balanced_accuracy():
    from otuformer.embedding.evaluator import compute_linear_probing_metrics

    embs, labels = make_data(n=50, dim=32, n_classes=5)

    result = compute_linear_probing_metrics(embs, labels)

    assert set(result) == {"Linear_Probing_Acc", "Linear_Probing_Balanced_Acc"}
    for value in result.values():
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0


def test_map_ranks_only_non_query_samples():
    embs = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    labels = np.array([0, 0, 1])

    assert compute_map(embs, labels) == pytest.approx(1.0)


def test_map_unavailable_when_no_non_query_relevant_item():
    embs = np.eye(2)
    labels = np.array([0, 1])

    assert compute_map(embs, labels) is None


def test_recall_unavailable_for_single_sample_and_oversized_k():
    single = np.array([[1.0, 0.0]])
    assert compute_recall_at_k(single, np.array([0]), k_values=[1])["Recall@1"] is None

    embs = np.random.default_rng(2).standard_normal((5, 4))
    labels = np.array([0, 0, 1, 1, 1])

    result = compute_recall_at_k(embs, labels, k_values=[1, 10, 20])

    assert isinstance(result["Recall@1"], float)
    assert result["Recall@10"] is None
    assert result["Recall@20"] is None


def test_clustering_unavailable_when_fewer_distinct_embeddings_than_classes():
    embs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1, 2, 3])

    result = compute_clustering_metrics(embs, labels)

    assert all(value is None for value in result.values())


def test_non_finite_fold_scores_return_unavailable(monkeypatch):
    import sklearn.model_selection as skms

    from otuformer.embedding.evaluator import compute_linear_probing_metrics

    partial_nan = np.array([np.nan, 0.9, 0.9, 0.9, 0.9])
    monkeypatch.setattr(skms, "cross_val_score", lambda *args, **kwargs: partial_nan)
    monkeypatch.setattr(
        skms,
        "cross_validate",
        lambda *args, **kwargs: {
            "test_accuracy": partial_nan,
            "test_balanced_accuracy": partial_nan,
        },
    )
    embs = np.random.default_rng(0).standard_normal((20, 8))
    labels = np.array([0] * 10 + [1] * 10)

    knn = compute_knn_accuracy(embs, labels, k_values=[1])
    probe = compute_linear_probing_metrics(embs, labels)

    assert knn["kNN_Acc_k1"] is None
    assert probe["Linear_Probing_Acc"] is None
    assert probe["Linear_Probing_Balanced_Acc"] is None


def test_cv_call_sites_raise_instead_of_scoring_failed_folds(monkeypatch):
    import sklearn.model_selection as skms

    from otuformer.embedding.evaluator import compute_linear_probing_metrics

    captured = {}

    def fake_cross_val_score(*args, **kwargs):
        captured["cross_val_score"] = kwargs
        return np.array([0.5])

    def fake_cross_validate(*args, **kwargs):
        captured["cross_validate"] = kwargs
        return {
            "test_accuracy": np.array([0.5]),
            "test_balanced_accuracy": np.array([0.5]),
        }

    monkeypatch.setattr(skms, "cross_val_score", fake_cross_val_score)
    monkeypatch.setattr(skms, "cross_validate", fake_cross_validate)
    embs = np.random.default_rng(0).standard_normal((20, 8))
    labels = np.array([0] * 10 + [1] * 10)

    compute_knn_accuracy(embs, labels, k_values=[1])
    compute_linear_probing_metrics(embs, labels)

    assert captured["cross_val_score"].get("error_score") == "raise"
    assert captured["cross_validate"].get("error_score") == "raise"


def test_run_umap_caps_neighbors_to_sample_count(tmp_path, monkeypatch):
    seen = {}

    class FakeUMAP:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def fit_transform(self, x):
            return np.zeros((len(x), 2))

    umap_module = types.ModuleType("umap")
    umap_submodule = types.ModuleType("umap.umap_")
    umap_submodule.UMAP = FakeUMAP
    monkeypatch.setitem(sys.modules, "umap", umap_module)
    monkeypatch.setitem(sys.modules, "umap.umap_", umap_submodule)
    monkeypatch.setattr("matplotlib.pyplot.savefig", lambda *args, **kwargs: None)

    from otuformer.embedding.evaluator import run_umap

    run_umap(np.eye(4), None, tmp_path / "umap.pdf", n_neighbors=15)

    assert seen["n_neighbors"] == 3


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
