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


def _toy_embeddings(classes=3, per_class=6, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(classes, dim))
    rows = []
    labels = []
    for index in range(classes):
        rows.append(centres[index] + rng.normal(scale=0.1, size=(per_class, dim)))
        labels.extend([f"class_{index}"] * per_class)
    return np.vstack(rows), np.array(labels)


def test_metric_outcomes_preserve_public_values(monkeypatch):
    import sklearn.model_selection as model_selection

    from otuformer.embedding import evaluator

    embeddings, labels = _toy_embeddings()
    calls = {"cross_val_score": 0, "cross_validate": 0}
    real_cvs = model_selection.cross_val_score
    real_cv = model_selection.cross_validate

    def counting_cvs(*args, **kwargs):
        calls["cross_val_score"] += 1
        return real_cvs(*args, **kwargs)

    def counting_cv(*args, **kwargs):
        calls["cross_validate"] += 1
        return real_cv(*args, **kwargs)

    monkeypatch.setattr(model_selection, "cross_val_score", counting_cvs)
    monkeypatch.setattr(model_selection, "cross_validate", counting_cv)

    knn = evaluator._compute_knn_accuracy_result(embeddings, labels)
    per_run = calls["cross_val_score"]
    assert per_run >= 1
    assert {key: outcome.value for key, outcome in knn.items()} == evaluator.compute_knn_accuracy(
        embeddings, labels
    )
    assert calls["cross_val_score"] == per_run * 2  # one CV per eligible k, no doubles

    probing = evaluator._compute_linear_probing_metrics_result(embeddings, labels)
    assert calls["cross_validate"] == 1  # one shared CV pass for both scores
    assert {
        key: outcome.value for key, outcome in probing.items()
    } == evaluator.compute_linear_probing_metrics(embeddings, labels)
    assert calls["cross_validate"] == 2  # the projection never re-runs diagnostics

    assert all(
        outcome.status == "available" for outcome in (knn["kNN_Acc_k1"], probing["Linear_Probing_Acc"])
    )

    recall = evaluator._compute_recall_at_k_result(embeddings, labels)
    assert {key: outcome.value for key, outcome in recall.items()} == evaluator.compute_recall_at_k(
        embeddings, labels
    )
    assert evaluator._compute_map_result(embeddings, labels).value == evaluator.compute_map(
        embeddings, labels
    )
    clustering = evaluator._compute_clustering_metrics_result(embeddings, labels)
    assert {
        key: outcome.value for key, outcome in clustering.items()
    } == evaluator.compute_clustering_metrics(embeddings, labels)
    assert clustering["NMI"].context["cluster_distance"].startswith("euclidean")
    assert clustering["Silhouette_Score"].context["silhouette_distance"] == "cosine"
    assert recall["Recall@1"].context["self_exclusion"] == "query excluded from candidates"
    assert knn["kNN_Acc_k1"].context["distance"] == "cosine"
    assert probing["Linear_Probing_Acc"].context["classifier"] == "LogisticRegression"


def test_metric_outcome_failure_branches(monkeypatch):
    import sklearn.model_selection as model_selection

    from otuformer.embedding import evaluator

    embeddings, labels = _toy_embeddings()

    one_class = evaluator._compute_knn_accuracy_result(
        embeddings, np.array(["only"] * len(labels))
    )
    assert {outcome.reason for outcome in one_class.values()} == {"insufficient_classes"}
    assert {outcome.value for outcome in one_class.values()} == {None}

    singleton = evaluator._compute_knn_accuracy_result(
        np.eye(2), np.array(["NA", "001"])
    )
    assert singleton["kNN_Acc_k1"].reason == "insufficient_class_count"
    assert evaluator.compute_knn_accuracy(np.eye(2), np.array(["NA", "001"]))[
        "kNN_Acc_k1"
    ] is None

    oversized = evaluator._compute_knn_accuracy_result(embeddings, labels, k_values=[1000])
    assert oversized["kNN_Acc_k1000"].reason == "ineligible_k"

    invalid_k = evaluator._compute_recall_at_k_result(embeddings, labels, k_values=[0])
    assert invalid_k["Recall@0"].reason == "ineligible_k"

    single = evaluator._compute_recall_at_k_result(np.eye(1), np.array(["a"]))
    assert single["Recall@1"].reason == "insufficient_samples"

    unique = evaluator._compute_map_result(embeddings, np.array([str(i) for i in range(len(labels))]))
    assert unique.reason == "no_relevant_pairs"

    collinear = evaluator._compute_clustering_metrics_result(
        np.ones((4, 3)), np.array(["a", "b", "c", "d"])
    )
    assert collinear["NMI"].reason == "insufficient_distinct_vectors"

    def nan_scores(*_args, **_kwargs):
        return np.array([float("nan")] * 3)

    monkeypatch.setattr(model_selection, "cross_val_score", nan_scores)
    nonfinite = evaluator._compute_knn_accuracy_result(embeddings, labels)
    assert nonfinite["kNN_Acc_k1"].reason == "nonfinite_result"

    def boom(*_args, **_kwargs):
        raise RuntimeError("cv exploded")

    monkeypatch.setattr(model_selection, "cross_val_score", boom)
    failed = evaluator._compute_knn_accuracy_result(embeddings, labels)
    assert failed["kNN_Acc_k1"].status == "failed"
    assert failed["kNN_Acc_k1"].reason == "evaluation_failed"
    assert evaluator.compute_knn_accuracy(embeddings, labels)["kNN_Acc_k1"] is None

    monkeypatch.setattr(model_selection, "cross_validate", boom)
    failed_probe = evaluator._compute_linear_probing_metrics_result(embeddings, labels)
    assert failed_probe["Linear_Probing_Acc"].reason == "evaluation_failed"
    assert evaluator.compute_linear_probing_metrics(embeddings, labels)[
        "Linear_Probing_Acc"
    ] is None
