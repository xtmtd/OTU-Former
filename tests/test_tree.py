import numpy as np
import pytest

from otuformer.delineation.tree import (
    build_upgma,
    compute_bootstrap_support,
    upgma_to_newick,
)


def simple_dist_matrix():
    return np.array(
        [
            [0.0, 0.1, 0.8, 0.9],
            [0.1, 0.0, 0.7, 0.8],
            [0.8, 0.7, 0.0, 0.1],
            [0.9, 0.8, 0.1, 0.0],
        ]
    )


def test_build_upgma_returns_linkage():
    d = simple_dist_matrix()
    z = build_upgma(d)
    assert z.shape == (3, 4)


def test_upgma_newick_output(tmp_path):
    d = simple_dist_matrix()
    ids = ["A", "B", "C", "D"]
    z = build_upgma(d)
    nwk_path = tmp_path / "tree.nwk"
    upgma_to_newick(z, ids, nwk_path)
    content = nwk_path.read_text()
    assert content.endswith(";")
    for id_ in ids:
        assert id_ in content


def test_upgma_newick_bootstrap_support_labels(tmp_path):
    d = simple_dist_matrix()
    ids = ["A", "B", "C", "D"]
    z = build_upgma(d)
    support = {frozenset({"A", "B"}): 88.2}
    nwk_path = tmp_path / "tree_bootstrap.nwk"
    upgma_to_newick(z, ids, nwk_path, support_dict=support)
    content = nwk_path.read_text()
    assert ")88.2:" in content


def test_compute_bootstrap_support_uses_features_and_writes_trees(tmp_path):
    x = np.array(
        [
            [1.0, 0.9, 0.2, 0.1],
            [1.1, 0.8, 0.2, 0.0],
            [-1.0, -0.9, -0.2, -0.1],
            [-1.1, -0.8, -0.2, 0.0],
        ]
    )
    ids = ["A", "B", "C", "D"]
    from otuformer.delineation.distance import compute_cosine_distances

    z = build_upgma(compute_cosine_distances(x))
    out = tmp_path / "bootstrap_trees.nwk"
    support = compute_bootstrap_support(
        x,
        ids,
        z,
        distance="cosine",
        support_mode="subsample",
        n_replicates=3,
        random_state=1,
        save_trees_path=out,
    )
    assert out.exists()
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == 3
    assert support


def test_build_nj_tree_returns_tree_node():
    skbio = pytest.importorskip("skbio")
    from otuformer.delineation.tree import build_nj_tree
    # 4-OTU toy distance matrix (symmetric, zero diagonal)
    dist = np.array([
        [0.0, 0.3, 0.5, 0.6],
        [0.3, 0.0, 0.4, 0.5],
        [0.5, 0.4, 0.0, 0.2],
        [0.6, 0.5, 0.2, 0.0],
    ])
    ids = ["A", "B", "C", "D"]
    tree = build_nj_tree(dist, ids)
    assert isinstance(tree, skbio.TreeNode)
    tips = {n.name for n in tree.tips()}
    assert tips == set(ids)


def test_build_nj_tree_negative_branch_clipped():
    pytest.importorskip("skbio")
    from otuformer.delineation.tree import build_nj_tree
    # Slightly non-additive matrix that triggers negative branches
    dist = np.array([
        [0.0, 0.1, 0.9, 0.9],
        [0.1, 0.0, 0.9, 0.9],
        [0.9, 0.9, 0.0, 0.1],
        [0.9, 0.9, 0.1, 0.0],
    ])
    ids = ["W", "X", "Y", "Z"]
    tree = build_nj_tree(dist, ids)
    for node in tree.traverse():
        if node.length is not None:
            assert node.length >= 0.0


def test_compute_nj_bootstrap_support_returns_support_dict():
    pytest.importorskip("skbio")
    from otuformer.delineation.tree import build_nj_tree, compute_nj_bootstrap_support
    from otuformer.delineation.distance import compute_cosine_distances
    rng = np.random.default_rng(0)
    c1 = rng.standard_normal(16)
    c2 = -c1
    centroids = np.vstack([
        c1 + rng.standard_normal(16) * 0.01,
        c1 + rng.standard_normal(16) * 0.01,
        c2 + rng.standard_normal(16) * 0.01,
        c2 + rng.standard_normal(16) * 0.01,
    ])
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / norms
    ids = ["OTU_1", "OTU_2", "OTU_3", "OTU_4"]
    support = compute_nj_bootstrap_support(
        centroids, ids, n_replicates=20, random_state=42
    )
    assert isinstance(support, dict)
    assert all(isinstance(k, frozenset) for k in support)
    assert all(0.0 <= v <= 100.0 for v in support.values())


def test_compute_nj_bootstrap_support_saves_trees(tmp_path):
    pytest.importorskip("skbio")
    from otuformer.delineation.tree import compute_nj_bootstrap_support
    from otuformer.delineation.distance import compute_cosine_distances
    rng = np.random.default_rng(1)
    centroids = rng.standard_normal((5, 16))
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / norms
    ids = [f"OTU_{i}" for i in range(5)]
    out = tmp_path / "nj_bootstrap.nwk"
    compute_nj_bootstrap_support(
        centroids, ids, n_replicates=5, random_state=7,
        save_trees_path=out
    )
    assert out.exists()
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == 5


def test_compute_nj_bootstrap_no_base_tree_param():
    """Verify compute_nj_bootstrap_support does NOT accept base_tree param."""
    import inspect
    from otuformer.delineation.tree import compute_nj_bootstrap_support
    sig = inspect.signature(compute_nj_bootstrap_support)
    assert "base_tree" not in sig.parameters
