import numpy as np

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
