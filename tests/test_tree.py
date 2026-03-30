import numpy as np

from otuformer.delineation.tree import build_upgma, upgma_to_newick


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
