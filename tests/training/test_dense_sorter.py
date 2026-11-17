"""Optional differentiable sorter adapter: orientation, guards, cache, laziness."""

from __future__ import annotations

import random
import sys

import numpy as np
import pytest
import torch

from otuformer.training import neighbor_ranking as ranking


@pytest.fixture()
def rng_states():
    torch.manual_seed(7)
    return (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
    )


def test_sorter_orientation_row_ce():
    """P[j,k] is input j at sorted rank k; the loss is the row-CE mean."""
    cosines = torch.tensor([[0.4, -0.7, 0.9, -0.2]], dtype=torch.float32)
    sorted_c, P = ranking.sort_values_and_permutation(cosines, cache={})

    # Row-vector times matrix, per query row: the design's `sorted = c @ P`.
    applied = torch.bmm(cosines.unsqueeze(1), P).squeeze(1)
    torch.testing.assert_close(applied, sorted_c, rtol=0, atol=1e-6)
    assert P[0].argmax(dim=0).tolist() == [1, 3, 0, 2]
    assert (P.sum(-1) - 1).abs().max() <= 1e-5

    reference = torch.tensor([[0.1, 0.5, -0.3, 0.7]], dtype=torch.float32)
    _, P_t = ranking.sort_values_and_permutation(reference, cache={})
    cells = -(P_t.detach() * P.clamp_min(torch.finfo(P.dtype).tiny).log())
    row_ce = cells.sum(-1).sum(-1) / 4.0  # per query, the row-CE mean
    # The all-cell mean is smaller by exactly the factor R for one query.
    torch.testing.assert_close(row_ce[0], 4.0 * cells.mean(), rtol=1e-5, atol=1e-5)
    assert not torch.allclose(row_ce[0], cells.mean())
    torch.testing.assert_close(row_ce, ranking.row_cross_entropy(P, P_t), rtol=1e-5, atol=1e-5)


def test_sorter_counts_and_guards():
    """Every supported R sorts, stays stochastic and keeps gradients finite."""
    torch.manual_seed(11)
    cache: dict = {}
    for R in range(2, 50):
        cosines = torch.rand(3, R, dtype=torch.float32) * 2.0 - 1.0
        cosines.requires_grad_(True)
        sorted_c, P = ranking.sort_values_and_permutation(cosines, cache=cache)
        assert torch.isfinite(P).all()
        assert P.min() >= 0.0
        assert (P.sum(-1) - 1).abs().max() <= 1e-5
        assert (P.sum(-2) - 1).abs().max() <= 1e-5
        loss = -(P.clamp_min(torch.finfo(P.dtype).tiny).log()).sum() / R
        loss.backward()
        assert cosines.grad is not None and torch.isfinite(cosines.grad).all()

    for R in (2, 3, 4, 7, 16, 17, 31, 49):
        tied = torch.zeros(2, R, dtype=torch.float32)
        _, P = ranking.sort_values_and_permutation(tied, cache={})
        assert (P.sum(-1) - 1).abs().max() <= 1e-5
        extreme = torch.tensor(
            [[-1.0] * (R - 1) + [1.0], [1.0] + [-1.0] * (R - 1)], dtype=torch.float32
        )
        _, P_extreme = ranking.sort_values_and_permutation(extreme, cache={})
        assert torch.isfinite(P_extreme).all() and P_extreme.min() >= 0.0


def test_probe_is_lazy_and_preserves_rng(rng_states, monkeypatch):
    """Probing touches no global RNG and imports the optional package lazily."""
    random_state, np_state, torch_state = rng_states
    monkeypatch.delitem(sys.modules, "diffsort", raising=False)
    assert "diffsort" not in sys.modules

    info = ranking.probe_ranking_dependency(torch.device("cpu"))

    assert info["name"] == "diffsort"
    assert info["version"] == "0.2.0"
    assert set(info["source_sha256"]) == {
        "diffsort/diffsort.py",
        "diffsort/functional.py",
        "diffsort/networks.py",
    }
    assert info["device"] == "cpu"
    assert random.getstate() == random_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_sorter_source_identity_and_lru(monkeypatch):
    """A modified same-version implementation is rejected; the cache stays bounded."""
    real = ranking._source_digests()
    monkeypatch.setattr(
        ranking, "_source_digests", lambda: {**real, "diffsort/networks.py": "0" * 64}
    )
    with pytest.raises(ranking.RankingDependencyError, match="source"):
        ranking.probe_ranking_dependency(torch.device("cpu"))

    monkeypatch.undo()
    torch.manual_seed(3)
    cache: dict = {}
    for R in (2, 3, 7, 16, 49, 2):
        cosines = torch.rand(1, R, dtype=torch.float32, requires_grad=True)
        ranking.sort_values_and_permutation(cosines, cache=cache)
        assert len(cache) <= 4
        for sorter in cache.values():
            for layer in sorter.sorting_network:
                for tensor in layer:
                    assert not tensor.requires_grad
                    assert tensor.grad_fn is None

    # An evicted and rebuilt sorter reproduces outputs and gradients.
    R = 7
    cosines = torch.rand(2, R, dtype=torch.float32)
    first, first_P = ranking.sort_values_and_permutation(cosines, cache={})
    cache.clear()
    second, second_P = ranking.sort_values_and_permutation(cosines, cache={})
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first_P, second_P, rtol=0, atol=0)

    grad_a = torch.rand(2, R, dtype=torch.float32, requires_grad=True)
    loss_a = ranking.row_cross_entropy(
        ranking.sort_values_and_permutation(grad_a, cache={})[1], first_P
    ).sum()
    loss_a.backward()
    grad_b = grad_a.detach().clone().requires_grad_(True)
    ranking.sort_values_and_permutation(grad_b, cache={})[1].sum().backward()
    assert torch.isfinite(grad_a.grad).all() and torch.isfinite(grad_b.grad).all()


def test_sorter_rejects_unsupported_reference_counts():
    for bad in (0, 1, 50, 64):
        with pytest.raises(ranking.RankingDependencyError):
            ranking.sort_values_and_permutation(torch.zeros(1, bad), cache={})


def test_sorter_rejects_nonfinite_cosines():
    cosines = torch.tensor([[0.1, float("nan")]], dtype=torch.float32)
    with pytest.raises(ranking.RankingDependencyError):
        ranking.sort_values_and_permutation(cosines, cache={})


def test_source_digests_accept_a_source_readable_zip_loader(monkeypatch, tmp_path):
    """A loader that can serve source bytes must not be rejected as "missing"."""
    import hashlib
    import sys
    import zipfile

    from otuformer.training import neighbor_ranking as ranking

    archive = tmp_path / "diffsort.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for relative in ranking.QUALIFIED_SOURCE_SHA256:
            zf.writestr(relative, "SOURCE-" + relative)

    class _ZipLoader:
        def get_source(self, name):
            stem = name.rsplit(".", 1)[-1]
            for relative in ranking.QUALIFIED_SOURCE_SHA256:
                if relative.rsplit("/", 1)[-1].split(".")[0] == stem:
                    return f"SOURCE-{relative}"
            raise ImportError(name)

    class _Spec:
        loader = _ZipLoader()

    class _Module:
        __file__ = str(archive)
        __spec__ = _Spec()

    for name in (
        "diffsort",
        "diffsort.diffsort",
        "diffsort.functional",
        "diffsort.networks",
    ):
        monkeypatch.setitem(sys.modules, name, _Module())
    digests = ranking._source_digests()
    for relative, digest in digests.items():
        assert digest == hashlib.sha256(f"SOURCE-{relative}".encode()).hexdigest()
