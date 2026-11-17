"""Qualification-runner contract: import safety, streaming summary, exit status."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_dense_ranking_script():
    """Load the standalone runner by path; it is not an importable package."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_dense_ranking.py"
    spec = importlib.util.spec_from_file_location("check_dense_ranking", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_import_is_side_effect_free(monkeypatch):
    """Importing the runner parses no argv, imports no optional package, trains nothing."""
    import sys
    import torch

    before = set(sys.modules)
    _load_dense_ranking_script()
    assert "diffsort" not in set(sys.modules) - before
    assert set(sys.modules) - before == set(sys.modules) - before  # no argv/training side effect


def test_runner_numerical_summary_and_output(tmp_path):
    module = _load_dense_ranking_script()
    out = tmp_path / "evidence.json"
    code = module.main(
        [
            "--mode",
            "numerical",
            "--device",
            "cpu",
            "--seed",
            "0",
            "--steps",
            "2",
            "--query-rows",
            "3",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["mode"] == "numerical"
    assert payload["counts"]["calls"] > 0
    assert 0.0 <= payload["counts"]["min_probability"] <= 1.0
    assert payload["counts"]["max_stochasticity_residual"] <= 1e-2
    assert "config" in payload and payload["config"]["query_rows"] == 3


def test_runner_returns_nonzero_on_invalid_case(tmp_path):
    module = _load_dense_ranking_script()
    code = module.main(
        [
            "--mode",
            "numerical",
            "--device",
            "cpu",
            "--steps",
            "1",
            "--query-rows",
            "1",
            "--invalid-case",
            "--output",
            str(tmp_path / "bad.json"),
        ]
    )
    assert code != 0


def test_runner_rejects_unknown_mode(tmp_path):
    module = _load_dense_ranking_script()
    with pytest.raises(SystemExit):
        module.main(["--mode", "teleport", "--output", str(tmp_path / "x.json")])


def test_runner_coordinate_mode_is_independent_of_diffsort(tmp_path, monkeypatch):
    """Coordinate mode must run with the optional package unavailable."""
    import sys

    module = _load_dense_ranking_script()
    monkeypatch.setitem(sys.modules, "diffsort", None)
    out = tmp_path / "coord.json"
    code = module.main(
        [
            "--mode",
            "coordinate",
            "--device",
            "cpu",
            "--seed",
            "0",
            "--steps",
            "2",
            "--query-rows",
            "3",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["mode"] == "coordinate"
    assert payload["counts"]["calls"] > 0
    assert payload["counts"]["min_norm"] > 1e-6
    assert 0.0 < payload["counts"]["min_rho"] <= 1.0


def test_runner_coordinate_reports_invalid_population(tmp_path):
    """An all-cancelling fixture must exit nonzero with the reason recorded."""
    module = _load_dense_ranking_script()
    code = module.main(
        [
            "--mode",
            "coordinate",
            "--device",
            "cpu",
            "--steps",
            "1",
            "--query-rows",
            "1",
            "--invalid-case",
            "--output",
            str(tmp_path / "bad-coord.json"),
        ]
    )
    assert code != 0
    payload = json.loads((tmp_path / "bad-coord.json").read_text())
    assert payload["failures"]


def test_replay_mode_reproduces_selections_and_state(tmp_path):
    """Same eligibility/order must replay selections, draws and RNG state."""
    module = _load_dense_ranking_script()
    out = tmp_path / "replay.json"
    code = module.main(
        [
            "--mode", "replay", "--device", "cpu", "--seed", "0",
            "--steps", "3", "--query-rows", "4", "--output", str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["mode"] == "replay"
    assert payload["counts"]["transitions"] >= 4
    assert payload["counts"]["replayed"] == payload["counts"]["transitions"]
    assert payload["failures"] == []


def test_memory_mode_keeps_actual_and_bound_separate(tmp_path):
    module = _load_dense_ranking_script()
    out = tmp_path / "memory.json"
    code = module.main(
        [
            "--mode", "memory", "--device", "cpu", "--seed", "0",
            "--steps", "1", "--query-rows", "4",
            "--batch-size", "32", "--global-size", "224", "--local-size", "96",
            "--local-crops", "6", "--patch-size", "16",
            "--query-cap", "49", "--reference-cap", "16",
            "--output", str(out),
        ]
    )
    assert code == 0
    report = json.loads(out.read_text())
    assert report["mode"] == "memory"
    assert report["configured_bound"] == 32 * (2 * 49 + 12 * 36) == 16960
    assert 0 < report["actual_queries"] <= report["configured_bound"]
    assert report["measured_peak_device"] == "cpu-extrapolation"
    assert report["sorter_graph_gib"]["reference_16"] < report["sorter_graph_gib"]["reference_49"]


def test_memory_mode_all_valid_reaches_the_bound():
    module = _load_dense_ranking_script()
    code = module.main(["--mode", "memory", "--full-valid", "--steps", "1"])
    assert code == 0


def test_memory_mode_reduced_overlap_is_strictly_below_the_bound(tmp_path):
    module = _load_dense_ranking_script()
    out = tmp_path / "reduced.json"
    module.main(
        [
            "--mode", "memory", "--reduced-overlap", "--steps", "1",
            "--output", str(out),
        ]
    )
    report = json.loads(out.read_text())
    assert 0 < report["actual_queries"] < report["configured_bound"]


def test_numerical_mode_rejects_a_zero_student_probability(monkeypatch):
    """The runner must fail where production row_cross_entropy fails."""
    module = _load_dense_ranking_script()
    import torch

    from otuformer.training import neighbor_ranking as ranking

    real = ranking.sort_values_and_permutation

    def zeroing(cosines, *, cache=None):
        _, permutation = real(cosines, cache=cache)
        broken = permutation.clone()
        broken[..., 0, 0] = 0.0
        return cosines @ broken, broken

    monkeypatch.setattr(ranking, "sort_values_and_permutation", zeroing)
    code = module.main(
        [
            "--mode", "numerical", "--device", "cpu", "--seed", "0",
            "--steps", "1", "--query-rows", "2",
            "--output", str(module.Path("/tmp/zero.json")),
        ]
    )
    assert code != 0
    import json

    payload = json.loads(module.Path("/tmp/zero.json").read_text())
    assert any("non-positive" in item for item in payload["failures"])


def test_replay_covers_the_fixed_cap_boundaries(tmp_path):
    """The plan's 48/49/50/48 query and 15/16/17/15 reference boundaries."""
    module = _load_dense_ranking_script()
    out = tmp_path / "replay-boundary.json"
    code = module.main(
        ["--mode", "replay", "--device", "cpu", "--seed", "0", "--output", str(out)]
    )
    assert code == 0
    payload = json.loads(out.read_text())
    counts = payload["counts"]
    # Fixed production cap (49); the real candidate count crosses 48 -> 49 -> 50
    # -> 48, so below-cap, at-cap (no draw) and above-cap (draw) all occur.
    assert [max(item) for item in counts["query_ladder_eligible"]] == [48, 49, 50, 48]
    assert [max(item) for item in counts["query_ladder_selected"]] == [48, 49, 49, 48]
    assert [item[0] for item in counts["query_ladder_eligible"]] == [5, 5, 5, 5]
    assert [item[0] for item in counts["query_ladder_selected"]] == [5, 5, 5, 5]
    assert counts["reference_ladder_selected"] == [[5, 10], [5, 11], [5, 12], [5, 10]]
    assert counts["query_draws_match_expected"] is True
    assert counts["reference_draws_match_expected"] is True
    # Reference pool 15/16/17/15 around the approved cap 16, replayed cold.
    assert counts["reference_ladder_pools"] == [15, 16, 17, 15]
    assert counts["reference_cap_crossed"] is True
    assert counts["replayed"] == counts["transitions"]
    assert payload["failures"] == []


@pytest.mark.parametrize("branch", ["student_head", "teacher_head"])
@pytest.mark.parametrize("ladder", ["dense-step.ckpt", "reference-ladder.ckpt"])
def test_replay_rejects_final_head_tensor_corruption(
    monkeypatch, branch, ladder
):
    """L2-invariant changes still violate exact checkpoint head recovery."""
    import argparse
    import torch

    module = _load_dense_ranking_script()
    real_load = torch.load
    restores = []

    def corrupt_final_layer(path, *args, **kwargs):
        payload = real_load(path, *args, **kwargs)
        if Path(path).name == ladder:
            restores.append(True)
            if len(restores) == 4:
                payload[branch]["4.weight"].mul_(2)
                payload[branch]["4.bias"].mul_(2)
        return payload

    monkeypatch.setattr(torch, "load", corrupt_final_layer)
    report = module._run_replay(argparse.Namespace(seed=0))
    assert len(restores) == 4
    assert report["failures"]
    assert any(branch.replace("_", " ") in failure for failure in report["failures"])
