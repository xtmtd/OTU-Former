#!/usr/bin/env python3
"""Explicit numerical/replay/memory qualification runner for dense objectives.

Standalone by design: this file is loaded by path from tests, is not an
importable package, and performs no work at import time. It never installs
dependencies, never changes production defaults and never launches training.

Modes: ``numerical`` (optional sorter), ``coordinate`` (norm/rho guards),
``replay`` (cap-crossing selection/loss/state replay through the production
dense lifecycle) and ``memory`` (query/reference budget accounting from real
geometry and, separately, plainly labelled assumptions).

Every report distinguishes three kinds of number:

* ``measured`` — produced by running the production code on this host;
* ``computed`` — deterministic arithmetic over a real configuration;
* ``assumed``  — a placeholder scenario that no geometry produced.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

# Keep the sweep inside the frozen recipe bounds so the runner cannot advertise
# reference counts the objective would reject.
REFERENCE_SWEEP = tuple(range(2, 50))

#: Fixed Q49/R16 replay. Row A keeps 5 candidates; row B crosses query48/49/50/48.
#: A separate two-row ladder selects (5,10)/(5,11)/(5,12)/(5,10), so canonical
#: reference pools cross 15/16/17/15 and only the above-cap step consumes draws.
REPLAY_QUERY_LADDER = ((49, 16), (49, 16), (49, 16), (49, 16))
#: The production query cap the ladder holds fixed.
REPLAY_FIXED_QUERY_CAP = 49
REPLAY_REFERENCE_LADDER = REPLAY_QUERY_LADDER
REPLAY_LADDER = REPLAY_QUERY_LADDER
REPLAY_QUERY_CANDIDATES = (48, 49, 50, 48)
REPLAY_REFERENCE_POOL = (15, 16, 17, 15)

#: The step executed after the final save/restore boundary, so the restored head,
#: optimizer moments and generator state must drive a real update.
REPLAY_TAIL_STEP = (8, 16)
#: The same tail for the reference ladder: a pool of 16 (cap 16) after restore.
REPLAY_REFERENCE_TAIL_STEP = (49, 16)

REPLAY_SOURCE_WH = (224, 224)
REPLAY_GLOBAL_VIEW = (224, 224)
REPLAY_TEACHER_VIEW = (224, 224)
#: Geometry supplies the eligible support; controlled subsets of its 14x14 grid
#: make exact boundary counts without changing the production sampling policy.
REPLAY_TEACHER_CROPS = (
    (0, 0, 224, 224),
    (0, 0, 224, 224),
)
#: Candidate count for every batch row, one value per ladder step (fixed cap 49).
REPLAY_CANDIDATE_COUNTS = (48, 49, 50, 48)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_dense_ranking",
        description=(
            "Qualification runner for the optional dense sorter. Prints bounded "
            "streaming summaries and writes JSON evidence; it never installs "
            "packages or trains a model."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("numerical", "coordinate", "replay", "memory"),
        required=True,
        help="Qualification mode to run.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for the probe.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the synthetic stream.")
    parser.add_argument(
        "--steps", type=int, default=10, help="Minibatches to stream per reference size."
    )
    parser.add_argument(
        "--query-rows", type=int, default=4, help="Synthetic query rows per minibatch."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON evidence path; omit to print the summary to stdout.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Memory mode: batch size."
    )
    parser.add_argument(
        "--global-size", type=int, default=224, help="Memory mode: global crop."
    )
    parser.add_argument(
        "--local-size", type=int, default=96, help="Memory mode: local crop."
    )
    parser.add_argument(
        "--local-crops", type=int, default=6, help="Memory mode: number of locals."
    )
    parser.add_argument(
        "--patch-size", type=int, default=16, help="Memory mode: patch size."
    )
    parser.add_argument(
        "--query-cap", type=int, default=49, help="Memory mode: query cap."
    )
    parser.add_argument(
        "--reference-cap", type=int, default=16, help="Memory mode: reference cap."
    )
    parser.add_argument(
        "--full-valid",
        action="store_true",
        help="Memory mode: also report the all-valid counting bound as an assumption.",
    )
    parser.add_argument(
        "--reduced-overlap",
        action="store_true",
        help="Memory mode: also report an assumed reduced-overlap scenario.",
    )
    parser.add_argument(
        "--invalid-case",
        action="store_true",
        help="Run the deliberately invalid fixture and expect a nonzero exit.",
    )
    return parser


def _geometry(*, source_wh, view_wh, ops):
    """Compose one realised trace from ordered operations."""
    from otuformer.vision.geometry import GeometryTrace, matrix_for

    matrix = np.eye(3)
    for op in ops:
        matrix = matrix_for(op) @ matrix
    return GeometryTrace(
        source_wh=tuple(int(value) for value in source_wh),
        view_wh=tuple(int(value) for value in view_wh),
        operations=tuple(ops),
        source_to_view=matrix,
        view_to_source=np.linalg.inv(matrix),
    )


def _resize(from_wh, to_wh):
    from otuformer.vision.geometry import GeometryOp

    return GeometryOp(
        "resize",
        {
            "from_wh": [int(from_wh[0]), int(from_wh[1])],
            "to_wh": [int(to_wh[0]), int(to_wh[1])],
            "interpolation": "bicubic",
            "antialias": True,
        },
    )


def _crop(from_wh, left, top, width, height):
    from otuformer.vision.geometry import GeometryOp

    return GeometryOp(
        "crop",
        {
            "from_wh": [int(from_wh[0]), int(from_wh[1])],
            "left": int(left),
            "top": int(top),
            "width": int(width),
            "height": int(height),
        },
    )


def _rotate(from_wh, canvas_wh, angle):
    from otuformer.vision.geometry import GeometryOp

    return GeometryOp(
        "rotate",
        {
            "from_wh": [int(from_wh[0]), int(from_wh[1])],
            "canvas_wh": [int(canvas_wh[0]), int(canvas_wh[1])],
            "angle": float(angle),
            "expand": False,
            "interpolation": "bicubic",
            "antialias": True,
            "center": [int(from_wh[0]) / 2.0, int(from_wh[1]) / 2.0],
        },
    )


def _constructed_cases(reference_count: int, device: torch.device) -> list[tuple[str, torch.Tensor]]:
    """Deliberately constructed order/tie/extreme fixtures for this R."""
    R = reference_count
    ascending = torch.linspace(-1.0, 1.0, R).unsqueeze(0)
    ties = torch.zeros(2, R)
    extreme = torch.stack([torch.full((R,), -1.0), torch.full((R,), 1.0)])
    extreme[0, -1] = 1.0
    extreme[1, 0] = -1.0
    near_tie = ascending.clone()
    if R > 1:
        near_tie[0, 0] = near_tie[0, 1] - 1e-6
    return [
        (name, case.to(device))
        for name, case in (
            ("ascending", ascending),
            ("ties", ties),
            ("extreme", extreme),
            ("near_tie", near_tie),
        )
    ]


def _coordinate_fixtures(device: torch.device, seed: int):
    """Real norm/rho fixtures: exact centres, interpolation and low-rho mixtures.

    Every case is a *valid* input under the frozen guards; the deliberately
    invalid cancellation/low-norm fixtures live behind ``--invalid-case`` so a
    real failure is never mixed into the expected-valid population.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    patch_hw = (16, 16)
    grid_hw = (3, 3)

    # 1. Exact patch centres: rho is 1 by construction (the tail the design
    #    says min_rho=1 alone does not exercise).
    grid = torch.rand(1, grid_hw[0], grid_hw[1], 8, generator=generator) + 0.5
    centres = torch.tensor(
        [[8.0, 8.0], [24.0, 8.0], [40.0, 24.0], [24.0, 40.0]], dtype=torch.float64
    )
    yield "centres", grid, centres

    # 2. Interpolated positions (patch midpoints and quarter points), where the
    #    bilinear mixture is a genuine combination and rho < 1 is expected.
    between = torch.tensor(
        [[16.0, 8.0], [24.0, 16.0], [32.0, 24.0], [8.0, 16.0]], dtype=torch.float64
    )
    yield "interpolated", grid, between

    # 3. Unequal-magnitude, nearly opposed neighbours: the midpoint mixture is
    #    small but finite, which is exactly the low-rho tail an exact-centre
    #    stream never reaches (the exact-opposition case is a norm failure and
    #    stays behind --invalid-case).
    unequal = torch.zeros(1, 2, 2, 8)
    jitter = torch.rand(1, 2, 2, 8, generator=generator) * 0.05
    unequal[0, 0, 0, 0] = 4.0
    unequal[0, 0, 1, 0] = -3.8
    unequal = unequal + jitter
    yield (
        "unequal_magnitude",
        unequal.to(device),
        torch.tensor([[16.0, 8.0]], dtype=torch.float64),
    )

    # 4. Small-magnitude features that still clear the absolute norm floor: the
    #    guard's lower bound is exercised without tripping it. The deliberately
    #    below-floor case stays behind --invalid-case.
    small = (torch.rand(1, 2, 2, 8, generator=generator) + 0.5) * 1e-3
    yield (
        "low_magnitude",
        small.to(device),
        torch.tensor([[8.0, 8.0]], dtype=torch.float64),
    )


def _coordinate_invalid_fixture(device: torch.device):
    """Deliberately cancelling contributors; must raise, not silently pass."""
    grid = torch.zeros(1, 1, 2, 8)
    grid[0, 0, 0] = 1000.0
    grid[0, 0, 1, :] = -1000.0
    grid[0, 0, 1, 0] = -999.999
    return grid.to(device), torch.tensor([[16.0, 8.0]], dtype=torch.float64)


def _run_coordinate(args: argparse.Namespace) -> dict[str, object]:
    """Stream the Coordinate numerical contract without the optional sorter."""
    from otuformer.training import dense_objective as objective

    device = torch.device(args.device)
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")

    patch_hw = (16, 16)
    started = time.perf_counter()
    calls = 0
    query_instances = 0
    per_shape: dict[str, int] = {}
    min_norm = float("inf")
    min_rho = 1.0
    failures: list[str] = []
    low_rho = 0
    low_rho_by_shape: dict[str, int] = {}
    min_rho_by_shape: dict[str, float] = {}

    if args.invalid_case:
        grid, xy = _coordinate_invalid_fixture(device)
        try:
            objective.sample_dense_features(
                grid,
                xy.unsqueeze(0).to(device),
                patch_hw=patch_hw,
                valid=torch.ones(1, 1, 2, dtype=torch.bool),
            )
        except objective.DenseNumericalError as exc:
            failures.append(str(exc))
        else:
            failures.append("invalid fixture did not fail")

    for step in range(int(args.steps)):
        for name, grid, xy in _coordinate_fixtures(device, int(args.seed) + step):
            rows = int(xy.shape[0])
            valid = torch.ones(1, *grid.shape[1:3], dtype=torch.bool)
            try:
                _, stats = objective.sample_dense_features(
                    grid, xy.unsqueeze(0).to(device), patch_hw=patch_hw, valid=valid
                )
            except objective.DenseNumericalError as exc:
                failures.append(f"{name}: {exc}")
                continue
            calls += 1
            # Count the rows this call actually sampled, not a configured guess.
            query_instances += rows
            per_shape[name] = per_shape.get(name, 0) + rows
            min_norm = min(min_norm, stats["min_norm"])
            min_rho = min(min_rho, stats["min_rho"])
            min_rho_by_shape[name] = min(
                min_rho_by_shape.get(name, 1.0), float(stats["min_rho"])
            )
            if stats["min_rho"] < objective.LOW_RHO_REVIEW_THRESHOLD:
                low_rho += 1
                low_rho_by_shape[name] = low_rho_by_shape.get(name, 0) + 1

    elapsed = time.perf_counter() - started
    return {
        "mode": "coordinate",
        "device": str(device),
        "torch": torch.__version__,
        "config": {
            "seed": int(args.seed),
            "steps": int(args.steps),
            "query_rows": int(args.query_rows),
            "patch_hw": list(patch_hw),
            "feature_norm_floor": objective.FEATURE_NORM_FLOOR,
            "cancellation_floor": objective.CANCELLATION_FLOOR,
            "low_rho_threshold": objective.LOW_RHO_REVIEW_THRESHOLD,
        },
        "counts": {
            "calls": calls,
            "query_instances": query_instances,
            "query_instances_by_shape": dict(sorted(per_shape.items())),
            "min_norm": min_norm if calls else None,
            "min_rho": min_rho if calls else None,
            "low_rho_population": low_rho,
            # Constructed fixtures are named, so a deliberate low-rho case is not
            # mistaken for a measured random-population tail.
            "low_rho_population_by_shape": dict(sorted(low_rho_by_shape.items())),
            "min_rho_by_shape": {
                key: round(value, 6) for key, value in sorted(min_rho_by_shape.items())
            },
        },
        "coverage": {
            "exact_centres": "query_instances_by_shape.centres",
            "interpolation": "query_instances_by_shape.interpolated",
            "unequal_magnitude": "query_instances_by_shape.unequal_magnitude",
            "note": (
                "min_rho below 1 is the interpolation tail; an exact-centre-only "
                "stream cannot qualify it."
            ),
        },
        "elapsed_seconds": round(elapsed, 6),
        "failures": failures,
    }


def _controlled_pairs(geometry, grid_hws, patch_hw, *, counts, query_cap, generator):
    """Build production pair records with exactly ``counts[row]`` candidates.

    Real geometry decides *which* student centres are eligible; the count is then
    trimmed to the plan's boundary values, because a crop quantises to whole patch
    columns. The sampling itself is the production ``_sample_indices``, so the
    no-draw / at-cap / above-cap branches are the ones training would take.
    """
    from otuformer.training import dense_geometry as _geo

    _geo._validate_layout(geometry, grid_hws, local_crops=0)
    pairs = []
    for teacher_view in (0,):
        for student_view in (1,):
            student_xy = _geo._grid_centers(grid_hws[student_view], patch_hw)
            samples = []
            teacher_masks = []
            for row, wanted in enumerate(counts):
                student_ok = _geo._patch_valid(
                    geometry[student_view][row], grid_hws[student_view], patch_hw
                ).reshape(-1)
                mapped = _geo._map_to_view(
                    geometry[student_view][row],
                    geometry[teacher_view][row],
                    student_xy,
                )
                teacher_mask = _geo._patch_valid(
                    geometry[teacher_view][row], grid_hws[teacher_view], patch_hw
                )
                teacher_ok = _geo._support_mask(
                    mapped, teacher_mask, grid_hws[teacher_view], patch_hw
                )
                teacher_masks.append(teacher_mask)
                eligible = np.flatnonzero(student_ok & teacher_ok)[: int(wanted)]
                chosen = _geo._sample_indices(eligible, query_cap, generator)
                samples.append(
                    _geo.DenseSampleQueries(
                        student_indices=torch.from_numpy(
                            eligible[chosen].astype(np.int64)
                        ),
                        teacher_xy=torch.from_numpy(
                            np.ascontiguousarray(mapped[eligible[chosen]])
                        ),
                        eligible_count=int(eligible.shape[0]),
                    )
                )
            pairs.append(
                _geo.DensePairQueries(
                    teacher_view=teacher_view,
                    student_view=student_view,
                    samples=tuple(samples),
                    student_valid=torch.from_numpy(
                        np.stack(
                            [
                                _geo._patch_valid(
                                    geometry[student_view][row],
                                    grid_hws[student_view],
                                    patch_hw,
                                )
                                for row in range(len(counts))
                            ]
                        )
                    ),
                    teacher_valid=torch.from_numpy(np.stack(teacher_masks)),
                )
            )
    return tuple(pairs)


def _replay_inputs(seed: int):
    """Real geometry plus real feature grids for the replay lifecycle."""
    patch_size = 16
    source = REPLAY_SOURCE_WH
    global_view = REPLAY_GLOBAL_VIEW
    teacher_view = REPLAY_TEACHER_VIEW
    # View 0 is the teacher (cropped per row); view 1 is the student global.
    teacher_slots = tuple(
        _geometry(
            source_wh=source,
            view_wh=teacher_view,
            ops=(
                _crop(source, left, top, width, height),
                _resize((width, height), teacher_view),
            ),
        )
        for left, top, width, height in REPLAY_TEACHER_CROPS
    )
    student_slots = tuple(
        _geometry(
            source_wh=source,
            view_wh=global_view,
            ops=(_resize(source, global_view),),
        )
        for _ in REPLAY_TEACHER_CROPS
    )
    geometry = (teacher_slots, student_slots)
    grid_hws = (
        (teacher_view[1] // patch_size, teacher_view[0] // patch_size),
        (global_view[1] // patch_size, global_view[0] // patch_size),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    hidden = 16
    teacher_grids = {
        view: torch.rand(
            len(REPLAY_TEACHER_CROPS), *grid_hws[view], hidden, generator=generator
        )
        + 0.5
        for view in (0, 1)
    }
    student_grids = {
        view: (
            torch.rand(
                len(REPLAY_TEACHER_CROPS), *grid_hws[view], hidden, generator=generator
            )
            + 0.5
        ).requires_grad_(True)
        for view in (0, 1)
    }
    return geometry, grid_hws, (patch_size, patch_size), student_grids, teacher_grids


def _snapshot_optimizer(optimizer) -> dict:
    """A comparable, detached deep copy of the whole AdamW state.

    Keyed by *position* in the parameter list, not by ``Parameter`` identity:
    a cold restore builds fresh parameters, so object keys would never compare
    equal even when the restored state is byte-identical.
    """
    ordered = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    snapshot: dict = {}
    for index, parameter in enumerate(ordered):
        entry = optimizer.state.get(parameter)
        if not entry:
            continue
        snapshot[index] = {
            name: (
                value.detach().to("cpu").clone()
                if torch.is_tensor(value)
                else value
            )
            for name, value in entry.items()
        }
    return snapshot


def _tensor_states_equal(left: dict, right: dict) -> bool:
    """Compare every persisted head tensor, including biases and the last layer."""
    return set(left) == set(right) and all(
        torch.equal(value, right[key]) for key, value in left.items()
    )


def _optimizer_states_equal(left: dict, right: dict) -> bool:
    """Every parameter's every entry must match, not just the step counter."""
    if set(left) != set(right):
        return False
    for index in left:
        if set(left[index]) != set(right[index]):
            return False
        for name, value in left[index].items():
            other = right[index][name]
            if torch.is_tensor(value):
                if not torch.is_tensor(other) or not torch.equal(value, other):
                    return False
            elif value != other:
                return False
    return True


def _replay_step(
    *,
    step: int,
    query_cap: int,
    reference_cap: int,
    geometry,
    grid_hws,
    patch_hw,
    student_grids,
    teacher_grids,
    runtime,
    ranking,
    optimizer,
    pairs=None,
):
    """One real dense step: pairs, both objectives, one backward, one update."""
    from otuformer.training import dense_geometry as _geometry_module
    from otuformer.training import dense_objective as _objective

    if pairs is None:
        pairs = _geometry_module.build_dense_pairs(
            geometry,
            grid_hws=grid_hws,
            patch_hw=patch_hw,
            local_crops=0,
            query_cap=int(query_cap),
            generator=runtime.query_generator,
        )
    coordinate = _objective.coordinate_loss(
        student_grids, teacher_grids, pairs, patch_hw=patch_hw
    )
    ranked = ranking.compute(
        student_grids,
        teacher_grids,
        pairs,
        patch_hw=patch_hw,
        reference_cap=int(reference_cap),
        generator=runtime.reference_generator,
    )
    terms = [term for term in (coordinate.raw, ranked.raw) if term is not None]
    loss = torch.stack(terms).sum()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if ranked.raw is not None:
        ranking.update_teacher(0.9)
        runtime.ranking_contributed_steps += 1
    record = {
        "step": int(step),
        "query_cap": int(query_cap),
        "reference_cap": int(reference_cap),
        "eligible": [int(sample.eligible_count) for pair in pairs for sample in pair.samples],
        "selected": [
            int(sample.student_indices.numel()) for pair in pairs for sample in pair.samples
        ],
        "coordinate_raw": None if coordinate.raw is None else float(coordinate.raw.detach()),
        "ranking_raw": None if ranked.raw is None else float(ranked.raw.detach()),
        "reference_population": ranked.statistics.get("reference_population"),
        "reference_count": ranked.statistics.get("reference_count"),
        "pair_pools": [
            (sum(sample.student_indices.numel() for sample in pair.samples),
             sum(bool(sample.student_indices.numel()) for sample in pair.samples))
            for pair in pairs
        ],
        "selected_identity": [
            (int(pair.teacher_view), int(pair.student_view), sample.student_indices.tolist())
            for pair in pairs
            for sample in pair.samples
        ],
        "optimizer_state": _snapshot_optimizer(optimizer),
        "query_rng": runtime.query_generator.get_state().clone(),
        "reference_rng": (
            None
            if runtime.reference_generator is None
            else runtime.reference_generator.get_state().clone()
        ),
        "teacher_head": {
            key: value.detach().cpu().clone()
            for key, value in ranking.teacher_head.state_dict().items()
        },
        "student_head": {
            key: value.detach().cpu().clone()
            for key, value in ranking.student_head.state_dict().items()
        },
    }
    for view, grid in student_grids.items():
        grid.grad = None
    return record, pairs


def _run_replay_lifecycle(*, seed: int, ladder, checkpoint_path: Path | None):
    """Run the real dense lifecycle, optionally restarting cold from disk.

    With ``checkpoint_path`` set the live objects are *discarded* after every
    save: a fresh head, a fresh optimizer and a runtime restored from the closed
    bundle replace them. Reusing the in-memory objects would hide a broken
    restore path, because a no-op load would still reproduce every number.
    """
    from otuformer.training import dense_recipe
    from otuformer.training import neighbor_ranking as _ranking_module
    from otuformer.training.neighbor_ranking import NeighborRankingObjective

    # A fresh process starts with an empty sorter cache and a fresh global RNG;
    # both runs must start from that same state or the comparison is vacuous.
    _ranking_module._SORTER_CACHE.clear()
    torch.manual_seed(int(seed))
    settings = dense_recipe.DenseSettings(
        coordinate_weight=1.0, ranking_weight=1.0, query_cap=49, reference_cap=16
    )
    runtime = dense_recipe.new_dense_runtime(settings, seed=int(seed))
    geometry, grid_hws, patch_hw, student_grids, teacher_grids = _replay_inputs(
        int(seed)
    )
    ranking = NeighborRankingObjective(16, out_dim=8, hidden=16)
    optimizer = torch.optim.AdamW(
        ranking.trainable_parameters(), lr=1e-3, weight_decay=0.05
    )
    records: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    boundaries = len(ladder)
    # One step past the last save boundary: the restored objects must also drive
    # a real update, not only reproduce the boundary's own numbers.
    steps = list(ladder) + [REPLAY_TAIL_STEP]
    for step, (query_cap, reference_cap) in enumerate(steps):
        if step < boundaries:
            # Fixed cap, controlled candidate count: the ladders cross the real
            # caps rather than raising them past the production bound.
            per_row = [5, REPLAY_CANDIDATE_COUNTS[step]]
            pairs = _controlled_pairs(
                geometry,
                grid_hws,
                patch_hw,
                counts=per_row,
                query_cap=REPLAY_FIXED_QUERY_CAP,
                generator=runtime.query_generator,
            )
        else:
            pairs = None
        record, _pairs = _replay_step(
            step=step,
            query_cap=query_cap,
            reference_cap=reference_cap,
            geometry=geometry,
            grid_hws=grid_hws,
            patch_hw=patch_hw,
            student_grids=student_grids,
            teacher_grids=teacher_grids,
            runtime=runtime,
            ranking=ranking,
            optimizer=optimizer,
            pairs=pairs,
        )
        records.append(record)
        if checkpoint_path is None or step >= boundaries:
            # The tail step intentionally runs from whatever state is live; it is
            # compared like every other step, so a wrong restore shows up here.
            continue
        bundle = {
            "dense_training": dense_recipe.dense_state_bundle(
                runtime, completed_steps=step + 1, ranking=ranking
            ),
            "student_head": ranking.student_head.state_dict(),
            "teacher_head": ranking.teacher_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step + 1,
        }
        torch.save(bundle, checkpoint_path)

        # ---- cold restart: nothing survives except the file on disk ----
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        del ranking, optimizer, runtime
        _ranking_module._SORTER_CACHE.clear()
        torch.manual_seed(int(seed) + 1000)
        runtime = dense_recipe.restore_dense_runtime(
            settings, loaded["dense_training"]
        )
        # A deliberately fresh head: if the restore were a no-op, the saved and
        # replayed numbers below could not match.
        ranking = NeighborRankingObjective(16, out_dim=8, hidden=16)
        fresh_before_load = 1.0 - torch.equal(
            ranking.student_head.state_dict()["0.weight"],
            loaded["student_head"]["0.weight"],
        )
        ranking.student_head.load_state_dict(loaded["student_head"], strict=True)
        ranking.teacher_head.load_state_dict(loaded["teacher_head"], strict=True)
        optimizer = torch.optim.AdamW(
            ranking.trainable_parameters(), lr=1e-3, weight_decay=0.05
        )
        optimizer.load_state_dict(loaded["optimizer"])
        # Rebuild the fixtures from the same declared seed: identical values prove
        # the *state* (not a surviving tensor) drives the continuation.
        geometry, grid_hws, patch_hw, student_grids, teacher_grids = _replay_inputs(
            int(seed)
        )
        head_keys = list(loaded["student_head"])
        checkpoints.append(
            {
                "step": step + 1,
                "contributed_steps": int(runtime.ranking_contributed_steps),
                "query_rng_identical": torch.equal(
                    runtime.query_generator.get_state(), record["query_rng"]
                ),
                "reference_rng_identical": torch.equal(
                    runtime.reference_generator.get_state(), record["reference_rng"]
                ),
                "head_identical": all(
                    torch.equal(
                        ranking.student_head.state_dict()[key],
                        loaded["student_head"][key],
                    )
                    for key in head_keys
                ),
                "teacher_head_identical": all(
                    torch.equal(
                        ranking.teacher_head.state_dict()[key],
                        loaded["teacher_head"][key],
                    )
                    for key in head_keys
                ),
                "restore_was_not_a_noop": bool(fresh_before_load),
                "moment_steps": sorted(
                    {
                        int(entry["step"])
                        for entry in optimizer.state.values()
                        if isinstance(entry, dict) and "step" in entry
                    }
                ),
                "optimizer_state_entries": len(optimizer.state),
            }
        )
    return records, checkpoints, ranking, runtime


def _run_reference_ladder(*, seed: int, checkpoint_path: Path | None):
    """Replay the reference-pool boundary: pool 15 -> 16 -> 17 -> 15, cap 16."""
    from otuformer.training import dense_recipe
    from otuformer.training import neighbor_ranking as _ranking_module
    from otuformer.training.neighbor_ranking import NeighborRankingObjective

    _ranking_module._SORTER_CACHE.clear()
    torch.manual_seed(int(seed))
    settings = dense_recipe.DenseSettings(
        coordinate_weight=0.0, ranking_weight=1.0, query_cap=49, reference_cap=16
    )
    runtime = dense_recipe.new_dense_runtime(settings, seed=int(seed))
    geometry, grid_hws, patch_hw, student_grids, teacher_grids = _replay_inputs(int(seed))
    ranking = NeighborRankingObjective(16, out_dim=8, hidden=16)
    optimizer = torch.optim.AdamW(
        ranking.trainable_parameters(), lr=1e-3, weight_decay=0.05
    )
    records: list[dict[str, object]] = []
    # Keep row A at 5 while row B crosses 10/11/12/10; both share one pool.
    boundaries = len(REPLAY_REFERENCE_LADDER)
    # One step past the last save boundary, so the restored reference RNG and
    # optimizer must actually drive an update.
    steps = list(REPLAY_REFERENCE_LADDER) + [REPLAY_REFERENCE_TAIL_STEP]
    for step, (query_cap, reference_cap) in enumerate(steps):
        population = REPLAY_REFERENCE_POOL[step] if step < boundaries else 16
        pairs = _controlled_pairs(
            geometry, grid_hws, patch_hw, counts=(5, population - 5),
            query_cap=49, generator=runtime.query_generator,
        )
        record, _pairs = _replay_step(
            step=step,
            query_cap=query_cap,
            reference_cap=reference_cap,
            geometry=geometry,
            grid_hws=grid_hws,
            patch_hw=patch_hw,
            student_grids=student_grids,
            teacher_grids=teacher_grids,
            runtime=runtime,
            ranking=ranking,
            optimizer=optimizer,
            pairs=pairs,
        )
        records.append(record)
        if checkpoint_path is not None and step < boundaries:
            bundle = {
                "dense_training": dense_recipe.dense_state_bundle(
                    runtime, completed_steps=step + 1, ranking=ranking
                ),
                "student_head": ranking.student_head.state_dict(),
                "teacher_head": ranking.teacher_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step + 1,
            }
            torch.save(bundle, checkpoint_path)
            loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            del ranking, optimizer, runtime
            _ranking_module._SORTER_CACHE.clear()
            runtime = dense_recipe.restore_dense_runtime(
                settings, loaded["dense_training"]
            )
            ranking = NeighborRankingObjective(16, out_dim=8, hidden=16)
            ranking.student_head.load_state_dict(loaded["student_head"], strict=True)
            ranking.teacher_head.load_state_dict(loaded["teacher_head"], strict=True)
            optimizer = torch.optim.AdamW(
                ranking.trainable_parameters(), lr=1e-3, weight_decay=0.05
            )
            optimizer.load_state_dict(loaded["optimizer"])
            geometry, grid_hws, patch_hw, student_grids, teacher_grids = _replay_inputs(
                int(seed)
            )
    return records


def _run_replay(args: argparse.Namespace) -> dict[str, object]:
    """Replay cap-crossing selections through the real dense lifecycle.

    Two runs are compared: the second round-trips a real checkpoint (closed
    dense bundle + head + AdamW state) after every step, so restored generator
    state, references, losses and heads must reproduce the continuous run.
    """
    import tempfile

    from otuformer.training import dense_recipe

    failures: list[str] = []
    started = time.perf_counter()
    continuous, _no_ckpt, _ranking, _runtime = _run_replay_lifecycle(
        seed=int(args.seed), ladder=REPLAY_LADDER, checkpoint_path=None
    )
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "dense-step.ckpt"
        with_checkpoints, checkpoints, ranking, runtime = _run_replay_lifecycle(
            seed=int(args.seed), ladder=REPLAY_LADDER, checkpoint_path=checkpoint_path
        )

    comparisons = 0
    replayed = 0
    for saved, fresh in zip(continuous, with_checkpoints):
        for key in (
            "eligible",
            "selected",
            "coordinate_raw",
            "ranking_raw",
            "reference_population",
            "reference_count",
        ):
            comparisons += 1
            if saved[key] == fresh[key]:
                replayed += 1
            else:
                failures.append(
                    f"step {saved['step']} {key}: {saved[key]!r} != {fresh[key]!r}"
                )
        for key, label in (
            ("query_rng", "query generator state"),
            ("reference_rng", "reference generator state"),
            ("selected_identity", "selected query indices"),
        ):
            comparisons += 1
            equal = (
                torch.equal(saved[key], fresh[key])
                if torch.is_tensor(saved[key])
                else saved[key] == fresh[key]
            )
            if equal:
                replayed += 1
            else:
                failures.append(f"step {saved['step']}: {label} diverged")
        for key in ("student_head", "teacher_head"):
            comparisons += 1
            if _tensor_states_equal(saved[key], fresh[key]):
                replayed += 1
            else:
                failures.append(f"step {saved['step']}: {key.replace('_', ' ')} diverged")
        # The whole AdamW state, not only the step counter: a zeroed exp_avg or a
        # corrupted exp_avg_sq must fail this comparison.
        comparisons += 1
        if _optimizer_states_equal(
            saved["optimizer_state"], fresh["optimizer_state"]
        ):
            replayed += 1
        else:
            failures.append(
                f"step {saved['step']}: restored optimizer moments differ"
            )

    for checkpoint in checkpoints:
        if not checkpoint["query_rng_identical"]:
            failures.append(f"checkpoint {checkpoint['step']}: query RNG not restored")
        if not checkpoint["reference_rng_identical"]:
            failures.append(
                f"checkpoint {checkpoint['step']}: reference RNG not restored"
            )
        if not checkpoint["head_identical"]:
            failures.append(
                f"checkpoint {checkpoint['step']}: head weights not restored"
            )
        if not checkpoint["teacher_head_identical"]:
            failures.append(
                f"checkpoint {checkpoint['step']}: teacher head not restored"
            )
        if not checkpoint["restore_was_not_a_noop"]:
            failures.append(
                f"checkpoint {checkpoint['step']}: the fresh head already equalled "
                "the saved head, so this round-trip proves nothing"
            )
        if not checkpoint["optimizer_state_entries"]:
            failures.append(
                f"checkpoint {checkpoint['step']}: the AdamW step created no state"
            )
        if checkpoint["moment_steps"] != [checkpoint["contributed_steps"]]:
            failures.append(
                f"checkpoint {checkpoint['step']}: restored AdamW steps "
                f"{checkpoint['moment_steps']} do not match the contribution count "
                f"{checkpoint['contributed_steps']}"
            )

    # A changed eligibility set must genuinely select differently: this is
    # measured, never asserted by construction.
    with tempfile.TemporaryDirectory() as directory:
        ladder_ckpt = Path(directory) / "reference-ladder.ckpt"
        reference_continuous = _run_reference_ladder(
            seed=int(args.seed), checkpoint_path=None
        )
        reference_restored = _run_reference_ladder(
            seed=int(args.seed), checkpoint_path=ladder_ckpt
        )
    # Only the ladder steps define the boundary sequence; the tail step is a
    # separate post-restore check.
    reference_pools = [
        int(record["reference_population"])
        for record in reference_continuous[: len(REPLAY_REFERENCE_LADDER)]
        if record["reference_population"] is not None
    ]
    if reference_pools != list(REPLAY_REFERENCE_POOL):
        failures.append(
            "reference-pool ladder did not reproduce "
            f"{list(REPLAY_REFERENCE_POOL)}: {reference_pools}"
        )
    # The ladder must also survive a cold restore with the SAME completeness as
    # the main replay: pools, losses, both generators, head and full optimizer
    # state, plus the tail step after the last boundary.
    for saved, fresh in zip(reference_continuous, reference_restored):
        for key in (
            "reference_population",
            "reference_count",
            "selected_identity",
            "ranking_raw",
            "coordinate_raw",
        ):
            comparisons += 1
            equal = (
                torch.equal(saved[key], fresh[key])
                if torch.is_tensor(saved[key])
                else saved[key] == fresh[key]
            )
            if equal:
                replayed += 1
            else:
                failures.append(
                    f"reference ladder step {saved['step']} {key} diverged"
                )
        for key, label in (
            ("query_rng", "query generator state"),
            ("reference_rng", "reference generator state"),
        ):
            comparisons += 1
            if torch.equal(saved[key], fresh[key]):
                replayed += 1
            else:
                failures.append(
                    f"reference ladder step {saved['step']}: {label} diverged"
                )
        for key in ("student_head", "teacher_head"):
            comparisons += 1
            if _tensor_states_equal(saved[key], fresh[key]):
                replayed += 1
            else:
                failures.append(
                    f"reference ladder step {saved['step']}: {key.replace('_', ' ')} diverged"
                )
        comparisons += 1
        if _optimizer_states_equal(
            saved["optimizer_state"], fresh["optimizer_state"]
        ):
            replayed += 1
        else:
            failures.append(
                f"reference ladder step {saved['step']}: optimizer moments diverged"
            )

    # Verify the exact no-draw/draw branches as well as restored trajectories.
    expected_runtime = dense_recipe.new_dense_runtime(
        dense_recipe.DenseSettings(1.0, 1.0, 49, 16), seed=int(args.seed)
    )
    query_draws_match = True
    for record in continuous[:len(REPLAY_LADDER)]:
        for count in record["eligible"]:
            if count > 49:
                torch.multinomial(
                    torch.ones(count, dtype=torch.float64), 49, replacement=False,
                    generator=expected_runtime.query_generator,
                )
        query_draws_match &= torch.equal(
            record["query_rng"], expected_runtime.query_generator.get_state()
        )
    reference_draws_match = True
    for records in (continuous, reference_continuous):
        expected = dense_recipe.new_dense_runtime(
            dense_recipe.DenseSettings(1.0, 1.0, 49, 16), seed=int(args.seed)
        ).reference_generator
        for record in records:
            # The main tail uses both opposite-global pairs in their native order.
            pair_pools = record["pair_pools"]
            for population, samples in pair_pools:
                if population > 16:
                    for _ in range(samples):
                        torch.multinomial(
                            torch.ones(population, dtype=torch.float64), 16,
                            replacement=False, generator=expected,
                        )
            reference_draws_match &= torch.equal(
                record["reference_rng"], expected.get_state()
            )
    if not query_draws_match or not reference_draws_match:
        failures.append("query/reference draw consumption differs from the fixed-cap policy")

    changed_selection = _replay_changed_eligibility(int(args.seed))
    if not changed_selection:
        failures.append(
            "a changed eligibility set selected identically; the replay claim "
            "would be vacuous"
        )

    reference_populations = sorted(
        {
            int(record["reference_population"])
            for record in continuous
            if record["reference_population"] is not None
        }
    )
    query_cap_crossed = any(
        any(eligible > record["query_cap"] for eligible in record["eligible"])
        for record in continuous
    )
    reference_cap_crossed = any(
        record["reference_population"] is not None
        and record["reference_population"] > record["reference_cap"]
        for record in continuous
    )
    elapsed = time.perf_counter() - started
    return {
        "mode": "replay",
        "device": "cpu",
        "torch": torch.__version__,
        "config": {
            "seed": int(args.seed),
            "steps": len(REPLAY_LADDER),
            "ladder": [list(item) for item in REPLAY_LADDER],
        },
        "lifecycle": {
            "exercised": [
                "build_dense_pairs on realised geometry",
                "coordinate_loss",
                "NeighborRankingObjective.compute (real sorter)",
                "one backward and one AdamW step",
                "ranking teacher-head EMA",
                "dense_state_bundle / torch.save / restore_dense_runtime",
                "independent query and reference CPU generator streams",
            ],
            "not_exercised": [
                "backbone/SSL forward and the surrounding pretrain loop",
                "multi-worker data loading",
                "accelerator devices",
            ],
        },
        "counts": {
            "transitions": comparisons,
            "replayed": replayed,
            "generator_checkpoints": len(checkpoints),
            "cold_restore_checks": checkpoints,
            "query_cap_crossed": bool(query_cap_crossed),
            "reference_cap_crossed": bool(reference_cap_crossed),
            "reference_populations": reference_populations,
            "reference_ladder_pools": reference_pools,
            "reference_ladder_selected": [
                record["selected"] for record in reference_continuous[:len(REPLAY_REFERENCE_LADDER)]
            ],
            "query_draws_match_expected": bool(query_draws_match),
            "reference_draws_match_expected": bool(reference_draws_match),
            "query_ladder_selected": [
                record["selected"] for record in continuous[:4]
            ],
            "query_ladder_eligible": [
                record["eligible"] for record in continuous[:4]
            ],
            "eligible_counts": [
                record["eligible"] for record in continuous
            ],
            "selected_counts": [record["selected"] for record in continuous],
            "changed_eligibility_selects_differently": bool(changed_selection),
            "ranking_contributed_steps": int(runtime.ranking_contributed_steps),
            "teacher_head_moves": bool(
                not torch.equal(
                    ranking.teacher_head.state_dict()["0.weight"],
                    ranking.student_head.state_dict()["0.weight"],
                )
            ),
        },
        "elapsed_seconds": round(elapsed, 6),
        "failures": failures,
    }


def _replay_changed_eligibility(seed: int) -> bool:
    """Return whether a different eligibility set really changes the selection."""
    from otuformer.training import dense_geometry as _geometry_module

    geometry, grid_hws, patch_hw, _student, _teacher = _replay_inputs(int(seed))
    generator_a = torch.Generator(device="cpu").manual_seed(7)
    generator_b = torch.Generator(device="cpu").manual_seed(7)
    pairs_a = _geometry_module.build_dense_pairs(
        geometry, grid_hws=grid_hws, patch_hw=patch_hw, local_crops=0,
        query_cap=49, generator=generator_a,
    )
    # Same seed, different realised eligibility changes the candidate set.
    widened = tuple(
        _geometry(
            source_wh=REPLAY_SOURCE_WH,
            view_wh=REPLAY_TEACHER_VIEW,
            ops=(
                _crop(REPLAY_SOURCE_WH, left, top, width, height),
                _resize((width, height), REPLAY_TEACHER_VIEW),
            ),
        )
        for left, top, width, height in ((0, 0, 96, 96), (0, 0, 112, 112))
    )
    geometry_b = (widened, geometry[1])
    pairs_b = _geometry_module.build_dense_pairs(
        geometry_b, grid_hws=grid_hws, patch_hw=patch_hw, local_crops=0,
        query_cap=49, generator=generator_b,
    )
    selected_a = [
        sample.student_indices.tolist() for pair in pairs_a for sample in pair.samples
    ]
    selected_b = [
        sample.student_indices.tolist() for pair in pairs_b for sample in pair.samples
    ]
    return selected_a != selected_b


def _memory_geometry(*, batch_size, global_size, local_size, local_crops, patch_size):
    """Real per-row geometry for the requested memory shape."""
    from otuformer.vision.geometry import GeometryTrace  # noqa: F401  (contract)

    source = (max(global_size, local_size) * 2, max(global_size, local_size) * 2)
    global_ops = (_resize(source, (global_size, global_size)),)
    rows = []
    for row in range(int(batch_size)):
        globals_row = tuple(
            _geometry(source_wh=source, view_wh=(global_size, global_size), ops=global_ops)
            for _ in range(2)
        )
        locals_row = []
        for index in range(int(local_crops)):
            left = (row * 3 + index * 7) % max(1, source[0] - local_size)
            top = (row * 5 + index * 11) % max(1, source[1] - local_size)
            if row % 4 == 3:
                # Every fourth image is rotated: its corner patches lose support,
                # which is the real mechanism that lowers eligible counts.
                locals_row.append(
                    _geometry(
                        source_wh=source,
                        view_wh=(local_size, local_size),
                        ops=(
                            _rotate(source, source, 35.0),
                            _resize(source, (local_size, local_size)),
                        ),
                    )
                )
            else:
                locals_row.append(
                    _geometry(
                        source_wh=source,
                        view_wh=(local_size, local_size),
                        ops=(
                            _crop(source, left, top, local_size, local_size),
                            _resize((local_size, local_size), (local_size, local_size)),
                        ),
                    )
                )
        rows.append(globals_row + tuple(locals_row))
    # Transpose row-major records into one slot per tensor view.
    views = len(rows[0])
    return tuple(tuple(row[index] for row in rows) for index in range(views))


def _run_memory(args: argparse.Namespace) -> dict[str, object]:
    """Report the counting bound beside measured geometry-derived populations.

    The sorter-graph numbers are source-snippet extrapolations from the design;
    they are never presented as GPU peaks, and the geometry-derived counts are
    labelled separately from the assumptions.
    """
    from otuformer.training import dense_geometry as _geometry_module

    patch = int(args.patch_size)
    global_grid = int(args.global_size) // patch
    local_grid = int(args.local_size) // patch
    local_crops = int(args.local_crops)
    query_cap = int(args.query_cap)
    per_pair_global = min(query_cap, global_grid * global_grid)
    per_pair_local = min(query_cap, local_grid * local_grid)
    bound = int(args.batch_size) * (
        2 * per_pair_global + 2 * local_crops * per_pair_local
    )

    started = time.perf_counter()
    failures: list[str] = []
    geometry = _memory_geometry(
        batch_size=int(args.batch_size),
        global_size=int(args.global_size),
        local_size=int(args.local_size),
        local_crops=local_crops,
        patch_size=patch,
    )
    grid_hws = (
        (global_grid, global_grid),
        (global_grid, global_grid),
    ) + tuple((local_grid, local_grid) for _ in range(local_crops))
    pairs = _geometry_module.build_dense_pairs(
        geometry,
        grid_hws=grid_hws,
        patch_hw=(patch, patch),
        local_crops=local_crops,
        query_cap=query_cap,
        generator=torch.Generator(device="cpu").manual_seed(int(args.seed)),
    )
    actual_queries = sum(
        int(sample.student_indices.numel()) for pair in pairs for sample in pair.samples
    )
    eligible_total = sum(
        int(sample.eligible_count) for pair in pairs for sample in pair.samples
    )
    per_pair = {}
    for pair in pairs:
        key = "global" if pair.student_view < 2 and pair.teacher_view < 2 else "local"
        per_pair[key] = per_pair.get(key, 0) + sum(
            int(sample.student_indices.numel()) for sample in pair.samples
        )
    if not 0 <= actual_queries <= bound:
        failures.append(
            f"measured queries {actual_queries} are outside the counting bound {bound}"
        )

    assumptions: dict[str, object] = {}
    if args.full_valid:
        assumptions["all_valid_queries"] = bound
        assumptions["all_valid_note"] = (
            "upper counting bound, not an observed population; real rotation and "
            "cropping remove eligible points"
        )
    if args.reduced_overlap:
        assumptions["reduced_overlap_queries"] = int(bound * 0.15)
        assumptions["reduced_overlap_note"] = (
            "plain assumption used to illustrate a sparse-overlap scenario, not a "
            "measurement"
        )

    bytes_per_query = {49: 240908, 16: 13760}
    gib = {
        f"reference_{ref}": round(actual_queries * rate / 2**30, 4)
        for ref, rate in bytes_per_query.items()
    }
    elapsed = time.perf_counter() - started
    return {
        "mode": "memory",
        "configured_bound": bound,
        "actual_queries": actual_queries,
        "eligible_queries": eligible_total,
        "queries_by_pair_kind": per_pair,
        "measured_peak_device": "cpu-extrapolation",
        "measured_peak_bytes": None,
        "sorter_graph_gib": gib,
        "population_sources": {
            "configured_bound": "computed",
            "actual_queries": "measured (real geometry fixture, no model run)",
            "eligible_queries": "measured (real geometry fixture)",
            "sorter_graph_gib": "computed (design source-snippet rate extrapolation)",
            "assumptions": "assumed",
        },
        "config": {
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "global_size": int(args.global_size),
            "local_size": int(args.local_size),
            "local_crops": local_crops,
            "patch_size": patch,
            "query_cap": query_cap,
            "reference_cap": int(args.reference_cap),
            "full_valid": bool(args.full_valid),
            "reduced_overlap": bool(args.reduced_overlap),
        },
        "assumptions": assumptions,
        "notes": (
            "Extrapolation only: excludes backbone/head/reference activations, "
            "backward workspace and existing objectives. Actual queries come from "
            "real geometry for this configuration, not from the bound."
        ),
        "elapsed_seconds": round(elapsed, 6),
        "failures": failures,
    }


def _run_numerical(args: argparse.Namespace) -> dict[str, object]:
    """Stream the frozen sorter recipe and summarise bounded statistics."""
    from otuformer.training import neighbor_ranking as ranking

    device = torch.device(args.device)
    if args.steps <= 0 or args.query_rows <= 0:
        raise SystemExit("--steps and --query-rows must be positive")

    dependency = ranking.probe_ranking_dependency(device)
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    started = time.perf_counter()
    calls = 0
    query_instances = 0
    min_probability = 1.0
    max_residual = 0.0
    failures: list[str] = []
    cache: dict = {}
    if args.invalid_case:
        cosines = torch.full((1, 4), float("nan"))
        try:
            ranking.sort_values_and_permutation(cosines, cache=cache)
        except ranking.RankingDependencyError as exc:
            failures.append(str(exc))
        except Exception as exc:  # pragma: no cover - unexpected path
            failures.append(f"unexpected: {exc}")
        else:
            failures.append("invalid fixture did not fail")

    def _production_checks(permutation, context: str) -> None:
        """Apply the same guards production training applies.

        A qualification stream that passes while production would reject the
        same tensor is not evidence: the student permutation must be strictly
        positive (unclamped log) and its row-cross-entropy must be finite.
        """
        if bool((permutation.detach() <= 0).any()):
            shortest = float(permutation.detach().min())
            failures.append(
                f"{context}: student permutation has a non-positive entry "
                f"(min {shortest:.3e}); production row_cross_entropy rejects this"
            )
            return
        teacher_target = permutation.detach().transpose(-1, -2).clone()
        try:
            loss = ranking.row_cross_entropy(permutation, teacher_target, context=context)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            failures.append(f"{context}: row_cross_entropy failed: {exc}")
            return
        if not bool(torch.isfinite(loss).all()):
            failures.append(f"{context}: row_cross_entropy is non-finite")
            return

    def _backward_checks(reference_count: int, context: str) -> None:
        """A small production CE backward: the graph must reach the cosines.

        Forward-only evidence cannot show that the differentiable ordering is
        still connected; a detached sorter would pass every check above.
        """
        cosines = (
            torch.rand(2, reference_count, generator=generator) * 2.0 - 1.0
        ).to(device)
        cosines = cosines.clone().requires_grad_(True)
        _, permutation = ranking.sort_values_and_permutation(cosines, cache=cache)
        teacher = permutation.detach().transpose(-1, -2).clone()
        try:
            loss = ranking.row_cross_entropy(permutation, teacher, context=context)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            failures.append(f"{context}: backward fixture CE failed: {exc}")
            return
        try:
            loss.sum().backward()
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            failures.append(f"{context}: backward through the sorter failed: {exc}")
            return
        if cosines.grad is None:
            failures.append(
                f"{context}: the sorter is detached; no gradient reached the cosines"
            )
            return
        if not bool(torch.isfinite(cosines.grad).all()):
            failures.append(f"{context}: non-finite cosine gradient")
            return
        # R=2 can legitimately cancel to zero (a symmetric two-row target), so the
        # non-degeneracy expectation only applies from R=4 up.
        if reference_count >= 4 and not bool((cosines.grad != 0).any()):
            failures.append(f"{context}: cosine gradient is identically zero")

    for reference_count in REFERENCE_SWEEP:
        for case_name, case in _constructed_cases(reference_count, device):
            _, permutation = ranking.sort_values_and_permutation(case, cache=cache)
            calls += 1
            query_instances += int(case.shape[0])
            min_probability = min(min_probability, float(permutation.min()))
            residual = float((permutation.sum(-1) - 1).abs().max())
            max_residual = max(max_residual, residual)
            if residual > ranking.STOCHASTICITY_TOLERANCE:
                failures.append(
                    f"R={reference_count} {case_name} constructed residual "
                    f"{residual:.3e}"
                )
            _production_checks(permutation, f"R={reference_count} {case_name}")
        _backward_checks(reference_count, f"R={reference_count}")
        for _ in range(int(args.steps)):
            cosines = torch.rand(
                int(args.query_rows), reference_count, generator=generator
            ) * 2.0 - 1.0
            cosines = cosines.to(device)
            _, permutation = ranking.sort_values_and_permutation(cosines, cache=cache)
            calls += 1
            query_instances += int(cosines.shape[0])
            min_probability = min(min_probability, float(permutation.min()))
            residual = float((permutation.sum(-1) - 1).abs().max())
            max_residual = max(max_residual, residual)
            if residual > ranking.STOCHASTICITY_TOLERANCE:
                failures.append(f"R={reference_count} residual {residual:.3e}")
            _production_checks(permutation, f"R={reference_count} random")

    elapsed = time.perf_counter() - started
    return {
        "mode": "numerical",
        "dependency": dependency,
        "device": str(device),
        "torch": torch.__version__,
        "config": {
            "seed": int(args.seed),
            "steps": int(args.steps),
            "query_rows": int(args.query_rows),
            "reference_sweep": list(REFERENCE_SWEEP),
            "steepness": ranking.SORT_STEEPNESS,
            "distribution": ranking.SORT_DISTRIBUTION,
            "network": ranking.SORT_NETWORK,
        },
        "counts": {
            "calls": calls,
            "query_instances": query_instances,
            "min_probability": min_probability,
            "max_stochasticity_residual": max_residual,
        },
        "elapsed_seconds": round(elapsed, 6),
        "failures": failures,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run one qualification mode; return a process exit status."""
    args = _build_parser().parse_args(argv)
    handlers = {
        "numerical": _run_numerical,
        "coordinate": _run_coordinate,
        "replay": _run_replay,
        "memory": _run_memory,
    }
    payload = handlers[args.mode](args)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    else:
        print(text)
    return 1 if payload["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
