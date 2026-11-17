"""Optional differentiable sorting adapter for the NeCo-style ranking objective.

This module is the only place the optional ``diffsort`` package is imported, and
it imports it lazily: a Coordinate-only, all-off or schema-discovery code path
must never touch it. The frozen recipe is bitonic + Cauchy, steepness 100,
float32, ascending cosine, with ``P[j, k]`` meaning "input reference ``j`` was
assigned sorted rank ``k``" and ``sorted = c @ P``.

Numeric contract (design §8, plan "Closed checkpoint schema"): unclamped
``log(P)``. Student probabilities must be strictly positive; teacher targets may
contain zeros because they carry no log. Row/column sums are validated against
the provisional ``1e-5`` residual, and coefficients stay nonnegative. The
adapter never clamps, never falls back to another distribution and never widens
a guard to make a run pass.

Cache: sort-network constants are device/dtype/size specific and expensive to
build. At most four reference sizes are retained, and only plain tensors are
stored so no query, reference or autograd graph is pinned.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

import torch

SORT_NETWORK = "bitonic"
SORT_DISTRIBUTION = "cauchy"
SORT_STEEPNESS = 100.0
SORT_DIRECTION = "ascending-cosine"

MIN_REFERENCE_POINTS = 2
MAX_REFERENCE_POINTS = 49
STOCHASTICITY_TOLERANCE = 1e-5
PROBABILITY_FLOOR = 0.0
CACHE_SIZE = 4

# Qualified artifact: diffsort 0.2.0 (PyPI wheel and sdist digests recorded in
# the design). The loaded implementation must match these three source digests;
# a matching version string alone is not evidence of an equivalent recipe.
QUALIFIED_VERSION = "0.2.0"
QUALIFIED_WHEEL_SHA256 = (
    "95bd5541ade48919cd926366f83165e07645b6e62fe798343b7f0d0332b8a08b"
)
QUALIFIED_SOURCE_SHA256 = {
    "diffsort/diffsort.py": "7dd6d3a72e25f17700d3444fae2ca3c91db0a08713054b95808029653fc18574",
    "diffsort/functional.py": "4c8ff02f74d6fe1e0b53a9685954cc25ac8f3e67ec46d7d52fdfa9f314533676",
    "diffsort/networks.py": "1680cf2a0e90a2d158991b91a99983ab4bb2e150feb773d6d440f4423b3b9c85",
}


class RankingDependencyError(RuntimeError):
    """The optional sorter is absent, unqualified or numerically unusable."""


def _import_diffsort():
    try:
        return importlib.import_module("diffsort")
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise RankingDependencyError(
            "Ranking requires the optional 'diffsort' package "
            f"(qualified release {QUALIFIED_VERSION}); install the 'ranking' extra. "
            f"Original error: {exc}"
        ) from exc


def _module_source_bytes(module: object, relative: str) -> bytes:
    """Read one packaged source file from whatever loader actually loaded it.

    A plain directory install exposes ``.py`` files, but a zip import (or another
    source-readable loader) does not: its ``__file__`` lives inside an archive, so
    ``Path.is_file()`` is False even though the source is perfectly readable.
    Rejecting that merely because it is not a regular file would treat "not a
    plain file" as "unverifiable".
    """
    name = f"diffsort.{Path(relative).stem}"
    loaded = sys.modules.get(name)
    if loaded is not None and getattr(loaded, "__file__", None):
        loader = getattr(getattr(loaded, "__spec__", None), "loader", None)
        getter = getattr(loader, "get_source", None)
        if callable(getter):
            try:
                source = getter(name)
            except (ImportError, OSError) as exc:
                raise RankingDependencyError(
                    f"Loaded diffsort module {name!r} did not expose its source: {exc}"
                ) from exc
            if isinstance(source, str):
                return source.encode("utf-8")
        path = Path(loaded.__file__)
        if path.is_file():
            return path.read_bytes()
    # Fall back to the neighbour of the package __init__ (plain installs).
    root = Path(module.__file__).resolve().parent
    target = root / Path(relative).name
    if target.is_file():
        return target.read_bytes()
    raise RankingDependencyError(
        f"Cannot verify the loaded diffsort implementation: {relative} is not "
        "readable through its loader and not present beside the package. A "
        "source-less/bytecode-only distribution is unsupported for enabled ranking."
    )


def _source_digests() -> dict[str, str]:
    """SHA-256 of the loaded package's source files, keyed by packaged path."""
    module = _import_diffsort()
    digests: dict[str, str] = {}
    for relative in QUALIFIED_SOURCE_SHA256:
        digests[relative] = hashlib.sha256(
            _module_source_bytes(module, relative)
        ).hexdigest()
    return digests


def probe_ranking_dependency(device: torch.device) -> dict[str, object]:
    """Lazily verify the optional package and run one tiny forward/backward.

    Preserves every caller RNG stream: no model, dataset or global seed is
    touched, and the probe tensors are freshly created.
    """
    module = _import_diffsort()
    version = str(getattr(module, "__version__", "") or "")
    try:
        from importlib.metadata import version as _dist_version

        version = _dist_version("diffsort")
    except Exception:  # pragma: no cover - metadata is optional
        pass

    digests = _source_digests()
    mismatched = {
        name: (digests[name], expected)
        for name, expected in QUALIFIED_SOURCE_SHA256.items()
        if digests[name] != expected
    }
    if mismatched:
        raise RankingDependencyError(
            "The loaded diffsort implementation does not match the qualified source "
            f"digests: {sorted(mismatched)}. Install the qualified release instead of "
            "a modified or unverifiable copy."
        )
    if version and version != QUALIFIED_VERSION:
        raise RankingDependencyError(
            f"diffsort {version} is not the qualified release {QUALIFIED_VERSION}; "
            "package identity is part of the saved ranking recipe."
        )

    cosines = torch.tensor(
        [[0.4, -0.7, 0.9, -0.2, 0.1]], dtype=torch.float32, device=device
    )
    cache: dict = {}
    values, permutation = sort_values_and_permutation(cosines, cache=cache)
    if not torch.isfinite(values).all() or not torch.isfinite(permutation).all():
        raise RankingDependencyError(
            "The tiny sorter probe produced a non-finite permutation on this device."
        )
    # Both the forward and the backward of the probe run on the selected device:
    # a CPU-only sorter pass would not be evidence for that device.
    cosines = cosines.clone().requires_grad_(True)
    values, permutation = sort_values_and_permutation(cosines, cache={})
    loss = -(permutation.clamp_min(torch.finfo(torch.float32).tiny).log()).sum()
    loss.backward()
    if cosines.grad is None or not torch.isfinite(cosines.grad).all():
        raise RankingDependencyError(
            "The tiny sorter probe produced a non-finite gradient on this device."
        )

    return {
        "name": "diffsort",
        "version": version or QUALIFIED_VERSION,
        "wheel_sha256": QUALIFIED_WHEEL_SHA256,
        "source_sha256": digests,
        "device": str(device),
        "probed_on": str(cosines.device),
        "probed_dtype": str(cosines.dtype),
        "network": SORT_NETWORK,
        "distribution": SORT_DISTRIBUTION,
        "steepness": SORT_STEEPNESS,
        "direction": SORT_DIRECTION,
    }


def _validate_reference_count(reference_count: int) -> None:
    if not (MIN_REFERENCE_POINTS <= reference_count <= MAX_REFERENCE_POINTS):
        raise RankingDependencyError(
            "The reference count must be within "
            f"{MIN_REFERENCE_POINTS}..{MAX_REFERENCE_POINTS}, got {reference_count}. "
            "Padding to a power of two or repeating references is not allowed."
        )


def _build_sorter(reference_count: int, device: torch.device, dtype: torch.dtype):
    module = _import_diffsort()
    sorter = module.DiffSortNet(
        SORT_NETWORK,
        reference_count,
        device=str(device),
        steepness=SORT_STEEPNESS,
        distribution=SORT_DISTRIBUTION,
    )
    return sorter


def _cached_sorter(reference_count: int, device: torch.device, dtype: torch.dtype):
    """Return a cached sort network.

    Only the network constants are stored, so an evicted and rebuilt sorter must
    reproduce the same outputs and gradients for the same inputs.
    """
    cache: dict = _SORTER_CACHE
    key = (reference_count, str(device), str(dtype))
    if key in cache:
        cache[key] = cache.pop(key)  # move to most-recently-used
        return cache[key]
    sorter = _build_sorter(reference_count, device, dtype)
    cache[key] = sorter
    while len(cache) > CACHE_SIZE:
        cache.pop(next(iter(cache)))
    return sorter


_SORTER_CACHE: dict = {}


def sort_values_and_permutation(
    cosines: torch.Tensor, *, cache: dict | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort cosine rows ascending and return ``(sorted_values, permutation)``.

    ``cosines`` is ``[N, R]``; ``permutation`` is ``[N, R, R]`` with
    ``sorted = cosines @ permutation``. The caller owns the loss: no clamp and no
    log happen here.
    """
    if cosines.ndim != 2:
        raise RankingDependencyError(
            f"cosines must be [N, R], got {tuple(cosines.shape)}."
        )
    reference_count = int(cosines.shape[1])
    _validate_reference_count(reference_count)
    if not torch.isfinite(cosines).all():
        raise RankingDependencyError(
            "cosines contain non-finite values; refusing to sort a corrupt field."
        )
    values = cosines if cosines.dtype is torch.float32 else cosines.to(torch.float32)

    if cache is None:
        sorter = _cached_sorter(reference_count, values.device, values.dtype)
    else:
        key = (reference_count, str(values.device), str(values.dtype))
        sorter = cache.get(key)
        if sorter is None:
            sorter = _build_sorter(reference_count, values.device, values.dtype)
            cache[key] = sorter
            while len(cache) > CACHE_SIZE:
                cache.pop(next(iter(cache)))
        else:
            cache[key] = cache.pop(key)

    sorted_values, permutation = sorter(values)
    _validate_permutation(permutation)
    return sorted_values, permutation


def _validate_permutation(permutation: torch.Tensor, *, context: str = "") -> None:
    """Reject non-finite, negative or non-stochastic permutations.

    Reads detached values so validation never touches a live autograd graph.
    """
    suffix = f" ({context})" if context else ""
    values = permutation.detach()
    finite, minimum, residual, column_residual = torch.stack((
        torch.isfinite(values).all(), values.min(),
        (values.sum(-1) - 1).abs().max(), (values.sum(-2) - 1).abs().max(),
    )).cpu().tolist()
    if not finite:
        raise RankingDependencyError(
            f"The sorter produced non-finite probabilities{suffix}."
        )
    if minimum < 0.0:
        raise RankingDependencyError(
            f"The sorter produced negative probabilities{suffix}."
        )
    if max(residual, column_residual) > STOCHASTICITY_TOLERANCE:
        raise RankingDependencyError(
            "The sorter produced non-stochastic probabilities"
            f"{suffix}: max residual {max(residual, column_residual):.3e} exceeds "
            f"{STOCHASTICITY_TOLERANCE:.1e}."
        )


def row_cross_entropy(
    student: torch.Tensor, teacher: torch.Tensor, *, context: str = ""
) -> torch.Tensor:
    """Per-query row-CE mean ``-(1/R) sum_{j,k} P_t[j,k] log P_s[j,k]``.

    The student log is taken unclamped, so a zero student probability is a
    numerical failure rather than a silently floored value.
    """
    _validate_permutation(teacher, context=f"teacher target {context}".strip())
    _validate_permutation(student, context=f"student prediction {context}".strip())
    if bool((student.detach() <= PROBABILITY_FLOOR).any()):
        raise RankingDependencyError(
            "The student permutation contains a non-positive probability"
            f"{' (' + context + ')' if context else ''}; log is undefined and no "
            "clamp is applied."
        )
    reference_count = int(student.shape[-1])
    cells = -teacher.detach() * student.log()
    return cells.sum(-1).sum(-1) / float(reference_count)


class NeighborRankingObjective(torch.nn.Module):
    """Independent student/EMA-teacher projection heads and the ranking loss.

    Training-only: the heads live outside the encoder state dicts so exported
    embeddings keep their current meaning. The teacher head takes no gradient and
    follows the existing teacher EMA momentum after successful steps.
    """

    def __init__(
        self, hidden_dim: int, *, out_dim: int = 256, hidden: int = 2048
    ) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.student_head = self._build_head(self.hidden_dim, self.out_dim, hidden)
        self.teacher_head = self._build_head(self.hidden_dim, self.out_dim, hidden)
        self.teacher_head.load_state_dict(self.student_head.state_dict(), strict=True)
        for parameter in self.teacher_head.parameters():
            parameter.requires_grad_(False)
        #: Raw (pre-L2) teacher projection norm minimum from the *last single
        #: view* projection. ``compute`` aggregates it per contributing view.

    @staticmethod
    def _build_head(dim: int, out_dim: int, hidden: int) -> torch.nn.Sequential:
        """Three biased Linear layers with GELU, NeCo's 2048/256 shape."""
        import torch.nn as nn

        head = nn.Sequential(
            nn.Linear(dim, int(hidden), bias=True),
            nn.GELU(),
            nn.Linear(int(hidden), int(hidden), bias=True),
            nn.GELU(),
            nn.Linear(int(hidden), int(out_dim), bias=True),
        )
        for module in head:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        return head

    def trainable_parameters(self):
        """The student head only; the teacher head is never optimized."""
        return [p for p in self.student_head.parameters() if p.requires_grad]

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """EMA the teacher head after a successful optimizer step."""
        for teacher, student in zip(
            self.teacher_head.parameters(), self.student_head.parameters()
        ):
            teacher.mul_(float(momentum)).add_(student.detach(), alpha=1.0 - float(momentum))

    def project_field(self, grid: torch.Tensor, *, return_norms: bool = False):
        """Project once; compute validates only its selected student centres."""
        finite_input = torch.isfinite(grid).all(dim=-1)
        projected = self.student_head(torch.where(torch.isfinite(grid), grid, 0.0).float())
        norms = torch.where(finite_input, projected.detach().norm(dim=-1), torch.nan)
        if not return_norms:
            self._check_projection_norm(norms, "student")
        # Original norms still guard every participant before this field is consumed.
        field = torch.nn.functional.normalize(
            torch.where(torch.isfinite(projected), projected, 0.0), dim=-1,
        )
        return (field, norms) if return_norms else field

    @torch.no_grad()
    def project_teacher_field(self, grid: torch.Tensor, *, return_norms: bool = False):
        """Project once; optionally return detached pre-L2 norms for pair support."""
        finite_input = torch.isfinite(grid).all(dim=-1)
        projected = self.teacher_head(torch.where(torch.isfinite(grid), grid, 0.0).float())
        norms = torch.where(finite_input, projected.detach().norm(dim=-1), torch.nan)
        if not return_norms:
            self._check_projection_norm(norms, "teacher")
        field = torch.nn.functional.normalize(
            torch.where(torch.isfinite(projected), projected, 0.0), dim=-1,
        )
        return (field, norms) if return_norms else field

    @staticmethod
    def _check_projection_norm(norms: torch.Tensor, role: str, context: str = "") -> None:
        """Reject a degenerate projection *before* L2 hides its magnitude.

        ``F.normalize`` maps any direction, including one produced by a
        near-zero projection, to a unit vector; the recipe requires the actual
        pre-L2 norm to exceed the shared feature-norm floor.
        """
        from otuformer.training.dense_objective import (
            FEATURE_NORM_FLOOR,
            DenseNumericalError,
        )

        finite, minimum = torch.stack((torch.isfinite(norms).all(), norms.min())).cpu().tolist()
        if not finite or minimum <= FEATURE_NORM_FLOOR:
            raise DenseNumericalError(
                f"The {role} ranking projection norm "
                f"{minimum:.3e} does not exceed {FEATURE_NORM_FLOOR:.1e} {context}; "
                "normalising it would fabricate a direction from numerical noise."
            )

    def compute(
        self,
        student_grids: dict[int, torch.Tensor],
        teacher_grids: dict[int, torch.Tensor],
        pairs,
        *,
        patch_hw: tuple[int, int],
        reference_cap: int,
        generator: torch.Generator,
        query_chunk: int = 0,
    ):
        """Pair-conditioned reference pools and the unclamped row-CE mean."""
        from otuformer.training import dense_objective as _objective
        from otuformer.training.dense_objective import DenseLossResult

        _validate_reference_count(int(reference_cap))
        per_pair: list[list[torch.Tensor]] = []
        branch_counts = _objective._branch_counts(pairs)
        probabilities: list[torch.Tensor] = []
        student_fields: dict[int, torch.Tensor] = {}
        teacher_fields: dict[int, torch.Tensor] = {}
        student_projection_norms: dict[int, torch.Tensor] = {}
        teacher_projection_norms: dict[int, torch.Tensor] = {}
        participating_projection_minima: list[torch.Tensor] = []
        populations: list[int] = []
        reference_counts: list[int] = []
        reference_shortages = 0
        unavailable_pairs = 0
        selected = 0
        eligible = 0
        contributing_samples = 0
        contributing_pairs = 0
        min_norm: list[float] = []
        min_rho: list[float] = []

        for pair in pairs:
            eligible += sum(int(sample.eligible_count) for sample in pair.samples
                            if sample.student_indices.numel() > 0)
            population = sum(int(sample.student_indices.numel()) for sample in pair.samples)
            if population < MIN_REFERENCE_POINTS:
                populations.append(population)
                unavailable_pairs += 1
                per_pair.append([])
                continue
            student_grid = student_grids[pair.student_view]
            teacher_grid = teacher_grids[pair.teacher_view]
            if pair.student_view not in student_fields:
                field, norms = self.project_field(student_grid, return_norms=True)
                student_fields[pair.student_view] = field
                student_projection_norms[pair.student_view] = norms
            if pair.teacher_view not in teacher_fields:
                field, norms = self.project_teacher_field(teacher_grid, return_norms=True)
                teacher_fields[pair.teacher_view] = field
                teacher_projection_norms[pair.teacher_view] = norms
            student_field = student_fields[pair.student_view]
            teacher_field = teacher_fields[pair.teacher_view]
            context = f"pair {pair.teacher_view}->{pair.student_view}"
            student_norms = student_projection_norms[pair.student_view].flatten(1)
            selected_student_norms = [
                student_norms[row, sample.student_indices.to(student_norms.device)]
                for row, sample in enumerate(pair.samples) if sample.student_indices.numel()
            ]
            projection_min = _projection_support_min(
                teacher_projection_norms[pair.teacher_view], pair, patch_hw=patch_hw,
            )
            try:
                self._check_projection_norm(torch.cat(selected_student_norms), "student", context)
                self._check_projection_norm(projection_min, "teacher", context)
            except _objective.DenseNumericalError:
                # Diagnose original rows only on failure; healthy projections remain cached.
                for row, sample in enumerate(pair.samples):
                    if not sample.student_indices.numel():
                        continue
                    self._check_projection_norm(
                        student_norms[row, sample.student_indices.to(student_norms.device)],
                        "student", f"{context} row {row}",
                    )
                    self._check_projection_norm(
                        _projection_support_min(teacher_projection_norms[pair.teacher_view],
                                                pair, patch_hw=patch_hw, only_row=row),
                        "teacher", f"{context} row {row}",
                    )
                raise
            # Sample THIS pair once; the same teacher points are targets and pool.
            points = _objective._pair_points(
                student_field, teacher_field, pair, patch_hw=patch_hw,
            )
            lengths = points[2] if points is not None else []
            population = sum(lengths)
            populations.append(population)
            student_points, flat_pool, lengths, stats = points
            participating_projection_minima.append(projection_min)
            # Keep the sampled teacher diagnostics this pair already computed.
            if stats.get("min_norm") is not None:
                min_norm.append(float(stats["min_norm"]))
            if stats.get("min_rho") is not None:
                min_rho.append(float(stats["min_rho"]))
            student_points, _ = _objective._unit_rows(
                student_points, context=f"ranking pair {pair.teacher_view}->{pair.student_view}",
            )
            resolved_cap = min(int(reference_cap), population)
            reference_counts.append(int(resolved_cap))
            if int(resolved_cap) < int(reference_cap):
                # A short pool is a measured shortage, not a padded list.
                reference_shortages += len(lengths)

            cosines_student: list[torch.Tensor] = []
            cosines_teacher: list[torch.Tensor] = []
            uniform = torch.ones(population, dtype=torch.float64)
            for student_row, teacher_row in zip(
                student_points.split(lengths), flat_pool.split(lengths),
            ):
                if resolved_cap >= population:
                    # Below/at the cap: canonical order, no extra draw.
                    reference = flat_pool
                else:
                    index = torch.multinomial(
                        uniform,
                        resolved_cap,
                        replacement=False,
                        generator=generator,
                    )
                    reference = flat_pool[index]
                reference = reference.detach()
                # Preserve dot-product arithmetic: near ties amplify bmm rounding in the sorter.
                cosines_student.append(student_row @ reference.T)
                cosines_teacher.append(teacher_row @ reference.T)
            cosines_s = torch.cat(cosines_student)
            cosines_t = torch.cat(cosines_teacher)
            chunk_losses = []
            step = int(query_chunk) if query_chunk else population
            for start in range(0, population, step):
                stop = start + step
                _, permutation_s = sort_values_and_permutation(
                    cosines_s[start:stop].contiguous(), cache=None,
                )
                with torch.no_grad():
                    _, permutation_t = sort_values_and_permutation(
                        cosines_t[start:stop].contiguous(), cache=None,
                    )
                chunk_losses.append(row_cross_entropy(
                    permutation_s, permutation_t,
                    context=f"pair {pair.teacher_view}->{pair.student_view}",
                ))
                probabilities.append(permutation_s.detach().min())
            losses = torch.cat(chunk_losses)
            per_pair.append([row.mean() for row in losses.split(lengths)])
            selected += population
            contributing_samples += len(lengths)
            contributing_pairs += 1
            _objective._contribute_branch(branch_counts, pair, lengths)

        raw = _reduce(per_pair)
        min_probability = float(torch.stack(probabilities).min()) if probabilities else None
        return DenseLossResult(
            raw=raw,
            statistics={
                "selected_points": selected,
                "eligible_points": eligible,
                "contributing_samples": contributing_samples,
                "contributing_pairs": contributing_pairs,
                **_objective._finish_branch_counts(branch_counts),
                # ``None`` populations are unavailable, never measured zeros.
                "reference_population": max(populations) if populations else None,
                "reference_population_min": min(populations) if populations else None,
                "reference_count": max(reference_counts) if reference_counts else None,
                "reference_count_min": (
                    min(reference_counts) if reference_counts else None
                ),
                "reference_shortage_samples": reference_shortages,
                "reference_unavailable_pairs": unavailable_pairs,
                "reference_cap": int(reference_cap),
                "min_probability": min_probability,
                # This objective's OWN numerical diagnostics, in the projected
                # ranking space: the Coordinate values describe a different
                # feature space and cannot stand in for these.
                "teacher_min_norm": (
                    min(min_norm) if min_norm else None
                ),
                "teacher_min_rho": min(min_rho) if min_rho else None,
                # ``teacher_min_norm`` is the *interpolated* field's norm; this
                # one is the raw projection before its per-patch L2, which is
                # what the projection guard actually tests.
                "teacher_projection_min_norm": (
                    float(torch.stack(participating_projection_minima).min())
                    if participating_projection_minima else None
                ),
            },
        )


def _projection_support_min(norms: torch.Tensor, pair, *, patch_hw,
                            only_row: int | None = None) -> torch.Tensor:
    """Minimum over real queries' positive-weight bilinear contributors only."""
    from otuformer.training.dense_objective import _bilinear_contributors

    samples = [(row, sample) for row, sample in enumerate(pair.samples)
               if sample.student_indices.numel() and (only_row is None or row == only_row)]
    xy = torch.cat([sample.teacher_xy.float() for _, sample in samples]).to(norms.device)
    batch = torch.cat([
        torch.full((sample.student_indices.numel(),), row, dtype=torch.int64)
        for row, sample in samples
    ]).to(norms.device)
    height, width = norms.shape[1:]
    row0, wy, col0, wx = _bilinear_contributors(xy, patch_hw, (height, width))
    minima = []
    for rows, ry in ((row0, 1 - wy), (row0 + 1, wy)):
        for cols, cx in ((col0, 1 - wx), (col0 + 1, wx)):
            active = (ry * cx > 0) & (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            values = norms[batch, rows.clamp(0, height - 1), cols.clamp(0, width - 1)]
            checked = torch.where(torch.isfinite(values), values, torch.nan)
            minima.append(torch.where(active, checked, torch.inf).min())
    return torch.stack(minima).min()


def _reduce(per_pair: list[list[torch.Tensor]]) -> torch.Tensor | None:
    from otuformer.training.dense_objective import reduce_dense_losses

    return reduce_dense_losses(per_pair)
