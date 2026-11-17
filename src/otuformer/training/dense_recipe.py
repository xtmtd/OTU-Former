"""Dense-only settings, RNG streams and the closed checkpoint bundles.

This module owns the *effective* dense recipe for one invocation: it resolves the
requested weights and caps once, derives the independent CPU RNG streams, and
serialises/validates the closed ``dense_training_config`` / ``dense_training``
bundles. :mod:`otuformer.training.pretrain_recipe` remains the invocation owner;
this module never imports the CLI, the optional ranking dependency or the loop.

Policy identity strings are frozen here. They are part of the recipe, so a
changed threshold changes the identity and invalidates same-recipe resume.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from otuformer.constants import SPATIAL_SUPPORT_VERSION

CONFIG_KEY = "dense_training_config"
STATE_KEY = "dense_training"
BUNDLE_VERSION = 1

QUERY_DEFAULT = 49
QUERY_BOUNDS = (1, 49)
REFERENCE_DEFAULT = 16
REFERENCE_BOUNDS = (2, 49)

COORDINATE_WEIGHT_DEFAULT = 0.0
RANKING_WEIGHT_DEFAULT = 0.0

RNG_POLICY = "sha256-seed-stream-v1"
GEOMETRY_POLICY = "realized-affine-complete-support-v1"
#: Reuse the existing geometry identity instead of duplicating the string.
SUPPORT_POLICY = SPATIAL_SUPPORT_VERSION
QUERY_POLICY = "student-centers-without-replacement-v1"
REDUCTION_POLICY = "point-sample-pair-mean-v1"
# Atomic by design: the three numeric thresholds share one identity so they can
# never be relaxed independently of the saved recipe.
NUMERIC_POLICY = "float32-norm-rho1e-6-stochastic1e-5-v1"
COORDINATE_FEATURES = "final-norm-raw-sample-l2-v1"

RANKING_POOL_POLICY = "pair-aligned-selected-teacher-v1"
RANKING_REFERENCE_POLICY = "per-sample-without-replacement-self-allowed-v1"
RANKING_FEATURE_POLICY = "project-l2-sample-cosine-v1"
RANKING_SORT = {
    "network": "bitonic",
    "distribution": "cauchy",
    "steepness_student": 100.0,
    "steepness_teacher": 100.0,
    "direction": "ascending-cosine",
    "permutation": "input-by-rank",
    "loss": "row-ce-unclamped-v1",
}
RANKING_HEAD = {
    "layers": 3,
    "hidden_dim": 2048,
    "output_dim": 256,
    "activation": "gelu",
    "bias": True,
}

CONFIG_BASE_KEYS = frozenset(
    {
        "version",
        "coordinate_weight",
        "ranking_weight",
        "query_cap",
        "batch_size",
        "steps_per_epoch",
        "objectives",
        "rng_policy",
        "geometry_policy",
        "support_policy",
        "query_policy",
        "reduction_policy",
        "numeric_policy",
        "coordinate_features",
        "ranking_config",
    }
)
CONFIG_REQUIRED_KEYS = CONFIG_BASE_KEYS - {"ranking_config"}
RANKING_CONFIG_KEYS = frozenset(
    {
        "version",
        "reference_cap",
        "pool_policy",
        "reference_policy",
        "head",
        "feature_policy",
        "sort",
        "dependency",
    }
)
STATE_REQUIRED_KEYS = frozenset({"version", "completed_steps", "query_rng"})
STATE_RANKING_KEYS = frozenset(
    {"student_head", "teacher_head", "reference_rng", "contributed_steps"}
)


@dataclass(frozen=True)
class DenseSettings:
    """One resolution of the dense weights and caps for this invocation."""

    coordinate_weight: float = COORDINATE_WEIGHT_DEFAULT
    ranking_weight: float = RANKING_WEIGHT_DEFAULT
    query_cap: int = QUERY_DEFAULT
    reference_cap: int | None = None

    @property
    def enabled(self) -> bool:
        return self.coordinate_weight > 0.0 or self.ranking_weight > 0.0

    @property
    def ranking_enabled(self) -> bool:
        return self.ranking_weight > 0.0


@dataclass
class DenseRuntime:
    """Independent CPU RNG streams plus the ranking contribution counter."""

    query_generator: torch.Generator
    reference_generator: torch.Generator | None = None
    ranking_contributed_steps: int = 0


def _weight(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{name} must be a real number, got {type(value).__name__}."
        )
    resolved = float(value)
    if not torch.isfinite(torch.tensor(resolved)):
        raise ValueError(f"{name} must be finite, got {resolved!r}.")
    if resolved < 0.0:
        raise ValueError(f"{name} must be non-negative, got {resolved!r}.")
    return resolved


def _cap(value: Any, name: str, bounds: tuple[int, int]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}.")
    low, high = bounds
    if not (low <= value <= high):
        raise ValueError(f"{name} must be within {low}..{high}, got {value}.")
    return int(value)


def _saved_config(config: dict[str, Any] | None) -> Mapping[str, Any] | None:
    """Return the saved dense config bundle, rejecting a partial pair."""
    if config is None:
        return None
    payload = config.get("config")
    if not isinstance(payload, dict):
        return None
    bundle = payload.get(CONFIG_KEY)
    if bundle is None:
        return None
    if not isinstance(bundle, dict):
        raise ValueError("config.dense_training_config must be a mapping.")
    return bundle


def _requested(args: Any, name: str) -> Any | None:
    """Return the requested value when the invocation supplied it explicitly."""
    if not bool(getattr(args, f"{name}_explicit", False)):
        return None
    return getattr(args, name, None)


def resolve_dense_settings(
    args: Any, checkpoint: dict[str, Any] | None
) -> DenseSettings:
    """Resolve the effective dense settings exactly once, before output setup.

    An omitted weight or cap inherits the saved value on resume; an explicit
    conflict is rejected because enabling, disabling or rescaling an objective
    requires a new run.
    """
    saved = _saved_config(checkpoint)
    # Requested-vs-omitted is decided by the CLI's explicitness flags, never by
    # comparing values: an explicit 0.0 is a real request that must conflict
    # with a saved positive weight instead of looking inherited.
    requested_coordinate = _requested(args, "coordinate_weight")
    requested_ranking = _requested(args, "ranking_weight")
    requested_query = _requested(args, "dense_query_points")
    requested_reference = _requested(args, "ranking_reference_points")

    if saved is None:
        if checkpoint is not None:
            # No saved bundle means both objectives were disabled; a resume can
            # never upgrade the recipe.
            for name, value in (
                ("coordinate_weight", requested_coordinate),
                ("ranking_weight", requested_ranking),
            ):
                if value not in (None, 0, 0.0):
                    raise ValueError(
                        f"Cannot enable {name} on resume: the saved run has no dense "
                        "training bundle. Start a new run instead."
                    )
        coordinate = _weight(
            COORDINATE_WEIGHT_DEFAULT if requested_coordinate is None else requested_coordinate,
            "coordinate_weight",
        )
        ranking = _weight(
            RANKING_WEIGHT_DEFAULT if requested_ranking is None else requested_ranking,
            "ranking_weight",
        )
        query = _cap(
            QUERY_DEFAULT if requested_query is None else requested_query,
            "dense_query_points",
            QUERY_BOUNDS,
        )
        if coordinate == 0.0 and ranking == 0.0 and _requested(
            args, "dense_query_points"
        ) is not None:
            raise ValueError(
                "--dense-query-points is not applicable when both dense weights are "
                "0; omit it or enable an objective."
            )
        if ranking == 0.0 and _requested(args, "ranking_reference_points") is not None:
            raise ValueError(
                "--ranking-reference-points is not applicable when "
                "--ranking-weight is 0; omit it or enable ranking."
            )
        reference = (
            None
            if ranking == 0.0
            else _cap(
                REFERENCE_DEFAULT if requested_reference is None else requested_reference,
                "ranking_reference_points",
                REFERENCE_BOUNDS,
            )
        )
        return DenseSettings(coordinate, ranking, query, reference)

    saved_coordinate = _weight(saved["coordinate_weight"], "coordinate_weight")
    saved_ranking = _weight(saved["ranking_weight"], "ranking_weight")
    saved_query = _cap(saved["query_cap"], "dense_query_points", QUERY_BOUNDS)
    saved_reference = (
        _cap(saved["ranking_config"]["reference_cap"], "ranking_reference_points", REFERENCE_BOUNDS)
        if saved_ranking > 0.0
        else None
    )

    coordinate = (
        saved_coordinate
        if requested_coordinate is None
        else _weight(requested_coordinate, "coordinate_weight")
    )
    ranking = (
        saved_ranking
        if requested_ranking is None
        else _weight(requested_ranking, "ranking_weight")
    )
    query = (
        saved_query
        if requested_query is None
        else _cap(requested_query, "dense_query_points", QUERY_BOUNDS)
    )
    reference = (
        saved_reference
        if requested_reference is None
        else _cap(requested_reference, "ranking_reference_points", REFERENCE_BOUNDS)
    )
    # Applicability applies on resume exactly as on a new run: an explicitly
    # supplied inactive option is a request this recipe cannot honour.
    if ranking == 0.0 and _requested(args, "ranking_reference_points") is not None:
        raise ValueError(
            "--ranking-reference-points is not applicable when --ranking-weight is "
            "0; omit it or enable ranking."
        )
    if coordinate == 0.0 and ranking == 0.0 and (
        _requested(args, "dense_query_points") is not None
    ):
        raise ValueError(
            "--dense-query-points is not applicable when both dense weights are 0; "
            "omit it or enable an objective."
        )
    for name, resolved, previous in (
        ("coordinate_weight", coordinate, saved_coordinate),
        ("ranking_weight", ranking, saved_ranking),
        ("dense_query_points", query, saved_query),
        ("ranking_reference_points", reference, saved_reference),
    ):
        if previous is not None and resolved != previous:
            raise ValueError(
                f"{name}={resolved!r} conflicts with the saved recipe "
                f"({previous!r}). Changing a dense objective, weight or cap "
                "requires a new run."
            )
    return DenseSettings(coordinate, ranking, query, reference)


def _stream_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def new_dense_runtime(settings: DenseSettings, *, seed: int) -> DenseRuntime | None:
    """Create the enabled-only RNG streams; ``None`` when both weights are zero."""
    if not settings.enabled:
        return None
    query = torch.Generator(device="cpu")
    query.manual_seed(_stream_seed(int(seed), "dense-query-v1"))
    reference: torch.Generator | None = None
    if settings.ranking_enabled:
        reference = torch.Generator(device="cpu")
        reference.manual_seed(_stream_seed(int(seed), "dense-reference-v1"))
    return DenseRuntime(query_generator=query, reference_generator=reference)


def dense_config_bundle(
    settings: DenseSettings,
    *,
    batch_size: int,
    steps_per_epoch: int,
    dependency: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Serialise the effective dense config; ``None`` when both weights are zero."""
    if not settings.enabled:
        return None
    objectives = []
    if settings.coordinate_weight > 0.0:
        objectives.append("coordinate")
    if settings.ranking_weight > 0.0:
        objectives.append("ranking")
    bundle: dict[str, Any] = {
        "version": BUNDLE_VERSION,
        "coordinate_weight": float(settings.coordinate_weight),
        "ranking_weight": float(settings.ranking_weight),
        "query_cap": int(settings.query_cap),
        "batch_size": int(batch_size),
        "steps_per_epoch": int(steps_per_epoch),
        "objectives": objectives,
        "rng_policy": RNG_POLICY,
        "geometry_policy": GEOMETRY_POLICY,
        "support_policy": SUPPORT_POLICY,
        "query_policy": QUERY_POLICY,
        "reduction_policy": REDUCTION_POLICY,
        "numeric_policy": NUMERIC_POLICY,
        "coordinate_features": COORDINATE_FEATURES,
    }
    if settings.ranking_enabled:
        if dependency is None:
            raise ValueError(
                "A positive ranking weight requires the qualified dependency record; "
                "run the sorter qualification and pass its identity."
            )
        # Record identity only: the probe's device/dtype evidence is not part of
        # the closed schema.
        dependency_record = {
            key: dependency[key] for key in _DEPENDENCY_KEYS if key in dependency
        }
        bundle["ranking_config"] = {
            "version": BUNDLE_VERSION,
            "reference_cap": int(settings.reference_cap),
            "pool_policy": RANKING_POOL_POLICY,
            "reference_policy": RANKING_REFERENCE_POLICY,
            "head": dict(RANKING_HEAD),
            "feature_policy": RANKING_FEATURE_POLICY,
            "sort": dict(RANKING_SORT),
            "dependency": dependency_record,
        }
    return bundle


def dense_state_bundle(
    runtime: DenseRuntime,
    *,
    completed_steps: int,
    ranking: Any | None,
) -> dict[str, Any]:
    """Serialise the dense RNG/state bundle at the epoch checkpoint boundary."""
    if runtime is None:
        raise ValueError("Cannot serialise dense state for a disabled runtime.")
    state: dict[str, Any] = {
        "version": BUNDLE_VERSION,
        "completed_steps": int(completed_steps),
        "query_rng": runtime.query_generator.get_state(),
    }
    if runtime.reference_generator is not None:
        if ranking is None:
            raise ValueError(
                "An enabled ranking run must serialise its two head state dicts."
            )
        state["ranking"] = {
            "student_head": ranking.student_head.state_dict(),
            "teacher_head": ranking.teacher_head.state_dict(),
            "reference_rng": runtime.reference_generator.get_state(),
            "contributed_steps": int(runtime.ranking_contributed_steps),
        }
    return state


def restore_dense_runtime(
    settings: DenseSettings, state: Mapping[str, Any]
) -> DenseRuntime:
    """Restore the RNG streams without consuming a draw."""
    if not settings.enabled:
        raise ValueError("Cannot restore dense state for a disabled recipe.")
    state_missing = sorted(STATE_REQUIRED_KEYS - set(state))
    state_unexpected = sorted(set(state) - (STATE_REQUIRED_KEYS | {"ranking"}))
    if state_missing or state_unexpected:
        raise ValueError(
            "dense_training fields do not match version 1 "
            f"(missing={state_missing}, unexpected={state_unexpected})."
        )
    query = torch.Generator(device="cpu")
    query.set_state(_as_rng_state(state["query_rng"], "query_rng"))
    reference: torch.Generator | None = None
    contributed = 0
    if settings.ranking_enabled:
        if "ranking" not in state:
            raise ValueError(
                "An enabled ranking run requires its saved head and reference RNG state."
            )
        ranking_state = state["ranking"]
        _require_keys(ranking_state, STATE_RANKING_KEYS, "dense_training.ranking")
        reference = torch.Generator(device="cpu")
        reference.set_state(_as_rng_state(ranking_state["reference_rng"], "reference_rng"))
        contributed = int(ranking_state["contributed_steps"])
    elif "ranking" in state:
        raise ValueError(
            "A Coordinate-only run must not carry ranking head/reference state."
        )
    return DenseRuntime(
        query_generator=query,
        reference_generator=reference,
        ranking_contributed_steps=contributed,
    )


def _require_keys(payload: Mapping[str, Any], required, label: str) -> None:
    missing = sorted(set(required) - set(payload))
    if missing:
        raise ValueError(f"{label} is missing required field(s) {missing}.")
    unexpected = sorted(set(payload) - set(required))
    if unexpected:
        raise ValueError(f"{label} has unexpected field(s) {unexpected}.")


def _as_rng_state(payload: Any, label: str) -> torch.Tensor:
    if not isinstance(payload, torch.Tensor) or payload.dtype is not torch.uint8:
        raise ValueError(f"{label} must be a restorable CPU generator ByteTensor.")
    state = payload.detach().clone().to("cpu")
    probe = torch.Generator(device="cpu")
    try:
        probe.set_state(state)
    except Exception as exc:  # pragma: no cover - torch-specific guard
        raise ValueError(f"{label} is not a restorable generator state: {exc}") from exc
    return state


def validate_dense_resume(
    config: Mapping[str, Any] | None,
    state: Mapping[str, Any] | None,
    *,
    iteration: int,
    batch_size: int,
    steps_per_epoch: int,
    dependency: Mapping[str, Any] | None = None,
    expected_support_versions: Mapping[str, Any] | None = None,
) -> None:
    """Validate a complete config/state pair against this invocation.

    Every condition is checked before output, log or CSV mutation so a rejected
    resume cannot damage an existing run directory. ``dependency`` is the
    identity of the implementation this executable actually loaded; a saved
    ranking recipe must match it, a version string alone is not evidence.
    """
    if config is None and state is None:
        return
    if config is None or state is None:
        raise ValueError(
            "dense_training and its config must be saved together; the checkpoint "
            "carries only one of the pair."
        )
    missing = sorted(CONFIG_REQUIRED_KEYS - set(config))
    unexpected = sorted(set(config) - CONFIG_BASE_KEYS)
    if missing or unexpected:
        raise ValueError(
            "dense_training_config fields do not match version "
            f"{BUNDLE_VERSION} (missing={missing}, unexpected={unexpected})."
        )
    ranking_config = config.get("ranking_config")
    if config["ranking_weight"] > 0.0:
        if ranking_config is None:
            raise ValueError(
                "A positive saved ranking weight requires its ranking_config bundle."
            )
        _require_keys(ranking_config, RANKING_CONFIG_KEYS, "dense_training_config.ranking_config")
        _validate_ranking_config(ranking_config, dependency=dependency)
    elif "ranking_config" in config:
        raise ValueError(
            "A Coordinate-only recipe must not carry a ranking_config bundle."
        )
    for name, expected in (
        ("rng_policy", RNG_POLICY),
        ("geometry_policy", GEOMETRY_POLICY),
        ("support_policy", SUPPORT_POLICY),
        ("query_policy", QUERY_POLICY),
        ("reduction_policy", REDUCTION_POLICY),
        ("coordinate_features", COORDINATE_FEATURES),
    ):
        if config.get(name) != expected:
            raise ValueError(
                f"The saved dense recipe records {name}={config.get(name)!r} but "
                f"this executable implements {expected!r}; the sampling, support "
                "or reduction would differ. Resume needs the matching recipe."
            )

    for label, payload in (
        ("dense_training_config", config),
        ("dense_training", state),
    ):
        version = payload.get("version")
        # ``True == 1`` and ``1.0 == 1`` both compare equal, so a loose check
        # would accept a boolean or float version as the real integer schema.
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError(
                f"Unsupported {label} version {version!r}; version must be the "
                f"integer {BUNDLE_VERSION}."
            )
        if version != BUNDLE_VERSION:
            raise ValueError(
                f"Unsupported {label} version {version!r}; "
                f"expected {BUNDLE_VERSION}."
            )
    for name in ("coordinate_weight", "ranking_weight"):
        _weight(config[name], name)
    for name, bounds in (
        ("query_cap", QUERY_BOUNDS),
    ):
        _cap(config[name], name, bounds)
    for name in ("batch_size", "steps_per_epoch"):
        _cap(config[name], name, (1, 10**9))

    if int(config["batch_size"]) != int(batch_size):
        raise ValueError(
            "The saved dense recipe used batch_size="
            f"{config['batch_size']} but this invocation requests {batch_size}; "
            "the ranking population and denominators would change."
        )
    if steps_per_epoch >= 0 and int(config["steps_per_epoch"]) != int(steps_per_epoch):
        raise ValueError(
            "The saved dense recipe used steps_per_epoch="
            f"{config['steps_per_epoch']} but this loader yields "
            f"{steps_per_epoch} step(s)."
        )
    if int(state["completed_steps"]) != int(iteration):
        raise ValueError(
            f"dense_training.completed_steps={state['completed_steps']} does not "
            f"match the checkpoint iteration {iteration}."
        )
    if config.get("numeric_policy") != NUMERIC_POLICY:
        raise ValueError(
            "The saved dense numerical policy differs from this executable's "
            "frozen policy; requalify before resuming."
        )
    _validate_dense_state(state, config)
    expected_objectives = []
    if config["coordinate_weight"] > 0.0:
        expected_objectives.append("coordinate")
    if config["ranking_weight"] > 0.0:
        expected_objectives.append("ranking")
    if list(config["objectives"]) != expected_objectives:
        raise ValueError(
            f"Saved objectives {config['objectives']} contradict the saved weights "
            f"{expected_objectives}."
        )


_HEAD_KEYS = frozenset(
    {"0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias"}
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DEPENDENCY_KEYS = frozenset(
    {"name", "version", "wheel_sha256", "source_sha256"}
)
#: Descriptive probe evidence: recorded, but not part of the recipe identity and
#: therefore not serialised into the closed config bundle.
_DEPENDENCY_EVIDENCE_KEYS = frozenset(
    {
        "device",
        "probed_on",
        "probed_dtype",
        "network",
        "distribution",
        "steepness",
        "direction",
    }
)


def _validate_ranking_config(
    ranking_config: Mapping[str, Any],
    *,
    dependency: Mapping[str, Any] | None,
) -> None:
    """Prove the saved ranking recipe is the one this executable implements."""
    if not isinstance(ranking_config, Mapping):
        raise ValueError("dense_training_config.ranking_config must be a mapping.")
    version = ranking_config["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(
            f"Unsupported ranking_config version {version!r}; version must be the "
            f"integer {BUNDLE_VERSION}."
        )
    if version != BUNDLE_VERSION:
        raise ValueError(
            "Unsupported ranking_config version "
            f"{ranking_config['version']!r}; expected {BUNDLE_VERSION}."
        )
    _cap(ranking_config["reference_cap"], "ranking_reference_points", REFERENCE_BOUNDS)
    for name, expected in (
        ("pool_policy", RANKING_POOL_POLICY),
        ("reference_policy", RANKING_REFERENCE_POLICY),
        ("feature_policy", RANKING_FEATURE_POLICY),
    ):
        if ranking_config[name] != expected:
            raise ValueError(
                f"The saved ranking recipe records {name}="
                f"{ranking_config[name]!r} but this executable implements "
                f"{expected!r}."
            )
    head = ranking_config["head"]
    if not isinstance(head, Mapping) or dict(head) != dict(RANKING_HEAD):
        raise ValueError(
            "The saved ranking head layout differs from the frozen "
            f"{dict(RANKING_HEAD)}; the objective would change."
        )
    # ``3.0 == 3`` and ``True == 1`` would compare equal, so a loose comparison
    # accepts a float layer count or a numeric bias flag as the frozen recipe.
    for name, value in head.items():
        expected = RANKING_HEAD[name]
        if type(value) is not type(expected):
            raise ValueError(
                f"The saved ranking head field {name}={value!r} has type "
                f"{type(value).__name__}, expected {type(expected).__name__}."
            )
    sort = ranking_config["sort"]
    if not isinstance(sort, Mapping) or dict(sort) != dict(RANKING_SORT):
        raise ValueError(
            "The saved sorter recipe differs from the frozen "
            f"{dict(RANKING_SORT)}; the ordering objective would change."
        )
    for name, value in sort.items():
        expected = RANKING_SORT[name]
        if type(value) is not type(expected):
            raise ValueError(
                f"The saved sorter field {name}={value!r} has type "
                f"{type(value).__name__}, expected {type(expected).__name__}."
            )
    saved_dependency = ranking_config["dependency"]
    _validate_dependency_record(saved_dependency, label="saved")
    if dependency is None:
        # Structural validation only: the pre-loader preflight cannot probe yet.
        # The loader-time call must supply the probed identity and compare it.
        return
    _validate_dependency_record(dependency, label="loaded")
    for name in ("name", "version"):
        if saved_dependency[name] != dependency[name]:
            raise ValueError(
                f"The saved ranking dependency {name}="
                f"{saved_dependency[name]!r} is not the loaded "
                f"{dependency[name]!r}; a version string alone is not proof of an "
                "equivalent recipe."
            )
    saved_sources = saved_dependency["source_sha256"]
    loaded_sources = dependency["source_sha256"]
    if dict(saved_sources) != dict(loaded_sources):
        mismatched = sorted(
            key
            for key in set(saved_sources) | set(loaded_sources)
            if saved_sources.get(key) != loaded_sources.get(key)
        )
        raise ValueError(
            "The loaded ranking implementation differs from the saved recipe "
            f"({mismatched}); install the qualified release instead of a modified "
            "or unverifiable copy."
        )


def _validate_dependency_record(record: Any, *, label: str) -> None:
    """Reject a malformed dependency identity (saved or just probed)."""
    if not isinstance(record, Mapping):
        raise ValueError(f"The {label} ranking dependency record must be a mapping.")
    missing = sorted(_DEPENDENCY_KEYS - set(record))
    if missing:
        raise ValueError(
            f"The {label} ranking dependency record is missing field(s) {missing}."
        )
    # Identity fields only: the probe also reports the device/dtype it ran on,
    # which is environment evidence rather than recipe identity.
    unexpected = sorted(set(record) - _DEPENDENCY_KEYS - _DEPENDENCY_EVIDENCE_KEYS)
    if unexpected:
        raise ValueError(
            f"The {label} ranking dependency record has unexpected field(s) "
            f"{unexpected}."
        )
    for name in ("name", "version"):
        value = record[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"The {label} ranking dependency {name} must be a non-empty string."
            )
    if not _SHA256_PATTERN.match(str(record["wheel_sha256"])):
        raise ValueError(
            f"The {label} ranking dependency wheel_sha256 must be 64 hex characters."
        )
    sources = record["source_sha256"]
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError(
            f"The {label} ranking dependency source_sha256 must be a non-empty mapping."
        )
    for key, value in sources.items():
        if not isinstance(key, str) or not _SHA256_PATTERN.match(str(value)):
            raise ValueError(
                f"The {label} ranking dependency source digest for {key!r} is not "
                "a 64-character hex digest."
            )


def _validate_dense_state(
    state: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    """Prove the saved dense state is complete and internally consistent."""
    completed = state.get("completed_steps")
    if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
        raise ValueError(
            "dense_training.completed_steps must be a non-negative integer, got "
            f"{completed!r}."
        )
    # Unknown top-level fields must fail here, in preflight, not later inside
    # restore_dense_runtime after output has already been touched.
    unexpected = sorted(set(state) - (STATE_REQUIRED_KEYS | {"ranking"}))
    if unexpected:
        raise ValueError(
            f"dense_training has unexpected field(s) {unexpected}."
        )
    _as_rng_state(state.get("query_rng"), "dense_training.query_rng")
    ranking_enabled = float(config["ranking_weight"]) > 0.0
    ranking_state = state.get("ranking")
    if not ranking_enabled:
        # Presence is the signal, not truthiness: a ``ranking: null`` key would
        # pass here and then be rejected by ``restore_dense_runtime`` after the
        # output directory exists.
        if "ranking" in state:
            raise ValueError(
                "A Coordinate-only recipe must not carry ranking head/reference "
                "state."
            )
        return
    if not isinstance(ranking_state, Mapping):
        raise ValueError(
            "An enabled ranking run requires its saved head and reference RNG state."
        )
    _require_keys(ranking_state, STATE_RANKING_KEYS, "dense_training.ranking")
    contributed = ranking_state["contributed_steps"]
    if isinstance(contributed, bool) or not isinstance(contributed, int):
        raise ValueError(
            "dense_training.ranking.contributed_steps must be an integer, got "
            f"{contributed!r}."
        )
    if not 0 <= int(contributed) <= int(completed):
        raise ValueError(
            "dense_training.ranking.contributed_steps="
            f"{contributed} is outside [0, {completed}]; a head cannot contribute "
            "on more steps than the run completed."
        )
    _as_rng_state(
        ranking_state["reference_rng"], "dense_training.ranking.reference_rng"
    )
    student_head = ranking_state["student_head"]
    teacher_head = ranking_state["teacher_head"]
    _validate_head_state(student_head, "student_head")
    _validate_head_state(teacher_head, "teacher_head")
    if set(student_head) != set(teacher_head):
        raise ValueError(
            "The saved ranking student and teacher heads have different "
            "parameter sets."
        )
    for key in student_head:
        if tuple(student_head[key].shape) != tuple(teacher_head[key].shape):
            raise ValueError(
                f"The saved ranking heads disagree on {key!r} shape; the EMA pair "
                "must be structurally identical."
            )


def _validate_head_state(head: Any, label: str) -> None:
    """Validate one saved three-Linear head state without importing ranking."""
    if not isinstance(head, Mapping):
        raise ValueError(f"dense_training.ranking.{label} must be a state dict.")
    if set(head) != _HEAD_KEYS:
        raise ValueError(
            f"dense_training.ranking.{label} must carry exactly "
            f"{sorted(_HEAD_KEYS)}; got {sorted(head)}."
        )
    for key, value in head.items():
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise ValueError(
                f"dense_training.ranking.{label}.{key} must be a floating tensor."
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(
                f"dense_training.ranking.{label}.{key} contains non-finite values."
            )
        # Finiteness at the recorded precision is not enough: the frozen recipe
        # trains in float32, so a float64 value just above the float32 range
        # becomes Inf on load. Reject it here, in preflight, not mid-training.
        try:
            cast = value.to(torch.float32)
        except (RuntimeError, TypeError) as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"dense_training.ranking.{label}.{key} cannot be loaded as the "
                f"training dtype: {exc}"
            ) from exc
        if not bool(torch.isfinite(cast).all()):
            raise ValueError(
                f"dense_training.ranking.{label}.{key} is not finite at the "
                "training float32 dtype; a value that is finite at its recorded "
                "precision can still overflow on load."
            )
    for layer in ("0", "2", "4"):
        weight = head[f"{layer}.weight"]
        bias = head[f"{layer}.bias"]
        if weight.ndim != 2:
            raise ValueError(
                f"dense_training.ranking.{label}.{layer}.weight must be 2-D."
            )
        if tuple(weight.shape[0:1]) != tuple(bias.shape[0:1]) or bias.ndim != 1:
            raise ValueError(
                f"dense_training.ranking.{label}.{layer} weight/bias shapes are "
                "inconsistent."
            )
    if int(head["0.weight"].shape[0]) != int(head["2.weight"].shape[1]):
        raise ValueError(
            f"dense_training.ranking.{label} layer widths are inconsistent."
        )
    if int(head["2.weight"].shape[0]) != int(head["4.weight"].shape[1]):
        raise ValueError(
            f"dense_training.ranking.{label} layer widths are inconsistent."
        )


def _group_flags_by_parameter(optimizer: Any) -> dict[int, dict[str, bool]]:
    """Map each optimized parameter id to its group's execution flags."""
    flags: dict[int, dict[str, bool]] = {}
    for group in getattr(optimizer, "param_groups", ()):
        resolved = {
            name: bool(group.get(name, False))
            for name in ("capturable", "fused", "foreach", "differentiable", "amsgrad")
        }
        for parameter in group.get("params", ()):
            flags[id(parameter)] = resolved
    return flags


def validate_ranking_head_optimizer_state(
    optimizer: Any, head_parameters, *, contributed_steps: int
) -> None:
    """Prove the restored AdamW state matches this head's contribution count.

    Method-agnostic last line of defence: it runs on the *live* optimizer after
    the saved state is loaded, so it also covers paths that do not rebuild a
    parameter partition during preflight. A head that never contributed must
    carry no state (AdamW would otherwise decay and step it), and a head that
    did contribute must have complete moments whose step equals that count.
    """
    contributed = int(contributed_steps)
    if contributed < 0:
        raise ValueError(
            "ranking contributed_steps must be non-negative, got "
            f"{contributed_steps!r}."
        )
    state = getattr(optimizer, "state", {})
    # The group owning each parameter decides which AdamW branch will run, and
    # that branch decides which tensors the next update reads.
    flags_by_parameter = _group_flags_by_parameter(optimizer)
    for index, parameter in enumerate(head_parameters):
        entry = state.get(parameter)
        flags = flags_by_parameter.get(id(parameter), {})
        if contributed == 0:
            if entry:
                raise ValueError(
                    "The ranking head has optimizer state "
                    f"(step {entry.get('step')!r}) but its recorded contribution "
                    "count is 0; a skipped objective must not be steered by "
                    "fabricated moments."
                )
            continue
        if not entry:
            raise ValueError(
                "The ranking head contributed "
                f"{contributed} step(s) but optimizer parameter {index} has no "
                "restored state; a partial head state is not resumable."
            )
        device_type = parameter.device.type
        # Measured on torch 2.12.1: the non-fused capturable branch asserts CUDA,
        # while the fused branch steps fine on CPU. They must not share a rule —
        # treating them alike rejected a legitimate CPU resume.
        if flags.get("capturable") and not flags.get("fused") and device_type != "cuda":
            raise ValueError(
                f"The ranking head parameter {index} belongs to a capturable "
                "AdamW group, which only runs on CUDA; this resume targets "
                f"{device_type}."
            )
        step = entry.get("step")
        if isinstance(step, bool):
            raise ValueError(
                f"The ranking head parameter {index} records a boolean AdamW step."
            )
        if torch.is_tensor(step):
            if step.numel() != 1:
                raise ValueError(
                    f"The ranking head parameter {index} records "
                    f"{step.numel()} step values; an optimizer step is one counter."
                )
            if flags.get("foreach") and not flags.get("fused"):
                # The foreach branch runs a dtype check on the step; an integer
                # tensor survives preflight and then fails the next update.
                allowed = (torch.float32, torch.float64)
                if step.dtype not in allowed and not (
                    parameter.device.type == "cpu" and step.dtype == torch.bfloat16
                ):
                    raise ValueError(
                        f"The ranking head parameter {index} records a {step.dtype} "
                        "step, which the foreach AdamW path rejects."
                    )
            step = step.reshape(()).item()
        if isinstance(step, bool) or not isinstance(step, (int, float)):
            raise ValueError(
                f"The ranking head parameter {index} records a non-numeric AdamW "
                f"step {step!r}."
            )
        if not torch.isfinite(torch.tensor(float(step))) or float(step) != int(step):
            raise ValueError(
                f"The ranking head parameter {index} records a fractional or "
                f"non-finite AdamW step {step!r}."
            )
        measured = int(step)
        if measured != contributed:
            raise ValueError(
                f"The ranking head parameter {index} has AdamW step {measured} but "
                f"the recorded contribution count is {contributed}; the head must "
                "be updated exactly on its contributing steps."
            )
        required = ["exp_avg", "exp_avg_sq"]
        if flags.get("amsgrad"):
            # AMSGrad reads max_exp_avg_sq on the next update; a missing tensor is
            # a mid-training KeyError, not a resume rejection.
            required.append("max_exp_avg_sq")
        for name in required:
            value = entry.get(name)
            if not torch.is_tensor(value):
                raise ValueError(
                    f"The ranking head parameter {index} is missing its {name}."
                )
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"The ranking head parameter {index} has a {name} with the "
                    "wrong shape."
                )
            # Finiteness must hold at the *training* dtype: a float64 moment that
            # only overflows float32 would otherwise pass here and fail after
            # loading, when logs already exist.
            if not value.is_floating_point():
                raise ValueError(
                    f"The ranking head parameter {index} has a non-floating "
                    f"{name}."
                )
            try:
                cast = value.to(dtype=parameter.dtype)
            except (RuntimeError, TypeError) as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"The ranking head parameter {index} {name} cannot be loaded "
                    f"as {parameter.dtype}: {exc}"
                ) from exc
            if not bool(torch.isfinite(cast).all()):
                raise ValueError(
                    f"The ranking head parameter {index} has a non-finite {name} "
                    f"at {parameter.dtype}."
                )
            if name == "exp_avg_sq" and bool((cast < 0).any()):
                raise ValueError(
                    f"The ranking head parameter {index} has a negative exp_avg_sq; "
                    "a second moment is never negative."
                )


def expected_head_shapes(hidden_dim: int, *, hidden: int, out_dim: int):
    """The frozen three-Linear head's ``{key: shape}``, including biases."""
    return {
        "0.weight": (int(hidden), int(hidden_dim)),
        "0.bias": (int(hidden),),
        "2.weight": (int(hidden), int(hidden)),
        "2.bias": (int(hidden),),
        "4.weight": (int(out_dim), int(hidden)),
        "4.bias": (int(out_dim),),
    }
