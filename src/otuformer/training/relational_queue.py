"""Ordered paired-teacher FIFO for the relational candidate distributions.

This is one narrow method-specific state owner: it stores the ordered image
records (two detached pre-center float32 teacher vectors plus the successful
optimization step that observed them), excludes the current batch from the
eligible history, refreshes/evicts after a successful step, and serializes a
strict version-1 logical state. It imports no CLI, dataset or trainer module and
never recenters, forwards a model or keeps a global instance.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch

QUEUE_STATE_VERSION = 1

#: The complete version-1 persisted field set; unexpected keys are rejected
#: rather than guessed at.
QUEUE_STATE_KEYS = (
    "version",
    "keys",
    "features",
    "observed_steps",
    "completed_steps",
    "data_fingerprint",
    "data_image_count",
    "data_size_bytes",
)

QUEUE_CONFIG_KEY = "relational_queue_config"

QUEUE_CONFIG_KEYS = ("version", "capacity")


def _require_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got {value!r}.")
    return value


@dataclass(frozen=True)
class RelationalQueueSnapshot:
    """One immutable view of the eligible history for a single step.

    ``features`` is ``[E, 2, D]`` in oldest-to-newest order; a zero-history
    snapshot keeps the ``[0, 2, D]`` shape instead of random padding.
    """

    keys: tuple[str, ...]
    features: torch.Tensor
    observed_steps: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.keys)

    def ages_at(self, step: int) -> tuple[int, ...]:
        """Successful-step ages: the previous step's observation has age one."""
        return tuple(step - observed for observed in self.observed_steps)


class RelationalQueue:
    """Oldest-to-newest image-keyed FIFO of paired teacher observations."""

    def __init__(self, capacity: int, out_dim: int, *, device: torch.device) -> None:
        self.capacity = _require_int(capacity, "queue capacity", minimum=1)
        self.out_dim = _require_int(out_dim, "queue out_dim", minimum=1)
        self.device = torch.device(device)
        self._records: "OrderedDict[str, tuple[torch.Tensor, int]]" = OrderedDict()
        self.completed_steps = 0

    def __len__(self) -> int:
        return len(self._records)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._records)

    @property
    def observed_steps(self) -> tuple[int, ...]:
        return tuple(step for _pair, step in self._records.values())

    def snapshot(
        self, batch_keys: Sequence[str], *, step: int
    ) -> RelationalQueueSnapshot:
        """Eligible history for ``step``: every current key excluded, no refresh."""
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError(f"step must be an integer, got {step!r}.")
        if step != self.completed_steps:
            raise ValueError(
                "The queue snapshot must describe the current completed step: "
                f"queue has {self.completed_steps}, got step={step}."
            )
        current = tuple(self._validated_keys(batch_keys, "batch_keys"))
        excluded = set(current)
        keys: list[str] = []
        pairs: list[torch.Tensor] = []
        steps: list[int] = []
        for key, (pair, observed) in self._records.items():
            if key in excluded:
                continue
            keys.append(key)
            pairs.append(pair)
            steps.append(observed)
        if pairs:
            features = torch.stack([pair.detach().clone() for pair in pairs], dim=0)
        else:
            features = torch.zeros(
                (0, 2, self.out_dim), dtype=torch.float32, device=self.device
            )
        return RelationalQueueSnapshot(
            keys=tuple(keys),
            features=features.to(device=self.device, dtype=torch.float32),
            observed_steps=tuple(steps),
        )

    def commit(
        self,
        batch_keys: Sequence[str],
        precenter_features: torch.Tensor,
        *,
        step: int,
    ) -> dict[str, int]:
        """Refresh the current images and insert their observations.

        All inputs are validated before any mutation: a rejected commit leaves
        the order, ages and completed count untouched.
        """
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError(f"step must be an integer, got {step!r}.")
        if step != self.completed_steps:
            raise ValueError(
                "The queue commit must describe the current completed step: "
                f"queue has {self.completed_steps}, got step={step}."
            )
        keys = self._validated_keys(batch_keys, "batch_keys")
        if not isinstance(precenter_features, torch.Tensor):
            raise ValueError(
                "precenter_features must be a tensor, got "
                f"{type(precenter_features).__name__}."
            )
        expected = (len(keys), 2, self.out_dim)
        if tuple(precenter_features.shape) != expected:
            raise ValueError(
                f"precenter_features must have shape {expected}, got "
                f"{tuple(precenter_features.shape)}."
            )
        if not precenter_features.is_floating_point():
            raise ValueError(
                "precenter_features must be a floating-point tensor, got "
                f"{precenter_features.dtype}."
            )
        detached = precenter_features.detach()
        if not bool(torch.isfinite(detached).all()):
            raise ValueError(
                "precenter_features contains non-finite values; refusing to "
                "insert them into the relational queue."
            )
        pairs = [
            detached[index].to(device=self.device, dtype=torch.float32).clone()
            for index in range(len(keys))
        ]

        refreshed = sum(1 for key in keys if key in self._records)
        for key, pair in zip(keys, pairs):
            # Re-inserting an existing image refreshes it at the newest end.
            self._records.pop(key, None)
            self._records[key] = (pair, step)
        evicted = 0
        while len(self._records) > self.capacity:
            self._records.popitem(last=False)
            evicted += 1
        self.completed_steps += 1
        return {"refreshed": refreshed, "evicted": evicted}

    def state_dict(
        self,
        *,
        data_fingerprint: str,
        data_image_count: int,
        data_size_bytes: int,
    ) -> dict[str, Any]:
        """The complete version-1 logical state, features detached on CPU."""
        if self._records:
            features = torch.stack(
                [pair.detach().to("cpu", torch.float32) for pair, _ in self._records.values()],
                dim=0,
            )
        else:
            features = torch.zeros(
                (0, 2, self.out_dim), dtype=torch.float32, device="cpu"
            )
        return {
            "version": QUEUE_STATE_VERSION,
            "keys": list(self._records),
            "features": features,
            "observed_steps": [step for _pair, step in self._records.values()],
            "completed_steps": int(self.completed_steps),
            "data_fingerprint": str(data_fingerprint),
            "data_image_count": int(data_image_count),
            "data_size_bytes": int(data_size_bytes),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        capacity: int,
        out_dim: int,
        completed_steps: int,
        allowed_keys: frozenset[str],
        data_fingerprint: str,
        data_image_count: int,
        data_size_bytes: int,
        device: torch.device,
    ) -> "RelationalQueue":
        """Validate and restore one version-1 state; never repairs silently."""
        if not isinstance(state, Mapping):
            raise ValueError(
                f"relational_queue state must be a mapping, got "
                f"{type(state).__name__}."
            )
        missing = [key for key in QUEUE_STATE_KEYS if key not in state]
        if missing:
            raise ValueError(
                "relational_queue state is incomplete; missing: "
                + ", ".join(missing)
            )
        extra = sorted(set(state) - set(QUEUE_STATE_KEYS))
        if extra:
            raise ValueError(
                "relational_queue state has unknown keys: " + ", ".join(extra)
            )
        version = state["version"]
        if isinstance(version, bool) or not isinstance(version, int) or version != 1:
            raise ValueError(
                f"relational_queue.version must be the integer 1, got {version!r}."
            )
        resolved_capacity = _require_int(capacity, "queue capacity", minimum=1)
        resolved_out_dim = _require_int(out_dim, "queue out_dim", minimum=1)
        expected_steps = _require_int(
            completed_steps, "completed_steps", minimum=0
        )
        saved_steps = _require_int(
            state["completed_steps"], "relational_queue.completed_steps", minimum=0
        )
        if saved_steps != expected_steps:
            raise ValueError(
                "relational_queue.completed_steps must equal the saved iteration: "
                f"state has {saved_steps}, checkpoint iteration is {expected_steps}."
            )

        keys = state["keys"]
        if isinstance(keys, (str, bytes)) or not isinstance(keys, (list, tuple)):
            raise ValueError(
                "relational_queue.keys must be a sequence of image keys, got "
                f"{type(keys).__name__}."
            )
        resolved_keys = tuple(keys)
        if any(not isinstance(key, str) or not key for key in resolved_keys):
            raise ValueError(
                "relational_queue.keys must contain non-empty image-key strings."
            )
        if len(set(resolved_keys)) != len(resolved_keys):
            raise ValueError(
                "relational_queue.keys contains duplicate image keys; the stored "
                "sequence must hold one record per identity."
            )
        if len(resolved_keys) > resolved_capacity:
            raise ValueError(
                f"relational_queue holds {len(resolved_keys)} records, beyond its "
                f"capacity {resolved_capacity}."
            )
        outside = sorted(key for key in resolved_keys if key not in allowed_keys)
        if outside:
            raise ValueError(
                "relational_queue stores image keys that are not part of the "
                "prepared selection: " + ", ".join(outside[:3])
            )

        features = state["features"]
        if not isinstance(features, torch.Tensor):
            raise ValueError(
                "relational_queue.features must be a tensor, got "
                f"{type(features).__name__}."
            )
        if features.requires_grad:
            raise ValueError(
                "relational_queue.features must be detached; a persisted tensor "
                "cannot require gradients."
            )
        expected_shape = (len(resolved_keys), 2, resolved_out_dim)
        if tuple(features.shape) != expected_shape:
            raise ValueError(
                f"relational_queue.features must have shape {expected_shape}, got "
                f"{tuple(features.shape)}."
            )
        if features.dtype != torch.float32:
            raise ValueError(
                "relational_queue.features must be float32, got "
                f"{features.dtype}."
            )
        if len(resolved_keys) and not bool(torch.isfinite(features).all()):
            raise ValueError(
                "relational_queue.features contains non-finite values."
            )

        observed = state["observed_steps"]
        if isinstance(observed, (str, bytes)) or not isinstance(
            observed, (list, tuple)
        ):
            raise ValueError(
                "relational_queue.observed_steps must be a sequence of integers."
            )
        resolved_observed = tuple(
            _require_int(value, "relational_queue.observed_steps entry", minimum=0)
            for value in observed
        )
        if len(resolved_observed) != len(resolved_keys):
            raise ValueError(
                "relational_queue.observed_steps must have one entry per stored "
                f"record ({len(resolved_keys)}), got {len(resolved_observed)}."
            )
        for step in resolved_observed:
            if step >= saved_steps:
                raise ValueError(
                    "relational_queue.observed_steps entries must be smaller than "
                    f"completed_steps {saved_steps}, got {step}."
                )
        ordered = list(resolved_observed)
        if ordered != sorted(ordered):
            raise ValueError(
                "relational_queue.observed_steps must be nondecreasing in FIFO "
                "order."
            )
        if not resolved_keys and saved_steps > 0:
            raise ValueError(
                "relational_queue is empty after "
                f"{saved_steps} completed step(s); a nonempty queue-enabled "
                "dataset leaves history behind. Refusing to continue without it."
            )

        if state["data_fingerprint"] != data_fingerprint:
            raise ValueError(
                "relational_queue.data_fingerprint does not match this "
                "invocation's prepared selection; renaming, reordering, adding, "
                "removing or changing image bytes invalidates an enabled resume."
            )
        if _require_int(
            state["data_image_count"],
            "relational_queue.data_image_count",
            minimum=1,
        ) != _require_int(data_image_count, "data_image_count", minimum=1):
            raise ValueError(
                "relational_queue.data_image_count does not match this "
                "invocation's prepared selection."
            )
        if _require_int(
            state["data_size_bytes"],
            "relational_queue.data_size_bytes",
            minimum=0,
        ) != _require_int(data_size_bytes, "data_size_bytes", minimum=0):
            raise ValueError(
                "relational_queue.data_size_bytes does not match this "
                "invocation's prepared selection."
            )

        queue = cls(resolved_capacity, resolved_out_dim, device=device)
        moved = features.detach().to(device=queue.device, dtype=torch.float32)
        for index, key in enumerate(resolved_keys):
            queue._records[key] = (moved[index].clone(), resolved_observed[index])
        queue.completed_steps = saved_steps
        return queue

    @staticmethod
    def _validated_keys(batch_keys: Iterable[str], label: str) -> tuple[str, ...]:
        if isinstance(batch_keys, (str, bytes)):
            raise ValueError(f"{label} must be a sequence of image keys.")
        keys = tuple(batch_keys)
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError(f"{label} must contain non-empty image-key strings.")
        if len(set(keys)) != len(keys):
            raise ValueError(
                f"{label} contains duplicate image keys; one batch may hold each "
                "image identity once."
            )
        return keys


def queue_config_bundle(capacity: int) -> dict[str, Any]:
    """The closed version-1 enabled configuration bundle."""
    return {
        "version": QUEUE_STATE_VERSION,
        "capacity": _require_int(capacity, "queue capacity", minimum=1),
    }


def validate_queue_config_bundle(payload: object) -> int:
    """Return the recorded positive capacity, rejecting any other shape."""
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"config.{QUEUE_CONFIG_KEY} must be a mapping, got "
            f"{type(payload).__name__}."
        )
    missing = [key for key in QUEUE_CONFIG_KEYS if key not in payload]
    if missing:
        raise ValueError(
            f"config.{QUEUE_CONFIG_KEY} is incomplete; missing: "
            + ", ".join(missing)
        )
    extra = sorted(set(payload) - set(QUEUE_CONFIG_KEYS))
    if extra:
        raise ValueError(
            f"config.{QUEUE_CONFIG_KEY} has unknown keys: " + ", ".join(extra)
        )
    version = payload["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError(
            f"config.{QUEUE_CONFIG_KEY}.version must be the integer 1, got "
            f"{version!r}."
        )
    return _require_int(
        payload["capacity"], f"config.{QUEUE_CONFIG_KEY}.capacity", minimum=1
    )


def storage_bytes(occupancy: int, out_dim: int) -> float:
    """Approximate paired-slot storage for ``occupancy`` float32 records."""
    return float(2 * int(occupancy) * int(out_dim) * 4)
