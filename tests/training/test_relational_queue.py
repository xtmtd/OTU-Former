"""Pure state-machine and codec tests for the relational FIFO owner."""

from __future__ import annotations

import copy

import pytest
import torch

from otuformer.training.relational_queue import (
    QUEUE_STATE_KEYS,
    RelationalQueue,
    RelationalQueueSnapshot,
    queue_config_bundle,
    storage_bytes,
    validate_queue_config_bundle,
)

DEVICE = torch.device("cpu")


def _pairs(count: int, out_dim: int = 2, *, base: float = 1.0) -> torch.Tensor:
    values = torch.arange(count * 2 * out_dim, dtype=torch.float32) + base
    return values.reshape(count, 2, out_dim)


def _state(**overrides):
    queue = RelationalQueue(2, 2, device=DEVICE)
    queue.commit(("a", "b", "c"), _pairs(3), step=0)
    state = queue.state_dict(
        data_fingerprint="f" * 64, data_image_count=4, data_size_bytes=99
    )
    state.update(overrides)
    return state


def _restore(state, **overrides):
    arguments = dict(
        capacity=2,
        out_dim=2,
        completed_steps=1,
        allowed_keys=frozenset({"a", "b", "c"}),
        data_fingerprint="f" * 64,
        data_image_count=4,
        data_size_bytes=99,
        device=DEVICE,
    )
    arguments.update(overrides)
    return RelationalQueue.from_state_dict(state, **arguments)


def test_fifo_snapshot_excludes_current_without_refresh():
    q = RelationalQueue(2, 2, device=torch.device("cpu"))
    q.commit(("a", "b", "c"), torch.ones(3, 2, 2), step=0)
    snap = q.snapshot(("b",), step=1)
    assert snap.keys == ("c",) and snap.observed_steps == (0,)
    assert q.snapshot((), step=1).keys == ("b", "c")
    assert q.completed_steps == 1


def test_fifo_empty_partial_full_and_capacity_edges():
    q = RelationalQueue(3, 2, device=DEVICE)
    empty = q.snapshot((), step=0)
    assert isinstance(empty, RelationalQueueSnapshot)
    assert empty.keys == () and empty.observed_steps == ()
    assert empty.features.shape == (0, 2, 2)
    assert q.completed_steps == 0

    # K > B: a single batch fills only the occupancy it observed.
    q.commit(("a",), _pairs(1), step=0)
    assert q.keys == ("a",) and q.completed_steps == 1
    q.commit(("b",), _pairs(1, base=2.0), step=1)
    partial = q.snapshot((), step=2)
    assert partial.keys == ("a", "b")
    assert partial.features.shape == (2, 2, 2)
    assert partial.ages_at(2) == (2, 1)

    q.commit(("c",), _pairs(1, base=3.0), step=2)
    full = q.snapshot((), step=3)
    assert full.keys == ("a", "b", "c")
    assert len(q) == 3 == q.capacity

    # K < B: only the newest K current records survive, in batch order.
    q.commit(("d", "e", "f", "g"), _pairs(4, base=4.0), step=3)
    assert q.keys == ("e", "f", "g")
    assert q.completed_steps == 4
    assert q.snapshot((), step=4).observed_steps == (3, 3, 3)


def test_fifo_capacity_narrower_than_batch_keeps_batch_order():
    q = RelationalQueue(2, 2, device=DEVICE)
    result = q.commit(("a", "b", "c", "d"), _pairs(4), step=0)
    assert result == {"refreshed": 0, "evicted": 2}
    assert q.keys == ("c", "d")
    assert q.observed_steps == (0, 0)
    assert q.completed_steps == 1
    snapshot = q.snapshot(("c",), step=1)
    assert snapshot.keys == ("d",)
    assert snapshot.ages_at(1) == (1,)


def test_fifo_repeated_key_refresh_counts_and_reorders():
    q = RelationalQueue(3, 2, device=DEVICE)
    q.commit(("a", "b"), _pairs(2), step=0)
    q.commit(("a", "c"), _pairs(2, base=10.0), step=1)
    assert q.keys == ("b", "a", "c")
    assert q.observed_steps == (0, 1, 1)
    assert q.snapshot(("a",), step=2).keys == ("b", "c")
    # A batch containing every stored key leaves no eligible history.
    assert q.snapshot(("a", "b", "c"), step=2).keys == ()
    result = q.commit(("a", "b", "c"), _pairs(3, base=20.0), step=2)
    assert result == {"refreshed": 3, "evicted": 0}
    assert q.keys == ("a", "b", "c")
    assert q.observed_steps == (2, 2, 2)


def test_fifo_reads_do_not_refresh_or_reorder():
    q = RelationalQueue(3, 2, device=DEVICE)
    q.commit(("a", "b"), _pairs(2), step=0)
    for step in (1, 1, 1):
        q.snapshot(("b",), step=step)
    assert q.keys == ("a", "b")
    assert q.observed_steps == (0, 0)
    assert q.completed_steps == 1


def test_fifo_stored_pairs_do_not_alias_inputs_or_snapshots():
    q = RelationalQueue(2, 2, device=DEVICE)
    features = _pairs(2)
    q.commit(("a", "b"), features, step=0)
    features.add_(100.0)
    snapshot = q.snapshot((), step=1)
    assert torch.all(snapshot.features < 100.0)
    snapshot.features.add_(1000.0)
    again = q.snapshot((), step=1)
    assert torch.all(again.features < 100.0)
    assert torch.equal(again.features, q.snapshot((), step=1).features)


def test_fifo_commit_rejects_bad_input_before_mutation():
    q = RelationalQueue(2, 2, device=DEVICE)
    q.commit(("a",), _pairs(1), step=0)
    before_keys = q.keys
    before_steps = q.completed_steps

    with pytest.raises(ValueError, match="duplicate image keys"):
        q.commit(("b", "b"), _pairs(2), step=1)
    with pytest.raises(ValueError, match="shape"):
        q.commit(("b",), _pairs(1, out_dim=3), step=1)
    with pytest.raises(ValueError, match="non-finite"):
        q.commit(("b",), torch.full((1, 2, 2), float("nan")), step=1)
    with pytest.raises(ValueError, match="current completed step"):
        q.commit(("b",), _pairs(1), step=5)
    with pytest.raises(ValueError, match="must be a tensor"):
        q.commit(("b",), [[1.0, 2.0]], step=1)

    assert q.keys == before_keys
    assert q.completed_steps == before_steps


def test_fifo_snapshot_requires_current_step():
    q = RelationalQueue(2, 2, device=DEVICE)
    with pytest.raises(ValueError, match="current completed step"):
        q.snapshot((), step=1)
    assert q.completed_steps == 0


def test_fifo_storage_grows_with_occupancy_only():
    q = RelationalQueue(128, 4, device=DEVICE)
    assert len(q) == 0
    q.commit(("a",), _pairs(1, out_dim=4), step=0)
    assert len(q) == 1
    assert storage_bytes(1, 4) == 32.0


def test_fifo_config_bundle_is_closed():
    assert queue_config_bundle(128) == {"version": 1, "capacity": 128}
    assert validate_queue_config_bundle({"version": 1, "capacity": 128}) == 128
    for payload, match in (
        ({"version": 1}, "incomplete"),
        ({"version": 1, "capacity": 1, "extra": 0}, "unknown keys"),
        ({"version": 2, "capacity": 1}, r"version must be the integer 1"),
        ({"version": True, "capacity": 1}, r"version must be the integer 1"),
        ({"version": 1, "capacity": 0}, ">= 1"),
    ):
        with pytest.raises(ValueError, match=match):
            validate_queue_config_bundle(payload)


def test_fifo_state_dict_is_the_complete_version_one_field_set():
    state = _state()
    assert tuple(state) == QUEUE_STATE_KEYS
    assert state["version"] == 1
    assert state["keys"] == ["b", "c"]
    assert state["observed_steps"] == [0, 0]
    assert state["completed_steps"] == 1
    assert state["features"].shape == (2, 2, 2)
    assert state["features"].device.type == "cpu"
    assert state["features"].dtype == torch.float32
    assert not state["features"].requires_grad


def test_fifo_codec_round_trip_preserves_order_and_next_eviction():
    state = _state()
    restored = _restore(state)
    assert restored.keys == ("b", "c")
    assert restored.observed_steps == (0, 0)
    assert restored.completed_steps == 1
    assert torch.equal(restored.snapshot((), step=1).features, state["features"].clone())
    restored.commit(("d",), _pairs(1), step=1)
    assert restored.keys == ("c", "d")

    empty = RelationalQueue(2, 2, device=DEVICE).state_dict(
        data_fingerprint="f" * 64, data_image_count=4, data_size_bytes=99
    )
    restored_empty = _restore(empty, completed_steps=0)
    assert len(restored_empty) == 0
    assert restored_empty.snapshot((), step=0).features.shape == (0, 2, 2)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda s: s.update({"unexpected": 1}), "unknown keys"),
        (lambda s: s.pop("keys"), "incomplete"),
        (lambda s: s.update({"version": 2}), "version must be the integer 1"),
        (lambda s: s.update({"version": True}), "version must be the integer 1"),
        (
            lambda s: s.update({"completed_steps": True}),
            "completed_steps must be an integer",
        ),
        (lambda s: s.update({"completed_steps": 0}), "must equal the saved iteration"),
        (lambda s: s.update({"keys": ["b", "b"]}), "duplicate image keys"),
        (lambda s: s.update({"keys": ["b", "zzz"]}), "not part of the prepared"),
        (lambda s: s.update({"keys": [1, 2]}), "non-empty image-key strings"),
        (
            lambda s: s.update({"features": torch.ones(3, 2, 2)}),
            "must have shape",
        ),
        (
            lambda s: s.update({"features": torch.ones(2, 2, 2, dtype=torch.float64)}),
            "must be float32",
        ),
        (
            lambda s: s.update(
                {
                    "features": torch.tensor(
                        [[[float("nan"), 1.0], [1.0, 1.0]]] * 2
                    )
                }
            ),
            "non-finite",
        ),
        (
            lambda s: s.update(
                {"features": torch.ones(2, 2, 2, requires_grad=True)}
            ),
            "must be detached",
        ),
        (lambda s: s.update({"observed_steps": [0]}), "one entry per stored record"),
        (lambda s: s.update({"observed_steps": [1, 1]}), "smaller than"),
        (lambda s: s.update({"observed_steps": [True, 0]}), "must be an integer"),
        (lambda s: s.update({"observed_steps": [0, -1]}), ">= 0"),
        (lambda s: s.update({"data_fingerprint": "0" * 64}), "data_fingerprint"),
        (lambda s: s.update({"data_image_count": 5}), "data_image_count"),
        (lambda s: s.update({"data_size_bytes": 12}), "data_size_bytes"),
        (lambda s: s.update({"data_image_count": True}), "must be an integer"),
        (
            lambda s: (
                s.update({"keys": [], "features": torch.zeros(0, 2, 2)}),
                s.update({"observed_steps": []}),
            ),
            "empty after",
        ),
    ],
)
def test_fifo_codec_rejects_malformed_state(mutate, match):
    state = _state()
    result = mutate(state)
    if isinstance(result, tuple):
        pass
    with pytest.raises(ValueError, match=match):
        _restore(state)


def test_fifo_codec_rejects_reversed_timestamps_and_bad_capacity():
    state = _state(observed_steps=[2, 1], completed_steps=3)
    with pytest.raises(ValueError, match="nondecreasing"):
        _restore(state, completed_steps=3)
    ok = _state()
    with pytest.raises(ValueError, match="beyond its capacity"):
        _restore(ok, capacity=1)
    with pytest.raises(ValueError, match=">= 1"):
        _restore(ok, capacity=0)
    with pytest.raises(ValueError, match="must be a mapping"):
        RelationalQueue.from_state_dict(
            ["not", "a", "mapping"],
            capacity=2,
            out_dim=2,
            completed_steps=1,
            allowed_keys=frozenset(),
            data_fingerprint="f" * 64,
            data_image_count=4,
            data_size_bytes=99,
            device=DEVICE,
        )


def test_fifo_round_trip_does_not_alias_the_saved_state():
    state = _state()
    snapshot = copy.deepcopy(state)
    restored = _restore(state)
    restored.snapshot((), step=1).features.add_(5.0)
    assert torch.equal(state["features"], snapshot["features"])
