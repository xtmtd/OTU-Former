"""Formula, parity and diagnostic tests for the FIFO relational objectives."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from otuformer.training.objectives import (
    _compute_global_loss,
    _compute_local_loss,
    _make_relational_step_targets,
    _ssl_loss,
)


def _normalized(values, dim=-1):
    return F.normalize(torch.tensor(values, dtype=torch.float32), dim=dim)


def _fixture(batch: int = 2, history: int = 1, out_dim: int = 2):
    center = torch.tensor([[0.1, -0.2]], dtype=torch.float32)
    current = [
        _normalized([[0.6, 0.8], [0.8, 0.6]])[:batch] - center,
        _normalized([[0.5, 0.9], [0.9, 0.5]])[:batch] - center,
    ]
    current = [F.normalize(slot, dim=-1) for slot in current]
    torch.manual_seed(3)
    history_features = torch.randn(history, 2, out_dim)
    return center, current, history_features


def test_fifo_rectangular_loss_keeps_student_gradient():
    s = torch.tensor([[0.6, 0.8], [0.8, 0.6]], requires_grad=True)
    t = torch.eye(2, requires_grad=True)
    c = torch.cat([t.detach(), torch.tensor([[0.8, 0.6]])])
    teacher_logits = t.detach() @ c.T / 0.1
    p = (teacher_logits - teacher_logits.mean(1, keepdim=True)).softmax(1)
    actual = _ssl_loss(s, t, 0.2, 0.1, candidates=c, teacher_probs=p)
    expected = -(p * (s @ c.T / 0.2).log_softmax(1)).sum(1).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert s.grad.abs().sum() > 0 and t.grad is None


def test_fifo_candidates_are_current_first_and_slot_local():
    center, current, history = _fixture()
    targets = _make_relational_step_targets(
        current, history, center, 0.1, collect_diagnostics=False
    )
    batch = int(current[0].shape[0])
    for slot in (0, 1):
        assert targets.candidates[slot].shape == (batch + 1, 2)
        assert targets.teacher_probs[slot].shape == (batch, batch + 1)
        # Current batch stays first and unchanged.
        torch.testing.assert_close(
            targets.candidates[slot][:batch], current[slot].detach()
        )
        # The history column belongs to that slot only and uses the pre-step center.
        expected = F.normalize(history[:, slot] - center, dim=-1)
        torch.testing.assert_close(targets.candidates[slot][batch:], expected)
        assert not targets.candidates[slot].requires_grad
        assert not targets.teacher_probs[slot].requires_grad


def test_fifo_rectangular_global_and_local_means_match_hand_formula():
    center, current, history = _fixture()
    targets = _make_relational_step_targets(
        current, history, center, 0.1, collect_diagnostics=False
    )
    student = torch.nn.Parameter(_normalized([[0.2, 0.9], [0.7, 0.4]]))
    second = torch.nn.Parameter(_normalized([[0.9, 0.3], [0.4, 0.8]]))

    def hand(slot, query):
        candidates = targets.candidates[slot]
        probs = targets.teacher_probs[slot]
        return -(probs * (query @ candidates.T / 0.2).log_softmax(1)).sum(1).mean()

    global_actual = _compute_global_loss(
        [student, second],
        current,
        0.2,
        0.1,
        relational_pairing="full",
        relational_targets=targets,
    )
    global_expected = (
        hand(0, student) + hand(1, student) + hand(0, second) + hand(1, second)
    ) / 4
    torch.testing.assert_close(global_actual, global_expected)

    collected: list[dict[str, torch.Tensor]] = []
    local_actual = _compute_local_loss(
        [student, second],
        current,
        0.2,
        0.1,
        relational_targets=targets,
        student_stats=collected,
    )
    local_expected = global_expected
    torch.testing.assert_close(local_actual, local_expected)
    assert len(collected) == 1
    assert set(collected[0]) == {"self_mass", "other_current_mass", "history_mass"}
    assert all(not value.requires_grad for value in collected[0].values())

    # History columns are strict additions: B + E, never B + 2E.
    assert targets.candidates[0].shape[0] == 3
    assert targets.candidates[0].shape[0] != 4


def test_fifo_empty_history_matches_loss_gradient_and_rng():
    center, current, _ = _fixture(history=0)
    empty = torch.zeros(0, 2, 2)
    targets = _make_relational_step_targets(
        current, empty, center, 0.1, collect_diagnostics=False
    )
    assert targets.candidates[0].shape == (2, 2)
    assert targets.teacher_stats is None

    base = _normalized([[0.2, 0.9], [0.7, 0.4]])
    old_student = base.clone().requires_grad_(True)
    new_student = base.clone().requires_grad_(True)
    old = _compute_global_loss(
        [old_student, old_student], current, 0.2, 0.1, relational_pairing="cross"
    )
    new = _compute_global_loss(
        [new_student, new_student],
        current,
        0.2,
        0.1,
        relational_pairing="cross",
        relational_targets=targets,
    )
    torch.testing.assert_close(old, new)
    old.backward()
    new.backward()
    torch.testing.assert_close(old_student.grad, new_student.grad)

    before = torch.get_rng_state().clone()
    _make_relational_step_targets(
        current, empty, center, 0.1, collect_diagnostics=True
    )
    assert torch.equal(before, torch.get_rng_state())


def test_fifo_current_history_share_pre_step_center():
    center, current, history = _fixture()
    later_center = center + 5.0
    targets = _make_relational_step_targets(
        current, history, center, 0.1, collect_diagnostics=False
    )
    targets_later = _make_relational_step_targets(
        current, history, later_center, 0.1, collect_diagnostics=False
    )
    assert not torch.allclose(
        targets.candidates[0], targets_later.candidates[0]
    )
    # The current columns never consume the center here: they arrive centered.
    torch.testing.assert_close(targets.candidates[0][:2], current[0].detach())

    # History is recentered exactly once here and does not update the center.
    frozen = center.clone()
    _make_relational_step_targets(
        current, history, center, 0.1, collect_diagnostics=True
    )
    assert torch.equal(frozen, center)


def test_fifo_probability_partitions_and_nonself_perplexity():
    torch.manual_seed(11)
    center = torch.zeros(1, 2)
    current = [
        F.normalize(torch.randn(3, 2), dim=-1),
        F.normalize(torch.randn(3, 2), dim=-1),
    ]
    history = torch.randn(2, 2, 2)
    targets = _make_relational_step_targets(
        current, history, center, 0.1, collect_diagnostics=True
    )
    stats = targets.teacher_stats
    assert stats is not None
    total = stats["self_mass"] + stats["other_current_mass"] + stats["history_mass"]
    torch.testing.assert_close(total, torch.tensor(1.0))

    row_perplexities = []
    for probs in targets.teacher_probs:
        for row in range(int(probs.shape[0])):
            row_probs = probs[row]
            nonself = torch.ones_like(row_probs, dtype=torch.bool)
            nonself[row] = False
            conditional = row_probs[nonself] / row_probs[nonself].sum()
            row_perplexities.append(
                torch.exp(-(conditional * conditional.log()).sum())
            )
    torch.testing.assert_close(
        stats["nonself_effective"], torch.stack(row_perplexities).mean()
    )

    # Empty history has exactly zero history mass, not an invented age/zero mix.
    empty = _make_relational_step_targets(
        current, torch.zeros(0, 2, 2), center, 0.1, collect_diagnostics=True
    )
    assert float(empty.teacher_stats["history_mass"]) == 0.0

    # One current image and no history has no nonself candidate at all.
    single = _make_relational_step_targets(
        [current[0][:1], current[1][:1]],
        torch.zeros(0, 2, 2),
        center,
        0.1,
        collect_diagnostics=True,
    )
    assert "nonself_effective" not in single.teacher_stats


def test_fifo_nonself_perplexity_survives_a_low_temperature():
    """The conditional distribution comes from the nonself logits themselves."""
    from otuformer.training.objectives import _teacher_row_statistics

    for temperature in (0.1, 0.025, 0.01):
        # Column 0 is the self candidate and it must *dominate*: with a dominant
        # self logit the self mass rounds to one at a low temperature, which is
        # exactly the regime where renormalizing the full softmax breaks (it
        # returned ~55406 or nothing at all there). The two equal nonself
        # candidates keep the exact answer at 2.
        logits = torch.tensor([[1.0, -1.0, -1.0]]) / temperature
        probs = F.softmax(logits - logits.mean(dim=1, keepdim=True), dim=1)
        stats = _teacher_row_statistics(logits, probs, 1)
        assert float(stats["self_mass"]) == pytest.approx(1.0)
        assert float(stats["nonself_effective"]) == pytest.approx(2.0)
        total = (
            float(stats["self_mass"])
            + float(stats["other_current_mass"])
            + float(stats["history_mass"])
        )
        assert total == pytest.approx(1.0)


def test_fifo_diagnostics_are_detached_and_log_step_only():
    center, current, history = _fixture()
    quiet = _make_relational_step_targets(
        current, history, center, 0.1, collect_diagnostics=False
    )
    assert quiet.teacher_stats is None

    loud = _make_relational_step_targets(
        current, history, center, 0.1, collect_diagnostics=True
    )
    stats = loud.teacher_stats
    assert stats is not None
    assert set(stats) == {
        "self_mass",
        "other_current_mass",
        "history_mass",
        "entropy",
        "nonself_effective",
    }
    for value in stats.values():
        assert isinstance(value, torch.Tensor)
        assert not value.requires_grad
        assert bool(torch.isfinite(value).all())
        assert value.dim() == 0

    # Diagnostics never change the loss or its gradient.
    base = _normalized([[0.2, 0.9], [0.7, 0.4]])
    quiet_student = base.clone().requires_grad_(True)
    loud_student = base.clone().requires_grad_(True)
    quiet_loss = _compute_global_loss(
        [quiet_student, quiet_student],
        current,
        0.2,
        0.1,
        relational_pairing="cross",
        relational_targets=quiet,
    )
    loud_loss = _compute_global_loss(
        [loud_student, loud_student],
        current,
        0.2,
        0.1,
        relational_pairing="cross",
        relational_targets=loud,
    )
    torch.testing.assert_close(quiet_loss, loud_loss)
    quiet_loss.backward()
    loud_loss.backward()
    torch.testing.assert_close(quiet_student.grad, loud_student.grad)


def test_fifo_loss_rejects_one_sided_rectangular_inputs():
    student = _normalized([[0.2, 0.9]])
    teacher = _normalized([[0.2, 0.9]])
    with pytest.raises(ValueError, match="supplied together"):
        _ssl_loss(student, teacher, 0.2, 0.1, candidates=teacher)
    with pytest.raises(ValueError, match="supplied together"):
        _ssl_loss(student, teacher, 0.2, 0.1, teacher_probs=teacher)
