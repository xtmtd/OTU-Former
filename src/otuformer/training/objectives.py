"""Pretraining loss terms and their per-step diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple
import torch
import torch.nn.functional as F
from otuformer.training.loss import (
    ArcFaceLoss,
    LOSS_REGISTRY,
    SubCenterArcFaceLoss,
    SupConLoss,
    dino_distillation_loss,
    ibot_patch_loss,
    koleo_loss,
    masked_patch_cosine_loss,
)
from .patch_objective import (
    PATCH_CENTER_MOMENTUM,
)

_RELATIONAL_PAIR_SETS: dict[str, tuple[tuple[int, int], ...]] = {
    "cross": ((0, 1), (1, 0)),
    "full": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "same": ((0, 0), (1, 1)),
}


@dataclass(frozen=True)
class _RelationalStepTargets:
    """This step's rectangular relational candidates and teacher targets.

    ``candidates`` holds one ``[B + E, D]`` detached matrix per teacher slot in
    current-batch-first order; ``teacher_probs`` holds the matching ``[B, B + E]``
    detached targets. ``teacher_stats`` is already averaged over the contributing
    slots, and unavailable measurements are omitted rather than NaN.
    """

    candidates: tuple[torch.Tensor, torch.Tensor]
    teacher_probs: tuple[torch.Tensor, torch.Tensor]
    teacher_stats: dict[str, torch.Tensor] | None


def _mean_or_none(values: list[torch.Tensor]) -> torch.Tensor | None:
    stacked = torch.stack(values)
    return stacked.mean()


def _teacher_row_statistics(
    logits: torch.Tensor,
    probabilities: torch.Tensor,
    current_count: int,
) -> dict[str, torch.Tensor]:
    """Per-slot teacher target structure, reduced to detached scalars.

    Entropy uses the stable log-softmax of the same logits; the conditional
    nonself perplexity is the mean per-row perplexity over the nonself cells and
    is omitted when a row has no nonself candidate at all.
    """
    rows = int(probabilities.shape[0])
    columns = int(probabilities.shape[1])
    log_probabilities = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        index = torch.arange(rows, device=probabilities.device)
        self_mass = probabilities[index, index]
        current_only = torch.zeros_like(probabilities)
        current_only[:, :current_count] = probabilities[:, :current_count]
        current_only[index, index] = 0.0
        other_current = current_only.sum(dim=1)
        history = probabilities[:, current_count:].sum(dim=1)
        entropy = -(probabilities * log_probabilities).sum(dim=1)

        nonself_mask = torch.ones_like(probabilities, dtype=torch.bool)
        nonself_mask[index, index] = False
        nonself_mass = (probabilities * nonself_mask).sum(dim=1)
        stats: dict[str, torch.Tensor] = {
            "self_mass": self_mass.mean().detach(),
            "other_current_mass": other_current.mean().detach(),
            "history_mass": history.mean().detach(),
            "entropy": entropy.mean().detach(),
        }
        if columns < 2:
            return stats
        usable = nonself_mass > 0.0
        if not bool(usable.any()):
            return stats
        safe_mass = torch.where(
            usable, nonself_mass, torch.ones_like(nonself_mass)
        )
        conditional = torch.where(
            nonself_mask,
            probabilities / safe_mass.unsqueeze(1),
            torch.zeros_like(probabilities),
        )
        conditional_log = torch.where(
            nonself_mask,
            log_probabilities - safe_mass.clamp_min(1e-30).log().unsqueeze(1),
            torch.zeros_like(probabilities),
        )
        row_perplexity = torch.exp(
            -(conditional * conditional_log).sum(dim=1)
        )
        stats["nonself_effective"] = row_perplexity[usable].mean().detach()
    return stats


def _make_relational_step_targets(
    teacher_global: list[torch.Tensor],
    history_features: torch.Tensor,
    center_before: torch.Tensor,
    temp_teacher: float,
    *,
    collect_diagnostics: bool,
) -> _RelationalStepTargets:
    """Build current-first candidates and teacher targets for both slots once.

    ``teacher_global`` is this step's already centered/normalized teacher output
    per slot; ``history_features`` is ``[E, 2, D]`` pre-center float32 history.
    History is recentered with the *pre-step* center once per slot, and the
    current batch stays first, so every slot holds ``B + E`` columns.
    """
    if len(teacher_global) != 2:
        raise ValueError(
            "relational targets need exactly two teacher slot features, got "
            f"{len(teacher_global)}."
        )
    candidates: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    slot_stats: list[dict[str, torch.Tensor]] = []
    for slot in range(2):
        current = teacher_global[slot]
        if history_features.numel():
            history = F.normalize(
                history_features[:, slot].detach().to(current.dtype) - center_before,
                dim=-1,
            )
            slot_candidates = torch.cat([current, history], dim=0)
        else:
            slot_candidates = current
        detached_candidates = slot_candidates.detach()
        with torch.no_grad():
            logits = current.detach() @ detached_candidates.T / temp_teacher
            logits = logits - logits.mean(dim=1, keepdim=True)
            probs = F.softmax(logits, dim=1)
            if collect_diagnostics:
                slot_stats.append(
                    _teacher_row_statistics(
                        logits, probs, int(current.shape[0])
                    )
                )
        candidates.append(detached_candidates)
        probabilities.append(probs.detach())
    teacher_stats: dict[str, torch.Tensor] | None = None
    if collect_diagnostics:
        keys = set(slot_stats[0])
        for stats in slot_stats[1:]:
            keys &= set(stats)
        teacher_stats = {
            key: _mean_or_none([stats[key] for stats in slot_stats])
            for key in sorted(keys)
        }
    return _RelationalStepTargets(
        candidates=(candidates[0], candidates[1]),
        teacher_probs=(probabilities[0], probabilities[1]),
        teacher_stats=teacher_stats,
    )


def _student_row_statistics(
    logits: torch.Tensor, probabilities: torch.Tensor, current_count: int
) -> dict[str, torch.Tensor]:
    """Detached student mass allocation over the candidate columns."""
    rows = int(probabilities.shape[0])
    index = torch.arange(rows, device=probabilities.device)
    with torch.no_grad():
        detached = probabilities.detach()
        self_mass = detached[index, index]
        current_only = torch.zeros_like(detached)
        current_only[:, :current_count] = detached[:, :current_count]
        current_only[index, index] = 0.0
        return {
            "self_mass": self_mass.mean(),
            "other_current_mass": current_only.sum(dim=1).mean(),
            "history_mass": detached[:, current_count:].sum(dim=1).mean(),
        }


def _merge_student_statistics(
    collected: list[dict[str, torch.Tensor]], target: list[dict[str, torch.Tensor]]
) -> None:
    """Average per-pair student statistics and append them to ``target``."""
    if not collected:
        return
    keys = set(collected[0])
    for stats in collected[1:]:
        keys &= set(stats)
    target.append(
        {key: _mean_or_none([stats[key] for stats in collected]) for key in sorted(keys)}
    )


def _relational_pair_loss(
    student_out: torch.Tensor,
    slot: int,
    targets: _RelationalStepTargets,
    temp_student: float,
    collected: list[dict[str, torch.Tensor]] | None = None,
) -> torch.Tensor:
    """One student query against one slot's rectangular candidate matrix."""
    candidates = targets.candidates[slot]
    teacher_probs = targets.teacher_probs[slot]
    student_logits = student_out @ candidates.T / temp_student
    log_probs = F.log_softmax(student_logits, dim=1)
    loss = -(teacher_probs * log_probs).sum(dim=1).mean()
    if collected is not None:
        collected.append(
            _student_row_statistics(
                student_logits,
                F.softmax(student_logits.detach(), dim=1),
                int(student_out.shape[0]),
            )
        )
    return loss


def _compute_global_loss(
    student_global: list[torch.Tensor],
    teacher_global: list[torch.Tensor],
    temp_student: float,
    temp_teacher: float,
    *,
    relational_pairing: str,
    relational_targets: _RelationalStepTargets | None = None,
    student_stats: list[dict[str, torch.Tensor]] | None = None,
) -> torch.Tensor:
    """Average the selected global student/teacher view pairs.

    ``cross`` averages A-to-B and B-to-A, ``full`` averages the whole Cartesian
    product, and the internal ``same`` pairing keeps the historical
    matching-index terms for legacy checkpoints. With ``relational_targets`` the
    pair selects its matching slot's rectangular candidate matrix instead of the
    batch-only teacher-output relation.
    """
    pairs = _RELATIONAL_PAIR_SETS.get(relational_pairing)
    if pairs is None:
        raise ValueError(
            f"Unsupported relational_pairing {relational_pairing!r}; expected one "
            f"of: {', '.join(_RELATIONAL_PAIR_SETS)}."
        )
    if len(student_global) != 2 or len(teacher_global) != 2:
        raise ValueError(
            "global relational loss needs exactly two student and two teacher "
            f"features, got {len(student_global)} and {len(teacher_global)}."
        )
    loss = torch.tensor(0.0, device=student_global[0].device)
    collected: list[dict[str, torch.Tensor]] = []
    for student_index, teacher_index in pairs:
        if relational_targets is not None:
            loss = loss + _relational_pair_loss(
                student_global[student_index],
                teacher_index,
                relational_targets,
                temp_student,
                collected if student_stats is not None else None,
            )
            continue
        loss = loss + _ssl_loss(
            student_global[student_index],
            teacher_global[teacher_index],
            temp_student,
            temp_teacher,
        )
    if student_stats is not None:
        _merge_student_statistics(collected, student_stats)
    return loss / len(pairs)


def _compute_local_loss(
    local_views_student: list[torch.Tensor],
    teacher_global: list[torch.Tensor],
    temp_student: float,
    temp_teacher: float,
    *,
    relational_targets: _RelationalStepTargets | None = None,
    student_stats: list[dict[str, torch.Tensor]] | None = None,
) -> torch.Tensor:
    """Compute local-to-global distillation loss following ref/ibot20260115.py.

    Averages over all local_student x teacher_global pairs. Local views are
    student queries only: with ``relational_targets`` each teacher slot keeps its
    current-plus-history candidate matrix, and the student statistics are
    aggregated per contributing pair.
    """
    if not local_views_student:
        return torch.tensor(0.0, device=teacher_global[0].device)
    device = local_views_student[0].device
    loss = torch.tensor(0.0, device=device)
    collected: list[dict[str, torch.Tensor]] = []
    for lv in local_views_student:
        for slot, t in enumerate(teacher_global):
            if relational_targets is not None:
                loss = loss + _relational_pair_loss(
                    lv,
                    slot,
                    relational_targets,
                    temp_student,
                    collected if student_stats is not None else None,
                )
                continue
            loss += _ssl_loss(lv, t, temp_student, temp_teacher)
    if student_stats is not None:
        _merge_student_statistics(collected, student_stats)
    loss /= max(1, len(local_views_student) * len(teacher_global))
    return loss


def _compute_dino_global_loss(
    student_logits: list[torch.Tensor],
    teacher_targets: list[torch.Tensor],
    temp_student: float,
) -> torch.Tensor:
    """Average the two cross-view pairs: student A -> teacher B, B -> A."""
    losses = [
        dino_distillation_loss(student_logits[0], teacher_targets[1], temp_student),
        dino_distillation_loss(student_logits[1], teacher_targets[0], temp_student),
    ]
    return torch.stack(losses).mean()


def _compute_dino_local_loss(
    local_logits: list[torch.Tensor],
    teacher_targets: list[torch.Tensor],
    temp_student: float,
) -> torch.Tensor:
    """Average every student-local -> teacher-global pair; zero without locals."""
    if not local_logits:
        return torch.zeros((), device=teacher_targets[0].device)
    losses = [
        dino_distillation_loss(logits, target, temp_student)
        for logits in local_logits
        for target in teacher_targets
    ]
    return torch.stack(losses).mean()


def _compute_koleo_raw(
    student_tokens: list[torch.Tensor],
    koleo_config: dict[str, Any],
) -> tuple[torch.Tensor, list[int]]:
    """Sum the per-global-crop KoLeo losses over the unmasked backbone CLS.

    Uses the existing unmasked global forwards, never locals and never a pooled
    cross-crop neighbor set, so two augmentations of one image cannot repel each
    other. Crops with fewer than two samples are skipped and counted.
    """
    device = student_tokens[0].device
    total = torch.zeros((), device=device)
    sizes: list[int] = []
    for tokens in student_tokens:
        backbone_cls = tokens[:, 0]
        count = int(backbone_cls.shape[0])
        sizes.append(count)
        if count < 2:
            continue
        total = total + koleo_loss(
            backbone_cls,
            normalize_eps=float(koleo_config["koleo_normalize_eps"]),
            distance_eps=float(koleo_config["koleo_distance_eps"]),
            log_eps=float(koleo_config["koleo_log_eps"]),
        )
    return total, sizes


def _dino_cls_diagnostics(
    teacher_targets: list[torch.Tensor], prototypes: int
) -> dict[str, float]:
    """CLS-target diagnostics over the cached assignments of both globals.

    Entropy and perplexity use finite-safe arithmetic: zero probabilities
    contribute zero entropy instead of NaN. These describe the teacher targets
    only; they are not morphology or collapse detection.
    """
    probabilities = torch.cat([t.detach().float() for t in teacher_targets], dim=0)
    positive = probabilities > 0
    log_probabilities = torch.where(
        positive, probabilities.log(), torch.zeros_like(probabilities)
    )
    entropy = float(-(probabilities * log_probabilities).sum(dim=-1).mean())

    marginal = probabilities.mean(dim=0)
    total = marginal.sum().clamp_min(1e-12)
    marginal = marginal / total
    marginal_positive = marginal > 0
    marginal_log = torch.where(
        marginal_positive, marginal.log(), torch.zeros_like(marginal)
    )
    perplexity = float(torch.exp(-(marginal * marginal_log).sum()))

    counts = torch.bincount(
        probabilities.argmax(dim=-1), minlength=prototypes
    ).float()
    active_ratio = float((counts > 0).float().mean())
    occupancy = float(counts.max() / counts.sum().clamp_min(1.0))
    return {
        "cls_assignment_entropy": entropy,
        "cls_prototype_perplexity": perplexity,
        "cls_active_prototype_ratio": active_ratio,
        "cls_max_prototype_occupancy": occupancy,
    }


def _koleo_instant_metrics(
    ssl_config: dict[str, Any],
    enabled: bool,
    koleo_raw: torch.Tensor,
    sizes: list[int],
) -> dict[str, Any]:
    """Status and counts for the optional KoLeo term (blank when N/A)."""
    weight = float(ssl_config["koleo_weight"])
    if not enabled:
        return {
            "koleo_status": "disabled",
            "koleo_loss_raw": None,
            "koleo_loss_weighted": None,
        }
    eligible = sum(size for size in sizes if size >= 2)
    skipped = sum(size for size in sizes if size < 2)
    if eligible == 0:
        # A skipped batch contributes exactly zero, still numeric.
        return {
            "koleo_status": "skipped",
            "koleo_loss_raw": 0.0,
            "koleo_loss_weighted": 0.0,
            "koleo_eligible_samples": 0,
            "koleo_skipped_samples": skipped,
        }
    raw = float(koleo_raw.item())
    return {
        "koleo_status": "active",
        "koleo_loss_raw": raw,
        "koleo_loss_weighted": weight * raw,
        "koleo_eligible_samples": eligible,
        "koleo_skipped_samples": skipped,
    }


def _update_teacher_center(
    center: torch.Tensor,
    teacher_global: torch.Tensor,
) -> torch.Tensor:
    """Update teacher center buffer with EMA of current-iteration teacher global outputs.

    Uses fixed 0.9/0.1 EMA coefficients matching ref/ibot20260115.py.
    """
    batch_mean = teacher_global.mean(dim=0, keepdim=True)
    return center * 0.9 + batch_mean * 0.1


def _ssl_loss(
    student_out: torch.Tensor,
    teacher_out: torch.Tensor,
    temp_student: float,
    temp_teacher: float,
    *,
    candidates: torch.Tensor | None = None,
    teacher_probs: torch.Tensor | None = None,
    stats: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Batch-only relational loss, or its rectangular candidate extension.

    ``candidates`` (``[B + E, D]``) and ``teacher_probs`` (``[B, B + E]``) must
    be supplied together; omitting both keeps the historical body unchanged.
    ``stats`` collects detached student mass from exactly the logits used here.
    """
    if (candidates is None) != (teacher_probs is None):
        raise ValueError(
            "candidates and teacher_probs must be supplied together; got "
            f"candidates={'set' if candidates is not None else None} and "
            f"teacher_probs={'set' if teacher_probs is not None else None}."
        )
    if candidates is not None:
        student_sim = student_out @ candidates.T / temp_student
        student_log_probs = F.log_softmax(student_sim, dim=1)
        if stats is not None:
            stats.update(
                _student_row_statistics(
                    student_sim,
                    F.softmax(student_sim.detach(), dim=1),
                    int(student_out.shape[0]),
                )
            )
        return -(teacher_probs * student_log_probs).sum(dim=1).mean()
    student_sim = student_out @ teacher_out.T / temp_student
    with torch.no_grad():
        teacher_sim = teacher_out @ teacher_out.T / temp_teacher
        teacher_sim = teacher_sim - teacher_sim.mean(dim=1, keepdim=True)
        teacher_probs = F.softmax(teacher_sim, dim=1)
    student_log_probs = F.log_softmax(student_sim, dim=1)
    return -(teacher_probs * student_log_probs).sum(dim=1).mean()


def _update_patch_center(
    patch_center: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
    """EMA of the iBOT patch center from current pre-EMA teacher logits."""
    return patch_center * PATCH_CENTER_MOMENTUM + teacher_logits.mean(
        dim=0, keepdim=True
    ) * (1.0 - PATCH_CENTER_MOMENTUM)
