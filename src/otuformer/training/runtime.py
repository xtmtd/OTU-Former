"""Seeding, device selection and the student/teacher EMA update."""

from __future__ import annotations

import random
import time
import numpy as np
import torch
import torch.nn as nn

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_cpus(cpus: int) -> None:
    if cpus > 0:
        torch.set_num_threads(cpus)


def _phase_start(label: str) -> float:
    """Announce a startup phase before its slow operation begins.

    A start with no matching completion means the phase is still running (or
    failed), which is exactly what a silent hang needs to look like. Wall time
    only: asynchronous device work is never synchronized here.
    """
    print(f"[Phase] {label}: start")
    return time.monotonic()


def _phase_done(label: str, started: float) -> None:
    print(f"[Phase] {label}: done in {time.monotonic() - started:.1f}s")


def _capture_rng_state() -> dict[str, object]:
    """Snapshot Python, NumPy, PyTorch CPU, and CUDA RNG states."""
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, object] | None) -> None:
    """Restore a snapshot produced by :func:`_capture_rng_state`."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda = state.get("cuda")
    if cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda)


def mps_fallback_state():
    """The package's knowledge of the switch: armed, disabled, or unknown."""
    from otuformer import MPS_FALLBACK_STATE

    return MPS_FALLBACK_STATE


def warn_if_mps_fallback_unarmed(device) -> None:
    """Warn on MPS only when the run really is unprotected.

    Setting the variable here cannot change an already-initialised MPS backend,
    and an unknown state may still be armed by the operator, so neither case
    may be reported as "you forgot to set it".
    """
    from otuformer import MPSFallbackState

    if device.type != "mps":
        return
    state = mps_fallback_state()
    if state is MPSFallbackState.UNKNOWN:
        print(
            "[Warning] MPS is selected and PyTorch was imported before "
            "otuformer, so this process cannot confirm that "
            "PYTORCH_ENABLE_MPS_FALLBACK took effect. If the process did not "
            "set PYTORCH_ENABLE_MPS_FALLBACK=1 before starting, some "
            "augmentation backward kernels are unimplemented on MPS and will "
            "abort; start the process with PYTORCH_ENABLE_MPS_FALLBACK=1."
        )
    elif state is MPSFallbackState.DISABLED:
        print(
            "[Warning] MPS is selected and PYTORCH_ENABLE_MPS_FALLBACK=0 was "
            "set explicitly. Some augmentation backward kernels are "
            "unimplemented on MPS and will abort unless the process is "
            "started with PYTORCH_ENABLE_MPS_FALLBACK=1."
        )


def _resolve_device(device: str) -> torch.device:
    requested = (device or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


@torch.no_grad()
def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    """Name-matched EMA update.

    Positional ``zip`` is unsafe once the student owns parameters the teacher
    does not (mask token, A+ predictor), so parameters are matched by name and a
    missing teacher counterpart is a hard error.
    """
    student_params = dict(student.named_parameters())
    for name, teacher_param in teacher.named_parameters():
        if name not in student_params:
            raise ValueError(
                f"Cannot EMA-update '{name}': the student module has no parameter "
                "with that name. Student-only parameters must not appear on the "
                "teacher module."
            )
        teacher_param.mul_(momentum).add_(student_params[name], alpha=1.0 - momentum)
