"""OTU-Former: image-based morphological OTU delineation toolkit.

This module arms the MPS CPU-fallback switch before any OTU-Former entry path
can touch the device. ``PYTORCH_ENABLE_MPS_FALLBACK`` only affects MPS, and it
is latched when PyTorch initialises: a few antialiased resampling backward
kernels this project reaches are unimplemented there, and without the fallback
the step aborts with ``NotImplementedError``.

Setting the variable from a command function or a training loop is therefore
too late. Importing ``otuformer`` is the earliest point the library controls;
when a host process has already imported ``torch``, that point is already past,
and this module reports that instead of pretending the run is protected.
"""

import enum
import os
import sys

_SWITCH_REQUESTED = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
# Snapshot the import order before we touch anything.
_TORCH_ALREADY_IMPORTED = "torch" in sys.modules

# Only MPS reads this. ``setdefault`` keeps an explicit operator choice,
# including an explicit opt-out.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.12.0"


@enum.unique
class MPSFallbackState(enum.Enum):
    """What this process knows about the MPS CPU-fallback switch.

    ``ARMED``: this import set the variable before PyTorch started, so the
    switch took effect. ``DISABLED``: the operator opted out explicitly.
    ``UNKNOWN``: PyTorch was already imported, so the variable may have been set
    by the operator beforehand (effective) or by this import (too late). The
    package cannot tell, and must not claim either.
    """

    ARMED = "armed"
    DISABLED = "disabled"
    UNKNOWN = "unknown"


#: The variable now in the environment is not proof the backend latched it.
MPS_FALLBACK_STATE = (
    MPSFallbackState.UNKNOWN
    if _TORCH_ALREADY_IMPORTED
    else (
        MPSFallbackState.DISABLED
        if _SWITCH_REQUESTED == "0"
        else MPSFallbackState.ARMED
    )
)

# True only when the switch is known to have been armed before PyTorch started.
MPS_FALLBACK_EFFECTIVE = MPS_FALLBACK_STATE is MPSFallbackState.ARMED
