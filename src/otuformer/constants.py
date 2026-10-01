"""Constants shared by the CLI and the training code.

Kept import-light so the CLI stays torch-free at import time.
"""

# Sub-center capacity bound, shared by both Sub-center modes. The compact
# penalty averages over K(K-1)/2 same-class pairs, so the bound keeps that
# quadratic term and the (C, K, D) head modest. K=1 is excluded because it is
# numerically ArcFace, which already has its own mode.
MIN_SUBCENTERS = 2
MAX_SUBCENTERS = 8

# Supervised losses with a class-level cross-entropy term. CB-DRW and pseudo
# feedback apply only to these; SupCon rejects both.
ARCFACE_FAMILY_LOSSES = (
    "arcface",
    "subcenter-arcface",
    "subcenter-arcface-compact",
)

# ---------------------------------------------------------------------------
# Canonical finite values shared by option declarations, CLI/core validators,
# and the read-only schema export. They live here (import-light, torch-free) so
# the CLI and the schema never keep a second copy of an accepted value set.
# ---------------------------------------------------------------------------

PATCH_LOSS_MODES = ("none", "consistency", "masked-feature", "ibot")
MASKING_STRATEGIES = ("random", "blockwise", "hybrid")
IBOT_PROTOTYPES_MIN = 2

REGISTER_TOKEN_CHOICES = ("none", "0", "4")

PRETRAIN_AUGMENTATIONS: tuple[str, ...] = ("global-barcode", "color-robust", "legacy")
FINETUNE_AUGMENTATIONS: tuple[str, ...] = ("none", "conservative")
ORIENTATION_POLICIES: tuple[str, ...] = ("invariant", "sensitive")

EVAL_TRANSFORM_CHOICES = ("center-crop", "whole-specimen-pad")
TOKEN_MODES = ("cls", "patch-topk", "attention-pool")
ATTENTION_POOLING_TYPES = ("lightweight", "multihead", "gated")

CAM_METHOD_CHOICES = (
    "gradcam",
    "gradcampp",
    "layercam",
    "scorecam",
    "eigencam",
    "ablationcam",
)
CAM_ARCH_CHOICES = ("cnn", "vit")
CAM_FIG_FORMAT_CHOICES = ("png", "jpg", "pdf")
CAM_SAVE_NPY_CHOICES = ("none", "raw", "normalized")
CAM_DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")

LOSS_CHOICES = (
    "arcface",
    "subcenter-arcface",
    "subcenter-arcface-compact",
    "supcon",
)
LONG_TAIL_MODES = ("none", "cb-drw")

# Supplemental numeric bounds checked by shared validators, not by the parser.
MASK_RATIO_BOUNDS = (0.0, 1.0)  # exclusive on both ends; "auto" is allowed
PSEUDO_SIMILARITY_BOUNDS = (-1.0, 1.0)  # inclusive
PSEUDO_MIN_GAP_BOUNDS = (0.0, 2.0)  # inclusive
