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

# SSL training identity, kept separate from the inference embedding layout: a
# DINO checkpoint trains a prototype head but still extracts the same normalized
# 2048-hidden MLP bottleneck as relational.
SSL_METHODS = ("relational", "dino")
SSL_HEADS = ("projection_mlp_2048", "dino_mlp_2048")
SSL_HEAD_BY_METHOD = {"relational": "projection_mlp_2048", "dino": "dino_mlp_2048"}
DINO_CENTERING_MODES = ("ema", "sinkhorn-knopp")
DINO_PROTOTYPES_MIN = 2

REGISTER_TOKEN_CHOICES = ("none", "0", "4")

PRETRAIN_AUGMENTATIONS: tuple[str, ...] = ("global-barcode", "color-robust", "legacy")
FINETUNE_AUGMENTATIONS: tuple[str, ...] = ("none", "conservative")
ORIENTATION_POLICIES: tuple[str, ...] = ("invariant", "sensitive")

# Relational target selectors. ``LEGACY_RELATIONAL_PAIRING`` is decode-only: it
# carries a historical ``--disable-cross-view-loss=true`` checkpoint forward and
# is never a request choice.
RELATIONAL_TEACHER_VIEW_CHOICES: tuple[str, ...] = ("weak", "ordinary")
RELATIONAL_CENTER_SOURCE_CHOICES: tuple[str, ...] = ("weak", "ordinary")
RELATIONAL_PAIRING_CHOICES: tuple[str, ...] = ("cross", "full")
LEGACY_RELATIONAL_PAIRING = "same"

EVAL_TRANSFORM_CHOICES = ("center-crop", "whole-specimen-pad")

# ---------------------------------------------------------------------------
# Spatial geometry and feature reading (design 2026-11-12). Version-1 products
# keep these identifiers fixed: they name a stored representation contract, not
# a tunable.
# ---------------------------------------------------------------------------

# Input normalization applied once after every geometric operation.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

OUTPUT_KIND_CHOICES = ("embedding", "spatial")
FEATURE_NORM_CHOICES = ("backbone", "none")
SPATIAL_LAYOUT_CHOICES = ("per-image", "sharded")
SPATIAL_VIEW_ROLES = ("base", "eval-reference", "eval-transform")

SPATIAL_SCHEMA_VERSION = 1
SPATIAL_PROTOCOL = "spatial-geometry-v1"
SPATIAL_SUPPORT_VERSION = "conservative-resampling-v1"
SPATIAL_COORDINATE_ADAPTER = "coco-pixel-index-to-edge-v1"
SPATIAL_MASK_SAMPLING = "categorical-nearest-neighbour-v1"
SPATIAL_KEY_RULE = "artifact-key-resolution-v1"
SPATIAL_ONNX_METADATA_KEY = "otuformer.spatial.v1"

SPATIAL_L2_EPSILON = 1e-12
SPATIAL_EVAL_SAMPLE_SIZE = 100
SPATIAL_QUERY_LIMIT = 64
SPATIAL_PCK_THRESHOLDS = (0.01, 0.02, 0.05)
SPATIAL_EVAL_VIEW_IDS = (
    "reference",
    "crop075",
    "tx10",
    "ty10",
    "scale090",
    "scale110",
    "rot-15",
    "rot15",
    "rot90",
    "hflip",
    "vflip",
    "combo",
)
SPATIAL_ATTENTION_LIMITATION = (
    "Fully valid patches exclude direct fill contamination, but ViT global "
    "attention can still propagate influence from invalid tokens."
)
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
