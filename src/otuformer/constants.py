"""Constants shared by the CLI and the training code.

Kept import-light so the CLI stays torch-free at import time.
"""

# Sub-center capacity bound, shared by both Sub-center modes. The compact
# penalty averages over K(K-1)/2 same-class pairs, so the bound keeps that
# quadratic term and the (C, K, D) head modest. K=1 is excluded because it is
# numerically ArcFace, which already has its own mode.
MIN_SUBCENTERS = 2
MAX_SUBCENTERS = 8
