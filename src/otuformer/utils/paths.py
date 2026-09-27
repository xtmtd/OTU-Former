"""Shared path canonicalization and file identity (stdlib only).

Both the training provenance code and the pseudo-label discovery code use these
helpers so their notion of "same file" and "same reference" cannot drift.
"""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

INODE_MODE = "inode"
REALPATH_MODE = "normalized-realpath"


def canonical_image_ref(ref: object, images_root: Path) -> str:
    """Canonical POSIX reference relative to ``images_root``.

    Relative references are normalized lexically and never resolved through the
    filesystem, so a recursively discovered image path does not change the
    manifest hash. An absolute reference must stay inside the image root. This
    is the existing v0.8.0 definition: no Unicode normalization, duplicates
    preserved.
    """
    text = str(ref)
    path = Path(text)
    if path.is_absolute():
        root = images_root.resolve()
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(
                f"Manifest image reference escapes --input-images-dir: {text}"
            )
        return resolved.relative_to(root).as_posix()
    return Path(os.path.normpath(text)).as_posix()


def path_identity(resolved: Path) -> object:
    """Per-scan file identity: nonzero ``(st_dev, st_ino)``, else NFC realpath."""
    stat = resolved.stat()
    if stat.st_ino:
        return (INODE_MODE, stat.st_dev, stat.st_ino)
    return (REALPATH_MODE, realpath_key(resolved))


def realpath_key(resolved: Path) -> str:
    """Portable fallback identity: NFC-normalized, case-folded real path."""
    return unicodedata.normalize("NFC", os.path.normcase(str(resolved)))
