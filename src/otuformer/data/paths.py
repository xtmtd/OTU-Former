"""Path identity and containment for the v2 reader (stdlib only).

The internal ``image_key`` is the normalized POSIX path relative to the image
root. It is stable when the root moves, not when an image is renamed.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

# Mirrors the extractor's supported suffixes without importing torch.
IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)


class ReferenceError(ValueError):
    """An image reference is unusable (missing, escaping, ambiguous, empty)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def normalize_key(reference: str) -> str:
    """Lexical POSIX normalization; never strips leading characters."""
    return Path(os.path.normpath(reference)).as_posix()


def is_within(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents


def relative_key(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def require_contained(root: Path, path: Path, text: str) -> Path:
    """Re-resolve ``path`` now and require it to stay under ``root``.

    Containment is checked at the moment of consumption as well as at selection:
    a file swapped for an escaping symlink after validation has valid-looking
    bytes but no longer belongs to the image root.
    """
    root_resolved = Path(root).expanduser().resolve()
    resolved = Path(path).expanduser().resolve()
    if not is_within(root_resolved, resolved):
        raise ReferenceError(
            "root_escape", f"image resolves outside the root: {text}"
        )
    return resolved


def _root_resolved(root: Path, resolved_root: Path | None = None) -> Path:
    """The root's real path, preferring one the caller already resolved.

    A relative root resolves against the working directory, which is why the
    written spelling is never compared as text: only this path is compared.
    """
    if resolved_root is not None:
        return Path(resolved_root).expanduser().resolve()
    return Path(root).expanduser().resolve()


def _written_key(root_resolved: Path, reference_path: Path) -> str | None:
    """The root-relative spelling an absolute reference actually wrote.

    The root prefix is matched on *resolved* components, not on text: the root may
    be reached through a link, and the same directory can be spelled two ways
    (``/tmp`` and ``/private/tmp`` on macOS). Matching text would miss both and
    collapse an internal alias into the real file name.
    """
    parts = reference_path.parts
    for index in range(1, len(parts)):
        try:
            prefix = Path(*parts[:index]).resolve()
        except OSError:  # pragma: no cover - defensive
            continue
        if prefix == root_resolved:
            return normalize_key("/".join(parts[index:]))
    return None


def _absolute_identity(
    root_resolved: Path, reference_path: Path
) -> tuple[str, Path]:
    """Split one absolute reference into its lexical key and resolved access path.

    The key keeps the reference's own spelling relative to the root, so an absolute
    reference through an internal symlink addresses the same alias a relative
    reference does. The access path is the real file, and containment applies to it.
    """
    resolved = reference_path.resolve()
    if not is_within(root_resolved, resolved):
        raise ReferenceError(
            "root_escape", f"absolute image reference escapes the root: {reference_path}"
        )
    # Keep the tail the reference wrote, so `linked/alias.png` and `alias.png` are
    # one identity. A folded key is only kept when it addresses the file the
    # reference reaches: `link/../one.png` folds to `one.png` while the filesystem
    # target is a/one.png.
    key = _written_key(root_resolved, reference_path)
    if key is not None:
        return _honest_key(root_resolved, key, resolved), resolved
    return relative_key(root_resolved, resolved), resolved


def lexical_key(
    root: Path, reference: str, *, resolved_root: Path | None = None
) -> str:
    """Root-relative key for one reference, as a source records it.

    Rejects URL-like references, references that normalize above the root, and
    absolute references that resolve outside it. Symlinks are resolved only to
    verify that the key addresses the file the reference actually reaches.
    """
    text = str(reference)
    if not text.strip():
        raise ReferenceError("empty_reference", "empty image reference")
    if "://" in text:
        raise ReferenceError(
            "unsupported_reference", f"URL-like image reference is not supported: {text}"
        )
    root_resolved = _root_resolved(root, resolved_root)
    reference_path = Path(text).expanduser()
    if reference_path.is_absolute():
        return _absolute_identity(root_resolved, reference_path)[0]
    key = normalize_key(text)
    if key == ".." or key.startswith("../"):
        raise ReferenceError("root_escape", f"image reference escapes the root: {text}")
    # Same written-path resolution as the selected rows: a source reference that
    # only looks contained after `..` folding must be rejected here too.
    return _honest_key(root_resolved, key, _resolve_written(root_resolved, text))


def resolve_reference(
    root: Path,
    reference: str,
    *,
    basename_fallback: bool = False,
    resolved_root: Path | None = None,
) -> tuple[str, Path]:
    """Return ``(image_key, access_path)`` for one reference under ``root``.

    The path is resolved as written before any lexical normalization, so a
    component that is a symlink out of the root is caught even when ``..`` later
    folds back in. Relative references keep their lexical spelling as the key; an
    absolute reference keeps its own spelling, unless that spelling folds to a
    different file than the one it reaches. ``basename_fallback`` applies to
    a bare file name only: matching the basename of ``a/one.png`` to ``b/one.png``
    would silently read a different image than the one requested.
    """
    text = str(reference)
    if not text.strip():
        raise ReferenceError("empty_reference", "empty image reference")
    if "://" in text:
        raise ReferenceError(
            "unsupported_reference", f"URL-like image reference is not supported: {text}"
        )

    root_resolved = _root_resolved(root, resolved_root)
    reference_path = Path(text).expanduser()
    if reference_path.is_absolute():
        key, resolved = _absolute_identity(root_resolved, reference_path)
        if not resolved.is_file():
            raise ReferenceError(
                "missing_image", f"image not found under the root: {text}"
            )
        return key, resolved

    key = normalize_key(text)
    # Resolve the written path segment by segment: containment is a property of the
    # real target, so `link/../one.png` cannot fold out of a symlinked `link` and
    # land on a different in-root image.
    resolved = _resolve_written(root_resolved, text)
    if resolved.exists():
        return _honest_key(root_resolved, key, resolved), resolved

    if basename_fallback and "/" not in key:
        matched = _unique_basename_match(root_resolved, key, text)
        if matched is not None:
            matched_resolved = matched.resolve()
            if not is_within(root_resolved, matched_resolved):
                raise ReferenceError(
                    "root_escape",
                    f"basename match resolves outside the root: {text}",
                )
            return relative_key(root_resolved, matched), matched_resolved

    raise ReferenceError("missing_image", f"image not found under the root: {text}")


def _resolve_written(root_resolved: Path, text: str) -> Path:
    """Resolve a written relative reference, checking containment at every step.

    Each step continues from the *resolved* position: with ``link -> a/b``, the
    reference ``link/../one.png`` means ``a/one.png`` on the filesystem, because
    ``..`` applies to where the link points, not to where it is written.
    """
    current = root_resolved
    for part in PurePosixPath(text).parts:
        if part in ("", "."):
            continue
        current = current.parent if part == ".." else current / part
        current = current.resolve()
        if not is_within(root_resolved, current):
            raise ReferenceError(
                "root_escape", f"image reference resolves outside the root: {text}"
            )
    return current


def _honest_key(root_resolved: Path, key: str, resolved: Path) -> str:
    """Keep the written key only when it names the file the reference reaches."""
    try:
        lexical = (root_resolved / key).resolve()
    except OSError:  # pragma: no cover - defensive
        return relative_key(root_resolved, resolved)
    if lexical == resolved:
        return key
    if is_within(root_resolved, resolved):
        return relative_key(root_resolved, resolved)
    raise ReferenceError(
        "root_escape", f"image reference resolves outside the root: {key}"
    )


def _unique_basename_match(root: Path, basename: str, text: str) -> Path | None:
    """Unique recursive basename match, or None when there is no candidate."""
    matches: list[Path] = []
    for directory, _subdirs, files in os.walk(root, followlinks=False):
        if basename in files:
            matches.append(Path(directory) / basename)
            if len(matches) > 1:
                raise ReferenceError(
                    "ambiguous_reference",
                    f"basename {basename!r} matches several files: {text}",
                )
    return matches[0] if matches else None


def iter_image_files(root: Path) -> list[Path]:
    """Supported image files under ``root`` sorted by lexical root-relative key."""
    found = [
        Path(directory) / name
        for directory, _subdirs, files in os.walk(root, followlinks=False)
        for name in files
        if Path(name).suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(found, key=lambda path: relative_key(root, path))
