"""Mask decoding for instance polygons/RLE and panoptic segment-ID PNGs.

``pycocotools`` is an optional extra: it is imported only when a supplied
annotation actually needs polygon/RLE operations, and a missing dependency fails
with installation guidance instead of silently skipping geometry checks.
"""

from __future__ import annotations

import copy
import io
import warnings
from contextlib import contextmanager
from typing import Any

import numpy as np

INSTALL_HINT = (
    "polygon/RLE annotations need the optional decoder: "
    "install it with `pip install 'otuformer[annotations]'`"
)


class MaskDependencyError(RuntimeError):
    """The optional mask decoder is not installed."""


class MaskDecodeError(ValueError):
    """A supplied segmentation cannot be decoded as valid COCO geometry."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require_pycocotools():
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:  # pragma: no cover - exercised with an import blocker
        raise MaskDependencyError(INSTALL_HINT) from exc
    return mask_utils


def decoder_version() -> str:
    try:
        from importlib.metadata import version

        return f"pycocotools-{version('pycocotools')}"
    except Exception:  # pragma: no cover - metadata always present when importable
        return "pycocotools-unknown"


def polygon_parts(segmentation: Any) -> list[list[float]]:
    """Return polygon parts for a list segmentation, or [] for RLE/absent input."""
    if not isinstance(segmentation, list):
        return []
    return [list(part) for part in segmentation]


def polygon_within_bounds(segmentation: Any, width: int, height: int) -> bool:
    """True when no polygon vertex leaves ``[0,width] x [0,height]``.

    Invalid geometry is rejected by validation, so it must never reach the
    rasterizer afterwards.
    """
    for part in polygon_parts(segmentation):
        for offset in range(0, len(part) - 1, 2):
            x, y = part[offset], part[offset + 1]
            if not (0 <= x <= width and 0 <= y <= height):
                return False
    return True


@contextmanager
def quiet_pycocotools_decode():
    """Silence pycocotools' NumPy 2 ``__array__`` deprecation around one decode.

    pycocotools 2.0.11 (still the current release) converts its Cython mask without
    the ``copy`` keyword NumPy 2 requires; the warning is raised from its own frame
    and no release fixes it. Suppressed exactly at the call site, never globally: it
    also silences the warning for CLI users, and any other warning still surfaces.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"__array__ implementation doesn't accept a copy keyword",
            category=DeprecationWarning,
        )
        yield


def decode_instance_mask(
    annotation: dict[str, Any], width: int, height: int
) -> np.ndarray:
    """Decode one annotation's segmentation to a ``(height, width)`` uint8 mask."""
    segmentation = copy.deepcopy(annotation.get("segmentation"))
    if segmentation is None:
        raise MaskDecodeError("no_segmentation", "annotation has no segmentation")
    mask_utils = require_pycocotools()
    rle = _to_rle(mask_utils, segmentation, width, height)
    try:
        with quiet_pycocotools_decode():
            mask = mask_utils.decode(rle)
    except ValueError as exc:
        raise MaskDecodeError("decode_failed", f"cannot decode mask: {exc}") from exc
    if mask.shape != (height, width):
        raise MaskDecodeError(
            "decoded_shape_mismatch",
            f"decoded mask has shape {mask.shape}, expected {(height, width)}",
        )
    return np.asarray(mask, dtype=np.uint8)


def _to_rle(mask_utils, segmentation: Any, width: int, height: int):
    try:
        return _to_rle_checked(mask_utils, segmentation, width, height)
    except MaskDecodeError:
        raise
    except (ValueError, TypeError, UnicodeError, KeyError, IndexError, AttributeError) as exc:
        # Decoder-side input errors (a non-ASCII compact RLE string, a wrong
        # counts shape) must become one stable validation code, not an exception
        # that escapes the structured report.
        raise MaskDecodeError("decode_failed", f"cannot decode mask: {exc}") from exc


def _to_rle_checked(mask_utils, segmentation: Any, width: int, height: int):
    if isinstance(segmentation, list):
        if not segmentation:
            raise MaskDecodeError("empty_polygon_list", "segmentation has no polygon part")
        try:
            return mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
        except ValueError as exc:
            raise MaskDecodeError("invalid_polygon", f"cannot rasterize polygon: {exc}") from exc
    if isinstance(segmentation, dict):
        size = segmentation.get("size")
        if (
            not isinstance(size, list)
            or len(size) != 2
            or [int(value) for value in size] != [height, width]
        ):
            raise MaskDecodeError(
                "rle_size_mismatch",
                f"RLE size {size!r} does not match {(height, width)}",
            )
        counts = segmentation.get("counts")
        if isinstance(counts, list):
            _require_run_totals(counts, width, height)
            try:
                return mask_utils.frPyObjects(segmentation, height, width)
            except ValueError as exc:
                raise MaskDecodeError("invalid_rle", f"cannot read RLE: {exc}") from exc
        if isinstance(counts, str):
            try:
                return {"size": [height, width], "counts": counts.encode("ascii")}
            except UnicodeError as exc:
                raise MaskDecodeError(
                    "invalid_rle_counts",
                    f"RLE 'counts' string is not ASCII: {exc}",
                ) from exc
        raise MaskDecodeError("invalid_rle_counts", "RLE 'counts' must be a list or string")
    raise MaskDecodeError(
        "invalid_segmentation", "segmentation must be a polygon list or an RLE object"
    )


def _require_run_totals(counts: list[Any], width: int, height: int) -> None:
    """The decoder accepts an under-covering run list, so check it here."""
    if not counts or not all(isinstance(value, int) and not isinstance(value, bool) for value in counts):
        raise MaskDecodeError("invalid_rle_counts", "RLE counts must be integers")
    if any(value < 0 for value in counts):
        raise MaskDecodeError("invalid_rle_counts", "RLE counts must be nonnegative")
    if sum(counts) != width * height:
        raise MaskDecodeError(
            "rle_run_total_mismatch",
            f"RLE runs cover {sum(counts)} pixels, expected {width * height}",
        )


def rasterized_diagnostics(
    annotation: dict[str, Any], width: int, height: int
) -> dict[str, Any]:
    """Diagnostic area/bbox from COCO rasterization, or {} when not applicable."""
    mask_utils = require_pycocotools()
    rle = _to_rle(mask_utils, copy.deepcopy(annotation.get("segmentation")), width, height)
    try:
        with quiet_pycocotools_decode():
            mask = mask_utils.decode(rle)
    except ValueError as exc:
        raise MaskDecodeError("decode_failed", f"cannot decode mask: {exc}") from exc
    return {
        "decoder": decoder_version(),
        "rasterized_area": float(np.count_nonzero(mask)),
        "rasterized_bbox": [float(value) for value in mask_utils.toBbox(rle)],
    }


def decode_panoptic_map(rgb_bytes: bytes, width: int, height: int) -> np.ndarray:
    """Decode an RGB segment-ID PNG into ``(height, width)`` unsigned ids."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(rgb_bytes)) as image:
            if image.format != "PNG":
                # A PNG name does not make a PNG; the pixel contract is PNG-only.
                raise MaskDecodeError(
                    "panoptic_not_png",
                    f"panoptic mask must be PNG, got format {image.format!r}",
                )
            if image.mode != "RGB":
                raise MaskDecodeError(
                    "panoptic_not_rgb",
                    f"panoptic PNG must be RGB, got mode {image.mode!r}",
                )
            if image.size != (width, height):
                raise MaskDecodeError(
                    "panoptic_size_mismatch",
                    f"panoptic PNG is {image.size}, expected {(width, height)}",
                )
            pixels = np.asarray(image, dtype=np.uint32)
    except MaskDecodeError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise MaskDecodeError(
            "panoptic_decode_failed", f"cannot decode panoptic PNG: {exc}"
        ) from exc
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    return red + np.uint32(256) * green + np.uint32(256 * 256) * blue


def panoptic_segment_stats(
    ids: np.ndarray, segment_ids: list[int]
) -> dict[int, dict[str, float]]:
    """Actual pixel area and pixel-derived bbox per declared segment id."""
    stats: dict[int, dict[str, float]] = {}
    for segment_id in segment_ids:
        rows, columns = np.nonzero(ids == segment_id)
        if rows.size == 0:
            stats[segment_id] = {"area": 0.0, "bbox": [0.0, 0.0, 0.0, 0.0]}
            continue
        min_row, max_row = int(rows.min()), int(rows.max())
        min_column, max_column = int(columns.min()), int(columns.max())
        stats[segment_id] = {
            "area": float(rows.size),
            "bbox": [
                float(min_column),
                float(min_row),
                float(max_column - min_column + 1),
                float(max_row - min_row + 1),
            ],
        }
    return stats
