"""Tests for train-only UMAP label/title semantics in run_umap()."""

from __future__ import annotations

import inspect

from otuformer.embedding import evaluator


def test_train_only_umap_does_not_append_split_suffix_to_labels():
    """run_umap() should not inject (train) or similar text into legend labels.
    Labels must be rendered exactly as passed by the caller."""
    src = inspect.getsource(evaluator.run_umap)
    assert "(train)" not in src, "run_umap must not hardcode '(train)' in legend labels"
    assert "(test)" not in src, "run_umap must not hardcode '(test)' in legend labels"
    # Label is set to str(lbl) — raw class label only
    assert "label=str(lbl)" in src, (
        "run_umap must set scatter label to str(lbl) without a split suffix"
    )


def test_umap_title_is_caller_controlled():
    """run_umap() must accept a title parameter and use it instead of a hardcoded string."""
    sig = inspect.signature(evaluator.run_umap)
    assert "title" in sig.parameters, "run_umap must accept a 'title' parameter"
    src = inspect.getsource(evaluator.run_umap)
    assert "UMAP Train" not in src, (
        "run_umap must not hardcode 'UMAP Train' — title must be caller-controlled"
    )
