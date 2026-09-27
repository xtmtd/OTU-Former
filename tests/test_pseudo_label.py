"""Candidate discovery, raw-CLS scoring, and pseudo-label diagnostics."""

import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from otuformer.embedding import pseudo_label as pl


def unit(angle_deg: float) -> np.ndarray:
    rad = math.radians(angle_deg)
    return np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)


def _expert(label: str, angle: float, ref: str | None = None) -> tuple[dict, str, np.ndarray]:
    ref = ref or f"{label}_{angle}.jpg"
    return {"image": ref, "label": label}, ref, unit(angle)


def _features(*triples) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for _row, ref, vector in triples:
        out[ref] = vector
    return out


def _cand(ref: str, angle: float) -> tuple[dict, str, np.ndarray]:
    return {"image": ref}, ref, unit(angle)


def _decide(expert_triples, candidate_triples, **kwargs):
    expert_rows = [t[0] for t in expert_triples]
    candidate_rows = [t[0] for t in candidate_triples]
    features = _features(*expert_triples, *candidate_triples)
    options = {"similarity_floor": 0.5, "min_gap": 0.1, "neighbors": 5}
    options.update(kwargs)
    return pl.decide_candidates(
        expert_rows, candidate_rows, features, **options
    )


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def _touch(root: Path, rel: str) -> Path:
    import zlib

    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = zlib.crc32(rel.encode("utf-8"))
    color = (seed % 256, (seed >> 8) % 256, (seed >> 16) % 256)
    Image.new("RGB", (8, 8), color).save(path)
    return path


def test_discovery_subtracts_experts_and_ignores_non_images(tmp_path):
    root = tmp_path / "images"
    _touch(root, "a.jpg")
    _touch(root, "sub/b.png")
    (root / "notes.txt").write_text("x")
    expert = [{"image": "a.jpg", "label": "A"}]

    candidates, rejections, summary = pl.discover_candidates(expert, root)

    assert [c["image"] for c in candidates] == ["sub/b.png"]
    assert rejections == []
    assert summary["discovered_candidate_count"] == 1


def test_discovery_does_not_follow_directory_symlinks(tmp_path):
    root = tmp_path / "images"
    real = tmp_path / "outside"
    _touch(real, "hidden.jpg")
    root.mkdir()
    (root / "linkdir").symlink_to(real, target_is_directory=True)
    _touch(root, "a.jpg")

    candidates, _rejections, summary = pl.discover_candidates([], root)

    assert [c["image"] for c in candidates] == ["a.jpg"]
    assert summary["skipped_directory_symlinks"] == ["linkdir"]
    assert summary["skipped_directory_symlink_count"] == 1


def test_discovery_keeps_in_root_file_symlink_link_path(tmp_path):
    root = tmp_path / "images"
    target = _touch(root, "sub/a.jpg")
    (root / "alias.jpg").symlink_to(target)

    candidates, rejections, _summary = pl.discover_candidates([], root)

    # The link and its target are the same resolved file: the alias keeps its
    # link-path reference and the duplicate target is rejected.
    assert [c["image"] for c in candidates] == ["alias.jpg"]
    assert rejections == [
        {"image": "sub/a.jpg", "rejection_reasons": ["duplicate_file_identity"]}
    ]


def test_discovery_rejects_escaping_file_symlink(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    outside = _touch(tmp_path / "outside", "x.jpg")
    (root / "escape.jpg").symlink_to(outside)

    candidates, rejections, _summary = pl.discover_candidates([], root)

    assert candidates == []
    assert rejections == [
        {"image": "escape.jpg", "rejection_reasons": ["escapes_image_root"]}
    ]


def test_discovery_duplicate_alias_keeps_canonical_first(tmp_path):
    root = tmp_path / "images"
    _touch(root, "a.jpg")
    os.link(root / "a.jpg", root / "b.jpg")

    candidates, rejections, _summary = pl.discover_candidates([], root)

    assert [c["image"] for c in candidates] == ["a.jpg"]
    assert rejections == [
        {"image": "b.jpg", "rejection_reasons": ["duplicate_file_identity"]}
    ]


def test_discovery_skips_strong_output_subtree_and_warns_on_weak(tmp_path):
    root = tmp_path / "data"
    _touch(root, "a.jpg")
    run = root / "runs" / "finetune1"
    (run / "logs").mkdir(parents=True)
    (run / "logs" / "finetune.log").write_text("log")
    (run / "finetune_latest.pth").write_bytes(b"weights")
    _touch(run, "candidate_inside_run.jpg")
    weak = root / "weak"
    (weak / "logs").mkdir(parents=True)
    (weak / "model.pth").write_bytes(b"weights")
    _touch(weak, "b.jpg")

    candidates, _rejections, summary = pl.discover_candidates([], root)

    refs = [c["image"] for c in candidates]
    assert "a.jpg" in refs
    assert "weak/b.jpg" in refs
    assert all("candidate_inside_run" not in ref for ref in refs)
    assert summary["skipped_output_subtree_count"] == 1
    assert summary["weak_signal_warnings"] == ["weak"]


def test_discovery_inode_collision_switches_to_realpath(tmp_path, monkeypatch):
    root = tmp_path / "images"
    _touch(root, "a.jpg")
    _touch(root, "b.jpg")

    real_stat = os.stat
    real_sha = pl._sha256_file

    def fake_stat(path, *args, **kwargs):
        stat = real_stat(path, *args, **kwargs)
        if Path(path).name in {"a.jpg", "b.jpg"}:
            values = list(stat)
            values[1] = 4242  # same st_dev/st_ino for different real paths
            return os.stat_result(values)
        return stat

    def fake_sha(path):
        if Path(path).name == "a.jpg":
            return "digest-a"
        if Path(path).name == "b.jpg":
            return "digest-b"
        return real_sha(path)

    monkeypatch.setattr(os, "stat", fake_stat)
    monkeypatch.setattr(pl, "_sha256_file", fake_sha)

    candidates, _rejections, summary = pl.discover_candidates([], root)

    # Same inode, different real paths, different bytes: inode metadata is
    # unreliable, so the whole scan falls back to normalized real paths.
    assert summary["file_identity_mode"] == pl.REALPATH_MODE
    assert summary["inode_collision_count"] == 1
    assert len(candidates) == 2


# --------------------------------------------------------------------------
# Decision rule
# --------------------------------------------------------------------------


def test_eligible_top3_mean_wins_and_undersupported_uses_max():
    a = [_expert("A", angle) for angle in (0.0, 1.0, 2.0)]
    b = [_expert("B", 90.0), _expert("B", 91.0)]
    candidate = _cand("c.jpg", 0.0)

    accepted, diagnostics, summary = _decide([*a, *b], [candidate])

    assert [row["label"] for row in accepted] == ["A"]
    assert diagnostics[0]["runner_up_class"] == "B"
    assert summary["eligible_class_count"] == 1
    assert summary["ineligible_class_count"] == 1


def test_floor_and_gap_equality_reject():
    a = [_expert("A", angle) for angle in (0.0, 1.0, 2.0)]
    b = [_expert("B", angle) for angle in (90.0, 91.0, 92.0)]
    candidate = _cand("c.jpg", 0.0)
    _accepted, diagnostics, _summary = _decide([*a, *b], [candidate])
    top1 = diagnostics[0]["top1_score"]

    _accepted, rejected, _summary = _decide(
        [*a, *b], [candidate], similarity_floor=top1
    )
    assert rejected[0]["rejection_reasons"] == ["floor"]


def test_gap_equality_rejects_with_eligible_and_undersupported_reasons():
    a = [_expert("A", angle) for angle in (0.0, 1.0, 2.0)]
    b = [_expert("B", 60.0), _expert("B", 61.0), _expert("B", 62.0)]
    candidate = _cand("c.jpg", 0.0)

    _accepted, diagnostics, _summary = _decide([*a, *b], [candidate])
    gap = diagnostics[0]["score_gap"]
    assert gap is not None

    _accepted, rejected, _summary = _decide(
        [*a, *b], [candidate], min_gap=gap
    )
    assert rejected[0]["rejection_reasons"] == ["gap_eligible"]

    # An under-supported runner-up vetoes through the same strict gap rule.
    c = [_expert("C", 0.5)]
    _accepted, rejected, _summary = _decide([*a, *c], [candidate], min_gap=gap)
    assert "gap_undersupported" in rejected[0]["rejection_reasons"]


def test_one_eligible_class_without_competitor_rejects():
    a = [_expert("A", angle) for angle in (0.0, 1.0, 2.0)]
    candidate = _cand("c.jpg", 0.0)

    accepted, diagnostics, _summary = _decide([*a], [candidate])

    assert accepted == []
    assert diagnostics[0]["rejection_reasons"] == ["no_runner_up"]


def test_mutual_knn_requires_the_same_seed_both_directions():
    a = [_expert("A", angle) for angle in (0.0, 10.0, 20.0)]
    b = [_expert("B", angle) for angle in (180.0, 181.0, 182.0)]
    c1 = _cand("c1.jpg", 0.5)
    c2 = _cand("c2.jpg", 0.6)

    accepted, diagnostics, _summary = _decide([*a, *b], [c1, c2], neighbors=1)

    accepted_refs = {row["image"] for row in accepted}
    assert accepted_refs == {"c1.jpg"}
    by_ref = {row["image"]: row for row in diagnostics}
    assert by_ref["c2.jpg"]["rejection_reasons"] == ["mutual_knn"]


def test_own_class_exclusion_lets_dense_class_candidate_pass():
    a = [_expert("A", angle) for angle in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)]
    b = [_expert("B", 180.0), _expert("B", 181.0), _expert("B", 182.0)]
    candidate = _cand("c.jpg", 6.0)

    accepted, _diagnostics, _summary = _decide([*a, *b], [candidate], neighbors=5)

    assert [row["label"] for row in accepted] == ["A"]


def test_invalid_candidate_rejected_and_invalid_seed_fails_the_run():
    a = [_expert("A", angle) for angle in (0.0, 1.0, 2.0)]
    b = [_expert("B", angle) for angle in (90.0, 91.0, 92.0)]
    good = _cand("c.jpg", 0.0)
    bad = {"image": "bad.jpg"}
    features = _features(*a, *b, good)
    features["bad.jpg"] = np.zeros(2, dtype=np.float32)

    accepted, diagnostics, _summary = pl.decide_candidates(
        [t[0] for t in a] + [t[0] for t in b],
        [good[0], bad],
        features,
        similarity_floor=0.5,
        min_gap=0.1,
        neighbors=5,
    )

    by_ref = {row["image"]: row for row in diagnostics}
    assert by_ref["bad.jpg"]["rejection_reasons"] == ["invalid_feature"]
    assert [row["label"] for row in accepted] == ["A"]

    zero_seed_features = {**features, "zero.jpg": np.zeros(2, dtype=np.float32)}
    with pytest.raises(pl.PseudoLabelError):
        pl.decide_candidates(
            [t[0] for t in a] + [{"image": "zero.jpg", "label": "A"}],
            [good[0]],
            zero_seed_features,
            similarity_floor=0.5,
            min_gap=0.1,
            neighbors=5,
        )


def test_class_cap_ranks_by_score_then_reference():
    a = [_expert("A", angle) for angle in (0.0, 0.2, 0.4)]
    b = [_expert("B", angle) for angle in (90.0, 90.2, 90.4)]
    candidates = [_cand(f"c{i:02d}.jpg", 0.0) for i in range(11)]

    accepted, diagnostics, summary = _decide([*a, *b], candidates, neighbors=32)

    assert len(accepted) == 9  # min(3*3, 30)
    # Equal scores rank by canonical reference ascending.
    assert [row["image"] for row in accepted] == [f"c{i:02d}.jpg" for i in range(9)]
    by_ref = {row["image"]: row for row in diagnostics}
    assert by_ref["c09.jpg"]["rejection_reasons"] == ["class_cap"]
    assert by_ref["c10.jpg"]["rejection_reasons"] == ["class_cap"]
    assert summary["seed_count_bins"]["3"]["accepted"] == 9


def test_near_duplicate_expert_copy_rejected(tmp_path):
    root = tmp_path / "images"
    _touch(root, "a1.jpg")
    _touch(root, "a2.jpg")
    _touch(root, "a3.jpg")
    _touch(root, "copy.jpg")
    _touch(root, "b.jpg")
    (root / "copy.jpg").write_bytes((root / "a1.jpg").read_bytes())

    experts = [
        {"image": "a1.jpg", "label": "A", "_path": root / "a1.jpg"},
        {"image": "a2.jpg", "label": "A", "_path": root / "a2.jpg"},
        {"image": "a3.jpg", "label": "A", "_path": root / "a3.jpg"},
        {"image": "b.jpg", "label": "B", "_path": root / "b.jpg"},
    ]
    vector = unit(0.0)
    features = {row["image"]: vector for row in experts if row["label"] == "A"}
    features["b.jpg"] = unit(90.0)
    features["copy.jpg"] = vector

    accepted, diagnostics, summary = pl.decide_candidates(
        experts,
        [{"image": "copy.jpg", "_path": root / "copy.jpg"}],
        features,
        similarity_floor=0.5,
        min_gap=0.1,
        neighbors=3,
    )

    by_ref = {row["image"]: row for row in diagnostics}
    assert accepted == []
    assert "expert_content_duplicate" in by_ref["copy.jpg"]["rejection_reasons"]
    assert by_ref["copy.jpg"]["possible_duplicate"] is True
    assert summary["near_duplicate_candidate_count"] >= 1


def test_accepted_content_duplicate_keeps_canonical_first(tmp_path):
    root = tmp_path / "images"
    _touch(root, "a1.jpg")
    _touch(root, "a2.jpg")
    _touch(root, "a3.jpg")
    _touch(root, "z1.jpg")
    _touch(root, "z2.jpg")
    (root / "z2.jpg").write_bytes((root / "z1.jpg").read_bytes())

    experts = [
        {"image": f"a{i}.jpg", "label": "A", "_path": root / f"a{i}.jpg"}
        for i in (1, 2, 3)
    ]
    vector = unit(0.0)
    features = {row["image"]: vector for row in experts}
    # A far competitor so the candidates clear the strict gap rule.
    features["b.jpg"] = unit(90.0)
    experts.append({"image": "b.jpg", "label": "B", "_path": root / "b.jpg"})
    _touch(root, "b.jpg")
    features["z1.jpg"] = unit(1.0)
    features["z2.jpg"] = unit(1.0)

    accepted, diagnostics, _summary = pl.decide_candidates(
        experts,
        [
            {"image": "z1.jpg", "_path": root / "z1.jpg"},
            {"image": "z2.jpg", "_path": root / "z2.jpg"},
        ],
        features,
        similarity_floor=0.5,
        min_gap=0.0,
        neighbors=3,
    )

    assert [row["image"] for row in accepted] == ["z1.jpg"]
    by_ref = {row["image"]: row for row in diagnostics}
    assert by_ref["z2.jpg"]["rejection_reasons"] == ["accepted_content_duplicate"]


def test_summary_bins_sum_to_candidate_count():
    a = [_expert("A", angle) for angle in (0.0, 1.0, 2.0)]
    b = [_expert("B", angle) for angle in (90.0, 91.0, 92.0)]
    candidates = [_cand(f"c{i}.jpg", 0.5) for i in range(4)]

    _accepted, _diagnostics, summary = _decide([*a, *b], candidates, neighbors=3)

    total = sum(bucket["candidates"] for bucket in summary["seed_count_bins"].values())
    assert total == summary["candidate_count"] == 4


def test_generate_pseudo_rows_rejects_empty_pool(tmp_path):
    root = tmp_path / "images"
    _touch(root, "a.jpg")

    with pytest.raises(pl.PseudoLabelError):
        pl.generate_pseudo_rows(
            expert_rows=[{"image": "a.jpg", "label": "A"}],
            image_root=root,
            pseudo_checkpoint=tmp_path / "missing.pth",
        )


def test_extract_raw_cls_returns_unormalized_backbone_cls(tmp_path):
    from otuformer.training.model import OTUFormerEncoder

    root = tmp_path / "images"
    _touch(root, "a.jpg")
    _touch(root, "b.jpg")
    encoder = OTUFormerEncoder(
        model_name="vit_tiny_patch16_224", out_dim=8, pretrained=False, img_size=32
    )
    checkpoint = tmp_path / "ft1.pth"
    torch.save(
        {
            "model_state_dict": encoder.state_dict(),
            "config": {"model_name": "vit_tiny_patch16_224", "out_dim": 8, "image_size": 32},
        },
        checkpoint,
    )

    before = sorted(p.name for p in root.iterdir())
    features = pl.extract_raw_cls(
        checkpoint,
        ["a.jpg", "b.jpg"],
        root,
        device="cpu",
        batch_size=2,
        num_workers=0,
    )
    after = sorted(p.name for p in root.iterdir())

    assert set(features) == {"a.jpg", "b.jpg"}
    assert before == after  # never writes into the image root
    for vector in features.values():
        assert vector.shape == (encoder.backbone.num_features,)
        # Raw CLS, not an L2-normalized or projector output.
        assert not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-6)


def test_training_seed_loo_diagnostics_need_four_seeds():
    a = [_expert("A", angle) for angle in (0.0, 1.0, 2.0, 3.0)]
    b = [_expert("B", angle) for angle in (90.0, 91.0, 92.0)]
    candidate = _cand("c.jpg", 0.0)

    _accepted, _diagnostics, summary = _decide([*a, *b], [candidate])

    loo = summary["training_seed_loo"]
    assert loo["training_seed_loo_count"] == 4
    assert "optimistic" in loo["note"]
    assert loo["training_seed_loo_within_top3"]["p50"] is not None

    _accepted, _diagnostics, summary = _decide([*b], [candidate])
    assert summary["training_seed_loo"]["training_seed_loo_count"] == 0


def test_realpath_fallback_still_subtracts_experts(tmp_path, monkeypatch):
    root = tmp_path / "images"
    _touch(root, "a.jpg")
    _touch(root, "b.jpg")

    real_stat = os.stat

    def fake_stat(path, *args, **kwargs):
        stat = real_stat(path, *args, **kwargs)
        values = list(stat)
        values[1] = 0  # st_ino unavailable -> whole scan uses realpaths
        return os.stat_result(values)

    monkeypatch.setattr(os, "stat", fake_stat)

    candidates, _rejections, summary = pl.discover_candidates(
        [{"image": "a.jpg", "label": "A"}], root
    )

    assert summary["file_identity_mode"] == pl.REALPATH_MODE
    assert [c["image"] for c in candidates] == ["b.jpg"]
    assert summary["candidate_pool_sha256"]


def test_generate_pseudo_rows_reports_progress(tmp_path, monkeypatch):
    from otuformer.embedding import pseudo_label as pl

    for name in ("A0.jpg", "A1.jpg", "A2.jpg", "B0.jpg", "B1.jpg", "B2.jpg", "c.jpg"):
        (tmp_path / name).write_bytes(name.encode())
    candidate = {"image": "c.jpg", "_path": tmp_path / "c.jpg"}
    monkeypatch.setattr(
        pl,
        "discover_candidates",
        lambda expert_rows, root: (
            [candidate],
            [],
            {"discovered_candidate_count": 1},
        ),
    )
    expert_rows = [
        {"image": f"A{i}.jpg", "label": "A"} for i in range(3)
    ] + [{"image": f"B{i}.jpg", "label": "B"} for i in range(3)]
    features = {
        "A0.jpg": unit(0.0),
        "A1.jpg": unit(1.0),
        "A2.jpg": unit(2.0),
        "B0.jpg": unit(90.0),
        "B1.jpg": unit(91.0),
        "B2.jpg": unit(92.0),
        "c.jpg": unit(0.0),
    }
    monkeypatch.setattr(pl, "extract_raw_cls", lambda *a, **k: features)

    messages: list[str] = []
    accepted, _diagnostics, _summary = pl.generate_pseudo_rows(
        expert_rows=expert_rows,
        image_root=tmp_path,
        pseudo_checkpoint=tmp_path / "ft1.pth",
        progress=messages.append,
    )

    assert accepted
    assert any("scanning" in message for message in messages)
    assert any("candidates discovered" in message for message in messages)
    assert any("scoring" in message for message in messages)
    assert any("accepted" in message for message in messages)


def test_cap_multiplier_and_absolute_cap_are_configurable():
    a = [_expert("A", angle) for angle in (0.0, 0.2, 0.4)]
    b = [_expert("B", angle) for angle in (90.0, 90.2, 90.4)]
    candidates = [_cand(f"c{i:02d}.jpg", 0.0) for i in range(11)]

    # Default absolute cap 50 leaves the 3x multiplier binding: 3*3 = 9.
    accepted, _diagnostics, summary = _decide([*a, *b], candidates, neighbors=32)
    assert len(accepted) == 9
    assert summary["thresholds"]["cap"] == "min(3*seed_count, 50)"

    # An explicit absolute cap overrides the multiplier when it is smaller.
    accepted, diagnostics, summary = _decide(
        [*a, *b], candidates, neighbors=32, absolute_cap=4
    )
    assert len(accepted) == 4
    assert summary["thresholds"]["cap"] == "min(3*seed_count, 4)"
    assert sum(
        1 for row in diagnostics if "class_cap" in row["rejection_reasons"]
    ) == 7


def test_counterfactual_counts_remove_exactly_one_rule():
    """Removing one rule must report the candidates that rule alone blocked."""
    a = [_expert("A", angle) for angle in (0.0, 0.2, 0.4)]
    b = [_expert("B", angle) for angle in (90.0, 90.2, 90.4)]
    # Closer to A than to B: passes gap and mutual-kNN, fails a high floor.
    accepted, diagnostics, summary = _decide(
        [*a, *b], [_cand("c.jpg", 20.0)],
        similarity_floor=0.999, min_gap=0.1, neighbors=1,
    )

    assert accepted == []
    row = diagnostics[0]
    assert "floor" in row["rejection_reasons"]
    assert row["mutual_neighbor"] is True
    counts = summary["counterfactual_pass_counts"]
    assert counts["removed_floor"] == 1
    assert counts["removed_gap"] == 0
    assert counts["removed_mutual_knn"] == 0


def test_near_duplicate_detection_is_not_gated_by_floor_or_gap(tmp_path):
    """A copied expert still reports content duplication when floor/gap reject."""
    root = tmp_path / "images"
    for name in ("a1.jpg", "a2.jpg", "b1.jpg", "b2.jpg", "b3.jpg", "copy.jpg"):
        _touch(root, name)
    (root / "copy.jpg").write_bytes((root / "a1.jpg").read_bytes())

    experts = [
        {"image": "a1.jpg", "label": "A", "_path": root / "a1.jpg"},
        {"image": "a2.jpg", "label": "A", "_path": root / "a2.jpg"},
        {"image": "b1.jpg", "label": "B", "_path": root / "b1.jpg"},
        {"image": "b2.jpg", "label": "B", "_path": root / "b2.jpg"},
        {"image": "b3.jpg", "label": "B", "_path": root / "b3.jpg"},
    ]
    features = {
        "a1.jpg": unit(0.0),
        "a2.jpg": unit(1.0),
        "b1.jpg": unit(90.0),
        "b2.jpg": unit(91.0),
        "b3.jpg": unit(92.0),
        "copy.jpg": unit(0.0),
    }
    accepted, diagnostics, summary = pl.decide_candidates(
        experts,
        [{"image": "copy.jpg", "_path": root / "copy.jpg"}],
        features,
        similarity_floor=0.5,
        min_gap=0.1,
        neighbors=3,
    )

    # A has only two seeds, so the copy cannot win and the gap rule rejects it;
    # the >0.999 content check must still run and report the exact duplicate.
    assert accepted == []
    row = diagnostics[0]
    assert "expert_content_duplicate" in row["rejection_reasons"]
    assert row["possible_duplicate"] is True
    assert summary["near_duplicate_candidate_count"] == 1
    assert summary["near_duplicate_candidate_rate"] == 1.0


def test_similarity_ties_ignore_expert_csv_row_order():
    """Canonical references, not CSV order, break similarity ties (design 5.2)."""
    features = {
        "a.jpg": unit(0.0),
        "m.jpg": unit(0.5),
        "z.jpg": unit(0.0),
        "b1.jpg": unit(90.0),
        "b2.jpg": unit(91.0),
        "b3.jpg": unit(92.0),
        "c.jpg": unit(0.0),
    }
    others = [{"image": f"b{i}.jpg", "label": "B"} for i in (1, 2, 3)]
    first = [{"image": "a.jpg", "label": "A"}, {"image": "z.jpg", "label": "A"},
             {"image": "m.jpg", "label": "A"}]
    second = list(reversed(first))

    neighbors = []
    for experts in (first + others, second + others):
        _accepted, diagnostics, _summary = pl.decide_candidates(
            experts,
            [{"image": "c.jpg"}],
            features,
            similarity_floor=0.5,
            min_gap=0.0,
            neighbors=1,
        )
        neighbors.append(diagnostics[0]["candidate_neighbor_refs"])

    assert neighbors[0] == neighbors[1] == ["a.jpg"]


def test_ref_dataset_wraps_unreadable_image(tmp_path):
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    dataset = pl._RefDataset(tmp_path, ["bad.jpg"], 32)

    with pytest.raises(pl.PseudoLabelError, match="cannot read"):
        dataset[0]


def test_seed_side_ties_use_canonical_reference_across_pools():
    """A candidate and an other-class seed tie; the lower reference wins."""
    experts = [
        {"image": "a0.jpg", "label": "A"},
        {"image": "a1.jpg", "label": "A"},
        {"image": "a2.jpg", "label": "A"},
        {"image": "b0.jpg", "label": "B"},
    ]
    features = {
        "a0.jpg": unit(0.0),
        "a1.jpg": unit(-30.0),
        "a2.jpg": unit(-50.0),
        "b0.jpg": unit(-20.0),
        "z.jpg": unit(20.0),
    }
    accepted, diagnostics, _summary = pl.decide_candidates(
        experts,
        [{"image": "z.jpg"}],
        features,
        similarity_floor=-1.0,
        min_gap=-1.0,
        neighbors=1,
    )

    # a0 and z tie for a0's single seed-side slot; canonical "b0.jpg" < "z.jpg"
    # owns it, so z is not reciprocal and mutual-kNN is the only rejection.
    assert accepted == []
    assert diagnostics[0]["mutual_neighbor"] is False
    assert diagnostics[0]["rejection_reasons"] == ["mutual_knn"]


def test_seed_side_tie_break_handles_seed_index_zero():
    """Seed index 0 must encode distinctly from the empty-slot sentinel."""
    experts = [
        {"image": "m0.jpg", "label": "M"},
        {"image": "m1.jpg", "label": "M"},
        {"image": "m2.jpg", "label": "M"},
        {"image": "a0.jpg", "label": "B"},
    ]
    features = {
        "m0.jpg": unit(0.0),
        "m1.jpg": unit(-30.0),
        "m2.jpg": unit(-50.0),
        "a0.jpg": unit(-20.0),
        "z.jpg": unit(20.0),
    }
    accepted, diagnostics, _summary = pl.decide_candidates(
        experts,
        [{"image": "z.jpg"}],
        features,
        similarity_floor=-1.0,
        min_gap=-1.0,
        neighbors=1,
    )

    # "a0.jpg" sorts first, so it is seed index 0 and ties with z for m0's
    # single slot. Its lower reference must win, so mutual-kNN rejects z.
    assert accepted == []
    assert diagnostics[0]["mutual_neighbor"] is False
    assert diagnostics[0]["rejection_reasons"] == ["mutual_knn"]


def test_early_exit_summary_keeps_diagnostic_fields():
    # Every class has fewer than three seeds, so no class can win.
    a = [_expert("A", 0.0), _expert("A", 0.2)]
    b = [_expert("B", 90.0), _expert("B", 90.2)]
    accepted, _diagnostics, summary = _decide([*a, *b], [_cand("c.jpg", 0.0)])

    assert accepted == []
    assert summary["counterfactual_pass_counts"]["removed_floor"] == 0
    assert summary["near_duplicate_candidate_count"] == 0
    assert summary["near_duplicate_candidate_rate"] == 0.0
    assert summary["training_seed_loo"]["training_seed_loo_count"] == 0


def test_early_exit_summary_when_only_candidate_features_invalid():
    a = [_expert("A", angle) for angle in (0.0, 0.2, 0.4)]
    b = [_expert("B", angle) for angle in (90.0, 90.2, 90.4)]
    zero = ({"image": "zero.jpg"}, "zero.jpg", np.zeros(2, dtype=np.float32))
    accepted, _diagnostics, summary = _decide([*a, *b], [zero])

    assert accepted == []
    assert summary["candidate_count"] == 1
    assert "counterfactual_pass_counts" in summary
    assert "training_seed_loo" in summary


def test_near_duplicate_rate_counts_pre_scan_rejections(tmp_path, monkeypatch):
    for name in ("A0.jpg", "A1.jpg", "A2.jpg", "B0.jpg", "B1.jpg", "B2.jpg", "c.jpg"):
        (tmp_path / name).write_bytes(name.encode())
    expert_rows = [{"image": f"A{i}.jpg", "label": "A"} for i in range(3)] + [
        {"image": f"B{i}.jpg", "label": "B"} for i in range(3)
    ]
    features = {
        "A0.jpg": unit(0.0),
        "A1.jpg": unit(1.0),
        "A2.jpg": unit(2.0),
        "B0.jpg": unit(90.0),
        "B1.jpg": unit(91.0),
        "B2.jpg": unit(92.0),
        "c.jpg": unit(0.0),  # >0.999 to A0: a near-duplicate diagnostic
    }
    monkeypatch.setattr(
        pl,
        "discover_candidates",
        lambda expert_rows, root: (
            [{"image": "c.jpg", "_path": tmp_path / "c.jpg"}],
            [{"image": "alias.jpg", "rejection_reasons": ["duplicate_file_identity"]}],
            {
                "discovered_candidate_count": 1,
                "candidate_pool_sha256": "x",
                "weak_signal_warnings": [],
            },
        ),
    )
    monkeypatch.setattr(pl, "extract_raw_cls", lambda *a, **k: features)

    _accepted, _diagnostics, summary = pl.generate_pseudo_rows(
        expert_rows=expert_rows,
        image_root=tmp_path,
        pseudo_checkpoint=tmp_path / "ft1.pth",
    )

    assert summary["candidate_count"] == 2
    assert summary["near_duplicate_candidate_count"] == 1
    assert summary["near_duplicate_candidate_rate"] == 0.5


def test_weak_output_signal_is_surfaced_as_warning(tmp_path, monkeypatch):
    for name in ("A0.jpg", "A1.jpg", "A2.jpg", "B0.jpg", "B1.jpg", "B2.jpg", "c.jpg"):
        (tmp_path / name).write_bytes(name.encode())
    expert_rows = [{"image": f"A{i}.jpg", "label": "A"} for i in range(3)] + [
        {"image": f"B{i}.jpg", "label": "B"} for i in range(3)
    ]
    features = {
        "A0.jpg": unit(0.0),
        "A1.jpg": unit(1.0),
        "A2.jpg": unit(2.0),
        "B0.jpg": unit(90.0),
        "B1.jpg": unit(91.0),
        "B2.jpg": unit(92.0),
        "c.jpg": unit(0.0),
    }
    monkeypatch.setattr(
        pl,
        "discover_candidates",
        lambda expert_rows, root: (
            [{"image": "c.jpg", "_path": tmp_path / "c.jpg"}],
            [],
            {
                "discovered_candidate_count": 1,
                "candidate_pool_sha256": "x",
                "weak_signal_warnings": ["runs/weak"],
            },
        ),
    )
    monkeypatch.setattr(pl, "extract_raw_cls", lambda *a, **k: features)
    messages: list[str] = []

    pl.generate_pseudo_rows(
        expert_rows=expert_rows,
        image_root=tmp_path,
        pseudo_checkpoint=tmp_path / "ft1.pth",
        progress=messages.append,
    )

    assert any("weak run-output signals" in message for message in messages)
    assert any("runs/weak" in message for message in messages)
