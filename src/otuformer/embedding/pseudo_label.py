"""Sparse-label pseudo-label generation: discovery, raw-CLS scoring, diagnostics.

The library boundary is strict: :func:`generate_pseudo_rows` returns accepted
rows, all diagnostic rows, and a summary, and never prints, creates
directories, or writes files. The CLI prints the empty-set diagnostics and
exits. Only the accepted ``known-pseudo`` rows may enter finetune#2 training.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from otuformer.embedding.extractor import IMAGE_EXTENSIONS, _load_model
from otuformer.training.dataset import build_eval_transform
from otuformer.utils.device import resolve_device
from otuformer.utils.paths import INODE_MODE, REALPATH_MODE, realpath_key


class PseudoLabelError(ValueError):
    """Lightweight failure for invalid sources, identities, or feature bytes."""


ELIGIBLE_MIN_SEEDS = 3
NEAR_DUPLICATE_SIMILARITY = 0.999
EVAL_TRANSFORM_NAME = "center-crop"
CAP_MULTIPLIER = 3
ABSOLUTE_CAP = 50
# Candidate block size for the streaming similarity pass. ponytail: dense
# per-block argsort instead of a heap-based top-k, and a per-candidate Python
# loop for diagnostics; revisit only if profiling on 100k candidates shows the
# cost matters.
CANDIDATE_BLOCK = 4096


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def _match_output_signature(path: Path) -> str | None:
    """Return the signature name when ``path`` is a strong output subtree."""
    logs = path / "logs"
    if not logs.is_dir():
        return None
    if (logs / "finetune.log").is_file() and (path / "finetune_latest.pth").is_file():
        return "finetune"
    if (logs / "pretrain.log").is_file() and (path / "SSL_latest.pth").is_file():
        return "pretrain"
    if (logs / "extract.log").is_file() and any(
        (path / name).exists()
        for name in ("embeddings.csv", "metrics.csv", "umap.pdf")
    ):
        return "extract"
    if (logs / "cam.log").is_file() and (
        (path / "cam_summary.csv").exists() or (path / "figures").is_dir()
    ):
        return "cam"
    if (logs / "annotate.log").is_file() and any(
        (path / name).exists()
        for name in (
            "annotation_summary.json",
            "otu_table.csv",
            "UPGMA_tree_partitions_annotated.pdf",
        )
    ):
        return "annotate"
    return None


def _collect_entries(root: Path):
    """Collect supported image entries plus traversal diagnostics.

    Directory symlinks are never followed (they are counted and reported); the
    traversal is our own so the behaviour does not depend on ``pathlib``
    version semantics.
    """
    entries: list[dict] = []
    skipped_dirs: list[str] = []
    skipped_outputs: list[str] = []
    weak_warnings: list[str] = []

    def visit(directory: Path) -> None:
        label = _match_output_signature(directory)
        if label is not None:
            skipped_outputs.append(f"{directory.relative_to(root).as_posix()} ({label})")
            return
        top_names = set(os.listdir(directory))
        if directory != root and (
            "logs" in top_names
            or any(name.endswith(".pth") for name in top_names)
        ):
            weak_warnings.append(directory.relative_to(root).as_posix())

        subdirs: list[Path] = []
        for entry in os.scandir(directory):
            child = Path(entry.path)
            if entry.is_symlink():
                if child.is_dir():
                    skipped_dirs.append(child.relative_to(root).as_posix())
                    continue
                if not child.is_file():
                    continue
                if child.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                entries.append({"path": child, "dir": directory})
                continue
            if entry.is_dir(follow_symlinks=False):
                subdirs.append(child)
                continue
            if entry.is_file(follow_symlinks=False) and (
                child.suffix.lower() in IMAGE_EXTENSIONS
            ):
                entries.append({"path": child, "dir": directory})
        for sub in sorted(subdirs, key=lambda p: p.name):
            visit(sub)

    visit(root)
    return entries, skipped_dirs, skipped_outputs, weak_warnings


def _entry_ref(entry: dict, root: Path) -> str:
    """Portable canonical reference: root-relative link path, never target."""
    link = entry["path"]
    rel = link.relative_to(entry["dir"]).as_posix()
    parent = entry["dir"].relative_to(root).as_posix()
    return rel if parent in ("", ".") else f"{parent}/{rel}"


def discover_candidates(
    expert_rows: list[dict], image_root: Path
) -> tuple[list[dict], list[dict], dict]:
    """Discover candidate images under ``image_root`` minus expert identities."""
    root = Path(image_root).resolve()
    if not root.is_dir():
        raise PseudoLabelError(f"image root is not a directory: {image_root}")

    # Validate expert references first, but compute their identities only after
    # the scan-wide identity mode is known; otherwise an inode->realpath fallback
    # would fail to subtract the experts themselves.
    expert_refs: list[tuple[str, Path]] = []
    for row in expert_rows:
        ref = _canonical_ref(row["image"], root)
        resolved = (root / ref).resolve()
        if resolved != root and root not in resolved.parents:
            raise PseudoLabelError(f"expert reference escapes the image root: {ref!r}")
        if not resolved.is_file():
            raise PseudoLabelError(f"expert image is missing: {ref!r}")
        expert_refs.append((ref, resolved))

    entries, skipped_dirs, skipped_outputs, weak_warnings = _collect_entries(root)

    stats = [entry["path"].stat() for entry in entries]
    mode = INODE_MODE
    collision_count = 0
    if any(stat.st_ino == 0 for stat in stats):
        mode = REALPATH_MODE
    else:
        groups: dict[tuple[int, int], list[dict]] = {}
        for entry, stat in zip(entries, stats):
            groups.setdefault((stat.st_dev, stat.st_ino), []).append(entry)
        for members in groups.values():
            if len({realpath_key(m["path"].resolve()) for m in members}) > 1 and len(
                {_sha256_file(m["path"]) for m in members}
            ) > 1:
                # Same inode, different real paths, different bytes: the inode
                # metadata is unreliable, so the whole scan uses realpaths.
                mode = REALPATH_MODE
                collision_count += 1

    expert_identities: dict[object, str] = {}
    for ref, resolved in expert_refs:
        identity = _scan_identity(resolved, mode)
        if identity in expert_identities:
            raise PseudoLabelError(f"duplicate expert path identity: {ref!r}")
        expert_identities[identity] = ref

    candidates: list[dict] = []
    rejections: list[dict] = []
    seen: dict[object, str] = {}
    for entry in sorted(entries, key=lambda e: _entry_ref(e, root)):
        ref = _entry_ref(entry, root)
        resolved = entry["path"].resolve()
        if resolved != root and root not in resolved.parents:
            rejections.append({"image": ref, "rejection_reasons": ["escapes_image_root"]})
            continue
        identity = _scan_identity(resolved, mode)
        if identity in expert_identities:
            continue
        if identity in seen:
            rejections.append(
                {"image": ref, "rejection_reasons": ["duplicate_file_identity"]}
            )
            continue
        seen[identity] = ref
        candidates.append({"image": ref, "_path": resolved})

    summary = {
        "file_identity_mode": mode,
        "inode_collision_count": collision_count,
        "skipped_directory_symlinks": skipped_dirs[:10],
        "skipped_directory_symlink_count": len(skipped_dirs),
        "skipped_output_subtrees": skipped_outputs[:10],
        "skipped_output_subtree_count": len(skipped_outputs),
        "weak_signal_warnings": weak_warnings[:10],
        "discovered_candidate_count": len(candidates),
        # Record-only: in normalized-realpath mode hard links are distinct files
        # by design, so this hash is provenance, never a resume gate.
        "candidate_pool_sha256": _refs_sha256(
            sorted({c["image"] for c in candidates} | {r["image"] for r in rejections})
        ),
    }
    return candidates, rejections, summary


def _canonical_ref(ref: object, root: Path) -> str:
    from otuformer.utils.paths import canonical_image_ref

    return canonical_image_ref(ref, root)


def _scan_identity(resolved: Path, mode: str) -> object:
    if mode == INODE_MODE:
        try:
            stat = resolved.stat()
        except OSError:
            stat = None
        if stat is not None and stat.st_ino:
            return ("inode", stat.st_dev, stat.st_ino)
    return ("realpath", realpath_key(resolved))


# --------------------------------------------------------------------------
# In-memory raw-CLS extraction
# --------------------------------------------------------------------------


class _RefDataset(Dataset):
    def __init__(self, root: Path, refs: list[str], size: int) -> None:
        self.items = [(ref, root / ref) for ref in refs]
        self.transform = build_eval_transform(EVAL_TRANSFORM_NAME, size)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        _ref, path = self.items[index]
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
        except Exception as exc:  # noqa: BLE001 - one lightweight error type
            raise PseudoLabelError(
                f"cannot read candidate image {_ref!r}: {exc}"
            ) from exc
        return tensor, index


def extract_raw_cls(
    checkpoint: Path,
    refs: list[str],
    image_root: Path,
    *,
    model_name: str = "vit_tiny_patch16_224",
    device: str = "auto",
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict[str, np.ndarray]:
    """Return ``{canonical_ref: raw pre-L2 CLS vector}`` using the checkpoint size."""
    target = resolve_device(device)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model, size = _load_model(Path(checkpoint), model_name, target)
    except PseudoLabelError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced as one lightweight error
        raise PseudoLabelError(f"cannot load pseudo source checkpoint: {exc}") from exc

    dataset = _RefDataset(Path(image_root), refs, size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    out: dict[str, np.ndarray] = {}
    try:
        with torch.no_grad():
            for images, indices in loader:
                images = images.to(target)
                feats = model.backbone.forward_features(images)
                if isinstance(feats, dict):
                    feats = feats["x"]
                cls = feats[:, 0].detach().cpu().numpy()
                for row, index in zip(cls, indices.tolist()):
                    out[refs[index]] = np.asarray(row, dtype=np.float32)
    except PseudoLabelError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced as one lightweight error
        raise PseudoLabelError(f"raw CLS extraction failed: {exc}") from exc
    if len(out) != len(refs):
        raise PseudoLabelError("raw CLS extraction returned an incomplete result")
    return out


# --------------------------------------------------------------------------
# Decision rule
# --------------------------------------------------------------------------


def _normalize_features(refs, features, *, on_invalid: str):
    kept: list[str] = []
    vectors: list[np.ndarray] = []
    invalid: set[str] = set()
    for ref in refs:
        vector = np.asarray(features.get(ref, []), dtype=np.float64)
        norm = np.linalg.norm(vector)
        if vector.size == 0 or not np.all(np.isfinite(vector)) or norm == 0.0:
            invalid.add(ref)
            if on_invalid == "raise":
                raise PseudoLabelError(f"invalid feature vector for seed {ref!r}")
            continue
        kept.append(ref)
        vectors.append(vector / norm)
    matrix = np.vstack(vectors) if vectors else np.zeros((0, 0))
    return matrix, kept, invalid


def _top_k_mean(matrix: np.ndarray, k: int) -> np.ndarray:
    width = min(k, matrix.shape[1])
    if width == 0:
        return np.full(matrix.shape[0], -np.inf)
    return np.sort(matrix, axis=1)[:, -width:].mean(axis=1)


def _class_scores(sims, class_groups, eligible):
    scores = np.empty((sims.shape[0], len(class_groups)), dtype=np.float64)
    for index, columns in enumerate(class_groups):
        block = sims[:, columns]
        scores[:, index] = (
            _top_k_mean(block, 3) if eligible[index] else block.max(axis=1)
        )
    return scores


def _strict_topk(sims: np.ndarray, k: int) -> np.ndarray:
    """Top-``k`` column indices per row; ties keep column order (deterministic)."""
    width = min(k, sims.shape[1])
    return np.argsort(-sims, axis=1, kind="stable")[:, :width]


def _seed_count_bin(count: int | None) -> str:
    if count is None:
        return "none"
    if count <= 3:
        return "3"
    if count <= 5:
        return "4-5"
    if count <= 10:
        return "6-10"
    return ">10"


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None, "p99": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
    }


def _new_diagnostic(ref: str, reasons: list[str] | None = None) -> dict:
    return {
        "image": ref,
        "accepted": False,
        "pseudo_label": None,
        "top1_score": None,
        "max_seed_similarity": None,
        "runner_up_class": None,
        "runner_up_score": None,
        "score_gap": None,
        "candidate_neighbor_refs": [],
        "mutual_neighbor": False,
        "possible_duplicate": False,
        "class_cap_rank": None,
        "rejection_reasons": list(reasons or []),
        "_winner_class": None,
    }


def decide_candidates(
    expert_rows: list[dict],
    candidate_rows: list[dict],
    features: dict[str, np.ndarray],
    *,
    similarity_floor: float,
    min_gap: float,
    neighbors: int,
    cap_multiplier: int = CAP_MULTIPLIER,
    absolute_cap: int = ABSOLUTE_CAP,
) -> tuple[list[dict], list[dict], dict]:
    """Apply the fixed precision-oriented rule to every candidate.

    ``expert_rows`` and ``candidate_rows`` must already carry root-canonical
    references; candidate rows may carry a resolved ``_path`` used only for the
    gated near-duplicate comparison.
    """
    # Canonical-reference order makes candidate input order irrelevant and the
    # stable sorts below break candidate ties by reference (design 5.2).
    candidate_rows = sorted(candidate_rows, key=lambda row: str(row["image"]))
    class_labels = sorted({str(row["label"]) for row in expert_rows})
    seeds_by_class: dict[str, list[str]] = {label: [] for label in class_labels}
    for row in expert_rows:
        seeds_by_class[str(row["label"])].append(str(row["image"]))
    # Canonical-reference order within a class is the deterministic secondary
    # key for similarity ties (design 5.2); CSV row order must not matter.
    for label in class_labels:
        seeds_by_class[label] = sorted(seeds_by_class[label])
    # Global canonical-reference order so similarity ties break by reference
    # across classes, not by class grouping or CSV row order (design 5.2).
    seed_refs = sorted(ref for label in class_labels for ref in seeds_by_class[label])
    if not seed_refs:
        raise PseudoLabelError("no expert seeds for scoring")
    expert_vectors, seed_refs, _ = _normalize_features(
        seed_refs, features, on_invalid="raise"
    )

    candidate_refs = [str(row["image"]) for row in candidate_rows]
    path_by_ref = {
        str(row["image"]): row.get("_path") for row in candidate_rows
    }
    cand_vectors, valid_refs, invalid_refs = _normalize_features(
        candidate_refs, features, on_invalid="reject"
    )
    valid_index = {ref: i for i, ref in enumerate(valid_refs)}

    class_groups = [
        np.array([seed_refs.index(ref) for ref in seeds_by_class[label]], dtype=int)
        for label in class_labels
    ]
    eligible = [len(seeds_by_class[label]) >= ELIGIBLE_MIN_SEEDS for label in class_labels]
    eligible_columns = [i for i, flag in enumerate(eligible) if flag]

    diagnostics = {ref: _new_diagnostic(ref) for ref in candidate_refs}
    for ref in invalid_refs:
        diagnostics[ref]["rejection_reasons"].append("invalid_feature")

    # Leave-one-out diagnostics need only the expert pool, so they are computed
    # even on the early-exit paths where no candidate can be scored.
    loo = _training_seed_loo(expert_vectors, class_groups, eligible)
    empty_extra = {
        "removed_floor": 0,
        "removed_gap": 0,
        "removed_mutual": 0,
        "near_duplicate_candidates": 0,
        "expert_content_duplicate_rejections": 0,
        "training_seed_loo": loo,
    }

    def finish() -> tuple[list[dict], list[dict], dict]:
        rows = [diagnostics[ref] for ref in sorted(diagnostics)]
        for row in rows:
            row.pop("_winner_class", None)
        summary = _summarize(
            class_labels=class_labels,
            seeds_by_class=seeds_by_class,
            eligible=eligible,
            diagnostics=diagnostics,
            similarity_floor=similarity_floor,
            min_gap=min_gap,
            neighbors=neighbors,
            cap_multiplier=cap_multiplier,
            absolute_cap=absolute_cap,
            extra=empty_extra,
        )
        return [], rows, summary

    if cand_vectors.shape[0] == 0 or not eligible_columns:
        for ref in valid_refs:
            diagnostics[ref]["rejection_reasons"].append("no_eligible_class")
        return finish()

    # Canonical-reference rank per pool id, used as the secondary sort key so
    # ties never depend on column placement or input order (design 5.2).
    ref_rank = {
        ref: index
        for index, ref in enumerate(sorted(set(seed_refs) | set(valid_refs)))
    }
    candidate_ref_ranks = np.array(
        [ref_rank[ref] for ref in valid_refs], dtype=np.float64
    )
    seed_ref_ranks = np.array([ref_rank[ref] for ref in seed_refs], dtype=np.float64)

    def _id_ranks(ids: np.ndarray) -> np.ndarray:
        # Padding is always -1. Candidate ids are >= 0; encoded seeds are
        # -("index" + 2) <= -2, so seed index 0 never collides with padding.
        ranks = np.full(ids.shape, np.inf, dtype=np.float64)
        valid = ids != -1
        positive = valid & (ids >= 0)
        negative = valid & (ids < 0)
        if positive.any():
            ranks[positive] = candidate_ref_ranks[ids[positive]]
        if negative.any():
            ranks[negative] = seed_ref_ranks[-ids[negative] - 2]
        return ranks

    n_valid = cand_vectors.shape[0]
    n_seeds = expert_vectors.shape[0]
    winner = np.full(n_valid, -1, dtype=int)
    top1 = np.full(n_valid, -np.inf)
    max_sim = np.full(n_valid, -np.inf)
    gap = np.full(n_valid, np.nan)
    runner = np.full(n_valid, -1, dtype=int)
    cand_top = np.full((n_valid, min(neighbors, n_seeds)), -1, dtype=int)

    seed_seed = expert_vectors @ expert_vectors.T
    own_class = np.zeros((n_seeds, n_seeds), dtype=bool)
    for columns in class_groups:
        own_class[np.ix_(columns, columns)] = True
    seed_other = np.where(own_class, -np.inf, seed_seed)
    seed_side_ids = _strict_topk(seed_other, neighbors)
    seed_side_vals = np.take_along_axis(seed_other, seed_side_ids, axis=1)

    running_vals = np.full((n_seeds, neighbors), -np.inf)
    running_ids = np.full((n_seeds, neighbors), -1, dtype=int)  # candidate index

    for start in range(0, n_valid, CANDIDATE_BLOCK):
        stop = min(start + CANDIDATE_BLOCK, n_valid)
        sims = cand_vectors[start:stop] @ expert_vectors.T
        scores = _class_scores(sims, class_groups, eligible)
        order = np.argsort(-scores, axis=1, kind="stable")
        block_top = _strict_topk(sims, neighbors)
        for row in range(sims.shape[0]):
            global_index = start + row
            eligible_scores = scores[row, eligible_columns]
            winner_column = eligible_columns[int(np.argmax(eligible_scores))]
            winner[global_index] = winner_column
            top1[global_index] = scores[row, winner_column]
            max_sim[global_index] = sims[row].max()
            others = [c for c in order[row].tolist() if c != winner_column]
            if others:
                runner[global_index] = others[0]
                gap[global_index] = top1[global_index] - scores[row, others[0]]
            cand_top[global_index, : block_top.shape[1]] = block_top[row]
        merged_vals = np.concatenate([running_vals, sims.T], axis=1)
        merged_ids = np.concatenate(
            [running_ids, np.arange(start, stop)[None, :].repeat(n_seeds, axis=0)],
            axis=1,
        )
        keep = np.argsort(-merged_vals, axis=1, kind="stable")[:, :neighbors]
        running_vals = np.take_along_axis(merged_vals, keep, axis=1)
        running_ids = np.take_along_axis(merged_ids, keep, axis=1)

    # Seed-side top-k over candidates plus other-class seeds; encode seeds as
    # ``-(index + 2)`` so index 0 is -2 and -1 stays the empty-slot sentinel.
    seed_side_encoded = -(seed_side_ids + 2)
    final_vals = np.concatenate([running_vals, seed_side_vals], axis=1)
    final_ids = np.concatenate([running_ids, seed_side_encoded], axis=1)
    keep = np.lexsort((_id_ranks(final_ids), -final_vals), axis=1)[:, :neighbors]
    final_ids = np.take_along_axis(final_ids, keep, axis=1)
    final_vals = np.take_along_axis(final_vals, keep, axis=1)
    seed_accepts = [
        {
            int(index)
            for index, value in zip(final_ids[seed].tolist(), final_vals[seed].tolist())
            if index >= 0 and value > -np.inf
        }
        for seed in range(n_seeds)
    ]

    expert_paths = {
        str(row["image"]): row.get("_path")
        for row in expert_rows
    }
    cap_candidates: dict[int, list[tuple[float, str]]] = {
        column: [] for column in eligible_columns
    }
    removed_floor = removed_gap = removed_mutual = 0
    near_duplicate = exact_content = 0
    for global_index, ref in enumerate(valid_refs):
        info = diagnostics[ref]
        if info["rejection_reasons"]:
            continue
        winner_column = int(winner[global_index])
        winner_class = class_labels[winner_column]
        info["_winner_class"] = winner_class
        info["top1_score"] = float(top1[global_index])
        info["max_seed_similarity"] = float(max_sim[global_index])
        if int(runner[global_index]) >= 0:
            runner_column = int(runner[global_index])
            info["runner_up_class"] = class_labels[runner_column]
            info["runner_up_score"] = float(
                top1[global_index] - gap[global_index]
            )
        info["candidate_neighbor_refs"] = [
            seed_refs[c] for c in cand_top[global_index].tolist() if c >= 0
        ]
        info["score_gap"] = (
            float(gap[global_index]) if not np.isnan(gap[global_index]) else None
        )

        passes_floor = info["top1_score"] > similarity_floor
        passes_gap = info["score_gap"] is not None and info["score_gap"] > min_gap

        # Near-duplicate content verification is a diagnostic safety net and
        # applies to every candidate above the similarity threshold, before
        # floor/gap accept or reject it (design section 4).
        if float(max_sim[global_index]) > NEAR_DUPLICATE_SIMILARITY:
            near_duplicate += 1
            candidate_path = path_by_ref.get(ref)
            if candidate_path is not None:
                digest = _sha256_file(candidate_path)
                for seed_index in np.nonzero(
                    (cand_vectors[global_index] @ expert_vectors.T)
                    > NEAR_DUPLICATE_SIMILARITY
                )[0].tolist():
                    seed_path = expert_paths.get(seed_refs[seed_index])
                    if seed_path is not None and _sha256_file(seed_path) == digest:
                        info["possible_duplicate"] = True
                        info["rejection_reasons"].append("expert_content_duplicate")
                        exact_content += 1
                        break
                else:
                    info["possible_duplicate"] = True

        if int(runner[global_index]) < 0:
            info["rejection_reasons"].append("no_runner_up")
            continue
        if not passes_floor:
            info["rejection_reasons"].append("floor")
        if not passes_gap:
            info["rejection_reasons"].append(
                "gap_eligible"
                if info["runner_up_class"]
                and eligible[class_labels.index(info["runner_up_class"])]
                else "gap_undersupported"
            )

        winner_seeds = seeds_by_class[winner_class]
        mutual = any(
            seed_ref in info["candidate_neighbor_refs"]
            and global_index in seed_accepts[seed_refs.index(seed_ref)]
            for seed_ref in winner_seeds
        )
        info["mutual_neighbor"] = bool(mutual)

        # Pre-cap counterfactuals: count candidates that would clear every
        # other rule if exactly one rule were removed (design section 5.3).
        # Exact content duplicates are a separate rejection rule, so they do
        # not count as "would pass all other rules".
        content_duplicate = "expert_content_duplicate" in info["rejection_reasons"]
        if passes_gap and mutual and not content_duplicate:
            removed_floor += 1
        if passes_floor and mutual and not content_duplicate:
            removed_gap += 1
        if passes_floor and passes_gap and not content_duplicate:
            removed_mutual += 1

        if not mutual:
            info["rejection_reasons"].append("mutual_knn")
            continue
        if content_duplicate:
            continue
        if not (passes_floor and passes_gap):
            continue
        cap_candidates[winner_column].append((info["top1_score"], ref))

    accepted: list[dict] = []
    for column, entries in cap_candidates.items():
        label = class_labels[column]
        entries.sort(key=lambda item: (-item[0], item[1]))
        limit = min(cap_multiplier * len(seeds_by_class[label]), absolute_cap)
        for rank, (score, ref) in enumerate(entries):
            info = diagnostics[ref]
            if rank >= limit:
                info["rejection_reasons"].append("class_cap")
                continue
            info["class_cap_rank"] = rank
            info["accepted"] = True
            info["pseudo_label"] = label
            accepted.append(
                {
                    "image": ref,
                    "label": label,
                    "source": "known-pseudo",
                    "class_cap_rank": rank,
                    "top1_score": score,
                    "_path": path_by_ref.get(ref),
                }
            )

    accepted = _drop_accepted_content_duplicates(accepted, diagnostics)

    summary = _summarize(
        class_labels=class_labels,
        seeds_by_class=seeds_by_class,
        eligible=eligible,
        diagnostics=diagnostics,
        similarity_floor=similarity_floor,
        min_gap=min_gap,
        neighbors=neighbors,
        cap_multiplier=cap_multiplier,
        absolute_cap=absolute_cap,
        extra={
            "removed_floor": removed_floor,
            "removed_gap": removed_gap,
            "removed_mutual": removed_mutual,
            "near_duplicate_candidates": near_duplicate,
            "expert_content_duplicate_rejections": exact_content,
            "training_seed_loo": loo,
        },
    )
    return accepted, [diagnostics[ref] for ref in sorted(diagnostics)], summary


def _training_seed_loo(expert_vectors, class_groups, eligible) -> dict:
    """Optimistic leave-one-out diagnostics for classes with >= 4 seeds.

    Never calibration: ArcFace has already pulled these classes together, so
    the values are useful only beside candidate top1/gap quantiles.
    """
    sims = expert_vectors @ expert_vectors.T
    within: list[float] = []
    runner_up: list[float] = []
    gaps: list[float] = []
    for class_index, columns in enumerate(class_groups):
        if len(columns) < 4:
            continue
        for column in columns.tolist():
            same = [c for c in columns.tolist() if c != column]
            within_score = float(
                _top_k_mean(sims[column : column + 1, same], 3)[0]
            )
            best = -np.inf
            for other_index, other in enumerate(class_groups):
                if other_index == class_index:
                    continue
                if eligible[other_index]:
                    value = float(
                        _top_k_mean(sims[column : column + 1, other], 3)[0]
                    )
                else:
                    value = float(sims[column, other].max())
                best = max(best, value)
            if best > -np.inf:
                within.append(within_score)
                runner_up.append(float(best))
                gaps.append(within_score - float(best))
    return {
        "training_seed_loo_within_top3": _quantiles(within),
        "training_seed_loo_runner_up": _quantiles(runner_up),
        "training_seed_loo_gap": _quantiles(gaps),
        "training_seed_loo_count": len(within),
        "note": "optimistic training-sample diagnostics; never calibration",
    }


def _drop_accepted_content_duplicates(
    accepted: list[dict], diagnostics: dict[str, dict]
) -> list[dict]:
    """Keep the canonical-reference-first row per content hash; no cap refill."""
    kept: list[dict] = []
    seen: dict[str, str] = {}
    for row in sorted(accepted, key=lambda item: item["image"]):
        path = row.get("_path")
        digest = _sha256_file(path) if path is not None else None
        if digest is not None and digest in seen:
            info = diagnostics[row["image"]]
            info["accepted"] = False
            info["pseudo_label"] = None
            info["rejection_reasons"].append("accepted_content_duplicate")
            continue
        if digest is not None:
            seen[digest] = row["image"]
        kept.append(row)
    return kept


def _summarize(
    *,
    class_labels: list[str],
    seeds_by_class: dict[str, list[str]],
    eligible: list[bool],
    diagnostics: dict[str, dict],
    similarity_floor: float,
    min_gap: float,
    neighbors: int,
    cap_multiplier: int = CAP_MULTIPLIER,
    absolute_cap: int = ABSOLUTE_CAP,
    extra: dict | None = None,
) -> dict:
    total = len(diagnostics)
    accepted = sum(1 for info in diagnostics.values() if info["accepted"])
    bins = {name: [0, 0] for name in ("none", "3", "4-5", "6-10", ">10")}
    for info in diagnostics.values():
        if info["accepted"] and info["pseudo_label"] is not None:
            bucket = _seed_count_bin(len(seeds_by_class[info["pseudo_label"]]))
        else:
            winner = info.get("_winner_class")
            bucket = _seed_count_bin(
                len(seeds_by_class[winner]) if winner else None
            )
        bins[bucket][0] += 1
        if info["accepted"]:
            bins[bucket][1] += 1
    summary = {
        "candidate_count": total,
        "accepted_count": accepted,
        "rejected_count": total - accepted,
        "acceptance_rate": (accepted / total) if total else 0.0,
        "seed_count_bins": {
            name: {
                "candidates": counts[0],
                "accepted": counts[1],
                "acceptance_rate": (counts[1] / counts[0]) if counts[0] else 0.0,
            }
            for name, counts in bins.items()
        },
        "thresholds": {
            "similarity_floor": similarity_floor,
            "min_gap": min_gap,
            "neighbors": neighbors,
            "eval_transform": EVAL_TRANSFORM_NAME,
            "cap": f"min({cap_multiplier}*seed_count, {absolute_cap})",
        },
        "score_quantiles": {
            "top1": _quantiles(
                [i["top1_score"] for i in diagnostics.values() if i["top1_score"] is not None]
            ),
            "gap": _quantiles(
                [i["score_gap"] for i in diagnostics.values() if i["score_gap"] is not None]
            ),
        },
        "class_summary": {
            label: {"seed_count": len(seeds_by_class[label]), "eligible": flag}
            for label, flag in zip(class_labels, eligible)
        },
        "eligible_class_count": sum(1 for flag in eligible if flag),
        "ineligible_class_count": sum(1 for flag in eligible if not flag),
        "ineligible_seed_count": sum(
            len(seeds_by_class[label])
            for label, flag in zip(class_labels, eligible)
            if not flag
        ),
    }
    if extra:
        summary["counterfactual_pass_counts"] = {
            "removed_floor": extra["removed_floor"],
            "removed_gap": extra["removed_gap"],
            "removed_mutual_knn": extra["removed_mutual"],
            "note": (
                "pre-cap counts that would pass every other rule if exactly "
                "one rule were removed"
            ),
        }
        summary["near_duplicate_candidate_count"] = extra["near_duplicate_candidates"]
        summary["near_duplicate_candidate_rate"] = (
            extra["near_duplicate_candidates"] / total if total else 0.0
        )
        summary["expert_content_duplicate_rejections"] = extra[
            "expert_content_duplicate_rejections"
        ]
        summary["training_seed_loo"] = extra["training_seed_loo"]
    return summary


def generate_pseudo_rows(
    *,
    expert_rows: list[dict],
    image_root: Path,
    pseudo_checkpoint: Path,
    similarity_floor: float = 0.75,
    min_gap: float = 0.10,
    neighbors: int = 15,
    cap_multiplier: int = CAP_MULTIPLIER,
    absolute_cap: int = ABSOLUTE_CAP,
    model_name: str = "vit_tiny_patch16_224",
    device: str = "auto",
    batch_size: int = 32,
    num_workers: int = 4,
    progress=None,
) -> tuple[list[dict], list[dict], dict]:
    """Discover candidates, score raw CLS, and return accepted/diagnostic rows.

    ``progress`` is an optional caller-side callback; the library itself still
    never prints, creates directories, or writes files.
    """
    def _emit(message: str) -> None:
        if progress is not None:
            progress(message)

    root = Path(image_root)
    _emit(f"[Info] Pseudo round: scanning {root} for candidate images ...")
    candidates, rejections, discovery = discover_candidates(expert_rows, root)
    for weak in discovery.get("weak_signal_warnings", []):
        _emit(
            f"[Warning] Pseudo scan: '{weak}' has weak run-output signals "
            "(logs/ or .pth); its images remain candidates. Review it."
        )
    if not candidates:
        raise PseudoLabelError("no candidate images discovered under the image root")
    _emit(
        f"[Info] Pseudo round: {len(candidates)} candidates discovered; "
        f"extracting raw CLS from {pseudo_checkpoint} ..."
    )

    normalized_expert = [
        {
            "image": _canonical_ref(row["image"], root),
            "label": str(row["label"]),
            "_path": root / _canonical_ref(row["image"], root),
        }
        for row in expert_rows
    ]
    refs = list(
        dict.fromkeys(
            [row["image"] for row in normalized_expert]
            + [c["image"] for c in candidates]
        )
    )
    features = extract_raw_cls(
        pseudo_checkpoint,
        refs,
        root,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    _emit(
        f"[Info] Pseudo round: scoring {len(candidates)} candidates "
        f"(floor={similarity_floor}, gap={min_gap}, k={neighbors}) ..."
    )
    accepted, diagnostics, summary = decide_candidates(
        normalized_expert,
        candidates,
        features,
        similarity_floor=similarity_floor,
        min_gap=min_gap,
        neighbors=neighbors,
        cap_multiplier=cap_multiplier,
        absolute_cap=absolute_cap,
    )
    for row in accepted:
        row.pop("_path", None)
    for row in diagnostics:
        row.pop("_winner_class", None)
    existing = {row["image"] for row in diagnostics}
    appended = 0
    for row in rejections:
        if row["image"] not in existing:
            diagnostics.append(_new_diagnostic(row["image"], row["rejection_reasons"]))
            appended += 1
    diagnostics.sort(key=lambda row: row["image"])
    summary.update(discovery)
    if appended:
        # Pre-scan rejections are candidates too: keep counts and the `none` bin
        # consistent with the complete diagnostic list.
        summary["candidate_count"] += appended
        summary["rejected_count"] += appended
        summary["acceptance_rate"] = (
            summary["accepted_count"] / summary["candidate_count"]
            if summary["candidate_count"]
            else 0.0
        )
        none_bin = summary["seed_count_bins"]["none"]
        none_bin["candidates"] += appended
        none_bin["acceptance_rate"] = (
            none_bin["accepted"] / none_bin["candidates"] if none_bin["candidates"] else 0.0
        )
        # The near-duplicate rate is over every discovered candidate, so it is
        # recomputed after pre-scan rejections join the count.
        summary["near_duplicate_candidate_rate"] = (
            summary.get("near_duplicate_candidate_count", 0)
            / summary["candidate_count"]
            if summary["candidate_count"]
            else 0.0
        )
    summary["rejection_reason_counts"] = _count_reasons(diagnostics)
    summary["feature_byte_hash"] = _feature_hash(refs, features)
    summary["preprocessing"] = {
        "eval_transform": EVAL_TRANSFORM_NAME,
        "checkpoint": str(pseudo_checkpoint),
    }
    _emit(f"[Info] Pseudo round: {len(accepted)} candidates accepted")
    return accepted, diagnostics, summary


def _refs_sha256(refs: list[str]) -> str:
    payload = "\n".join(refs).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _count_reasons(diagnostics: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in diagnostics:
        for reason in row["rejection_reasons"]:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _feature_hash(refs: list[str], features: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(b"raw-cls-v1")
    for ref in sorted(refs):
        vector = np.asarray(features.get(ref, []), dtype="<f4")
        digest.update(ref.encode("utf-8"))
        digest.update(np.asarray(vector.shape, dtype="<i8").tobytes())
        digest.update(vector.tobytes())
    return digest.hexdigest()
