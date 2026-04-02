"""Expert correction write-back and annotate post-processing helpers."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


_RAW_NAME_RE = re.compile(r"^partition_(?P<cutoff>.+)_assignments\.csv$")


def infer_cluster_context(raw_assignments_path: Path) -> dict[str, Path | str]:
    raw_name_match = _RAW_NAME_RE.match(raw_assignments_path.name)
    if raw_name_match is None:
        raise ValueError(
            "--raw-assignments must be named partition_<cutoff>_assignments.csv"
        )

    tables_dir = raw_assignments_path.parent
    if tables_dir.name != "tables":
        raise ValueError(
            "--raw-assignments must come from standard cluster output path "
            "(.../UPGMA/partitions/tables/partition_<cutoff>_assignments.csv)."
        )
    partitions_dir = tables_dir.parent
    if partitions_dir.name != "partitions":
        raise ValueError(
            "--raw-assignments must come from standard cluster output path "
            "(.../UPGMA/partitions/tables/partition_<cutoff>_assignments.csv)."
        )
    upgma_dir = partitions_dir.parent
    if upgma_dir.name != "UPGMA":
        raise ValueError(
            "--raw-assignments must come from standard cluster output path "
            "(.../UPGMA/partitions/tables/partition_<cutoff>_assignments.csv)."
        )

    cluster_run_dir = upgma_dir.parent
    if not (cluster_run_dir / "logs" / "cluster.log").exists():
        raise ValueError(
            "Detected cluster run directory does not contain logs/cluster.log."
        )

    return {
        "cutoff_tag": raw_name_match.group("cutoff"),
        "cluster_run_dir": cluster_run_dir,
        "upgma_dir": upgma_dir,
        "partitions_dir": partitions_dir,
        "tables_dir": tables_dir,
        "cluster_log": cluster_run_dir / "logs" / "cluster.log",
    }


def validate_raw_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "cluster"}
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise ValueError(
            f"--raw-assignments missing required columns: {', '.join(missing)}"
        )
    result = assignments.copy()
    result["id"] = result["id"].astype(str)
    result["cluster"] = result["cluster"].astype(str)
    if "sample" in result.columns:
        result["sample"] = result["sample"].astype(str)
    return result


def canonicalize_corrections(corrections: pd.DataFrame) -> pd.DataFrame:
    if "id" in corrections.columns:
        id_col = "id"
    elif "image" in corrections.columns:
        id_col = "image"
    else:
        raise ValueError("--corrections must contain 'id' or 'image' column.")

    if "cluster" in corrections.columns:
        cluster_col = "cluster"
    elif "corrected_cluster" in corrections.columns:
        cluster_col = "corrected_cluster"
    else:
        raise ValueError("--corrections must contain 'cluster' column.")

    cols = [id_col, cluster_col]
    if "sample" in corrections.columns:
        cols.append("sample")
    result = corrections[cols].copy()
    result.rename(columns={id_col: "id", cluster_col: "cluster"}, inplace=True)
    result["id"] = result["id"].astype(str)
    result["cluster"] = result["cluster"].astype(str)
    if "sample" in result.columns:
        result["sample"] = result["sample"].astype(str)
    return result


def validate_corrections(corrections: pd.DataFrame) -> None:
    if len(corrections) == 0:
        return

    empty_id = corrections["id"].str.strip() == ""
    empty_cluster = corrections["cluster"].str.strip() == ""
    if empty_id.any() or empty_cluster.any():
        raise ValueError("--corrections contains empty id/cluster values.")

    grouped = corrections.groupby("id")["cluster"].nunique(dropna=False)
    conflict_ids = grouped[grouped > 1].index.tolist()
    if conflict_ids:
        preview = ", ".join(conflict_ids[:10])
        raise ValueError(
            "--corrections contains conflicting duplicate ids with different cluster "
            f"values (examples: {preview})."
        )


def validate_corrections_against_assignments(
    assignments: pd.DataFrame, corrections: pd.DataFrame
) -> None:
    if len(corrections) == 0:
        return
    assignment_ids = set(assignments["id"].astype(str))
    corr_ids = set(corrections["id"].astype(str))
    missing = sorted(corr_ids - assignment_ids)
    if missing:
        preview = ", ".join(missing[:20])
        raise ValueError(
            "--corrections contains IDs not found in --raw-assignments. "
            f"First missing IDs: {preview}"
        )


def apply_corrections(
    assignments: pd.DataFrame, corrections: pd.DataFrame
) -> pd.DataFrame:
    result = validate_raw_assignments(assignments)
    corrections_df = canonicalize_corrections(corrections)
    validate_corrections(corrections_df)
    validate_corrections_against_assignments(result, corrections_df)
    corrections_df = corrections_df.drop_duplicates(subset=["id"], keep="last")

    if len(corrections) == 0:
        return result

    corr_map = dict(zip(corrections_df["id"], corrections_df["cluster"]))
    mask = result["id"].astype(str).isin(corr_map)
    result.loc[mask, "cluster"] = result.loc[mask, "id"].astype(str).map(corr_map)
    return result


def build_changed_only_table(
    original: pd.DataFrame, corrected: pd.DataFrame
) -> pd.DataFrame:
    changed = (
        original["cluster"].astype(str).values
        != corrected["cluster"].astype(str).values
    )
    cols = ["id"]
    if "sample" in original.columns:
        cols.append("sample")
    base = original.loc[changed, cols].copy()
    base["old_cluster"] = original.loc[changed, "cluster"].astype(str).values
    base["new_cluster"] = corrected.loc[changed, "cluster"].astype(str).values
    return base


def build_annotation_summary(original: pd.DataFrame, corrected: pd.DataFrame) -> dict:
    changed = original["cluster"].values != corrected["cluster"].values
    n_corrections = int(changed.sum())
    affected_original = set(original.loc[changed, "cluster"])
    affected_corrected = set(corrected.loc[changed, "cluster"])
    return {
        "n_corrections": n_corrections,
        "n_clusters_affected": len(affected_original | affected_corrected),
        "original_cluster_distribution": original["cluster"].value_counts().to_dict(),
        "corrected_cluster_distribution": corrected["cluster"].value_counts().to_dict(),
    }


def build_otu_table(assignments: pd.DataFrame) -> pd.DataFrame:
    df = assignments.copy()
    if "sample" not in df.columns:
        df["sample"] = "all_samples"
    otu = (
        df.groupby(["sample", "cluster"])
        .size()
        .unstack(fill_value=0)
        .sort_index(axis=1)
    )
    otu = otu.reset_index()
    return otu


def _strip_log_prefix(line: str) -> str:
    if line.startswith("[") and "] " in line:
        return line.split("] ", 1)[1]
    return line


def parse_cluster_params(cluster_log: Path) -> dict[str, object]:
    lines = [
        _strip_log_prefix(line.rstrip("\n"))
        for line in cluster_log.read_text(encoding="utf-8").splitlines()
    ]
    start = None
    end = None
    for i, line in enumerate(lines):
        if line.strip() == "Parameters:":
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == "{":
                    start = j
                    break
            break
    if start is None:
        return {}
    depth = 0
    for j in range(start, len(lines)):
        depth += lines[j].count("{")
        depth -= lines[j].count("}")
        if depth == 0:
            end = j
            break
    if end is None:
        return {}
    payload = "\n".join(lines[start : end + 1])
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return {}
    return {}


def resolve_repo_root(cluster_run_dir: Path) -> Path:
    if cluster_run_dir.parent.name == "runs":
        return cluster_run_dir.parent.parent
    return Path.cwd()


def resolve_embeddings_path(params: dict[str, object], cluster_run_dir: Path) -> Path:
    emb = str(params.get("embeddings", "")).strip()
    if emb == "":
        raise ValueError(
            "Unable to resolve embeddings path from cluster log parameters."
        )
    emb_path = Path(emb)
    if emb_path.is_absolute():
        return emb_path
    repo_root = resolve_repo_root(cluster_run_dir)
    return (repo_root / emb_path).resolve()


def resolve_distance_mode(params: dict[str, object]) -> str:
    mode = str(params.get("distance", "")).strip().lower()
    if mode not in {"cosine", "euclidean"}:
        raise ValueError("Unable to resolve distance mode from cluster log parameters.")
    return mode


def compute_distance_matrix_for_ids(
    ids: list[str],
    embeddings_path: Path,
    distance_mode: str,
    pca_whitening: bool = False,
    pca_components: int = 256,
    local_scaling: bool = False,
    local_k: int = 0,
    local_k_strategy: str = "adaptive",
) -> np.ndarray:
    from otuformer.delineation.distance import (
        apply_local_scaling,
        apply_pca_whitening,
        auto_select_k,
        compute_cosine_distances,
        compute_euclidean_distances,
    )

    emb_df = pd.read_csv(embeddings_path)
    id_to_row = {rid: i for i, rid in enumerate(emb_df["id"].astype(str))}
    missing = [rid for rid in ids if rid not in id_to_row]
    if missing:
        preview = ", ".join(missing[:20])
        raise ValueError(
            "Some IDs from assignments are missing in embeddings CSV. "
            f"First missing IDs: {preview}"
        )

    dim_cols = [c for c in emb_df.columns if c.startswith("dim_")]
    if not dim_cols:
        raise ValueError("Embeddings CSV has no dim_* columns.")

    idx = [id_to_row[rid] for rid in ids]
    x = emb_df.iloc[idx][dim_cols].to_numpy()

    if pca_whitening:
        n_comp = min(pca_components, x.shape[1], x.shape[0])
        x, _ = apply_pca_whitening(x, n_components=n_comp)

    if distance_mode == "euclidean":
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x_norm = x / np.maximum(norms, 1e-12)
        dist = compute_euclidean_distances(x_norm)
    else:
        dist = compute_cosine_distances(x)

    if local_scaling:
        k = (
            local_k
            if local_k > 0
            else auto_select_k(len(ids), strategy=local_k_strategy)
        )
        dist, _ = apply_local_scaling(dist, k=k)

    return dist


def summarize_intra_class_distances(
    dist_matrix: np.ndarray, ids: list[str], id_to_cluster: dict[str, str]
) -> pd.DataFrame:
    rows = []
    for cluster in sorted(set(id_to_cluster.values())):
        idx = [i for i, rid in enumerate(ids) if id_to_cluster.get(rid) == cluster]
        n_samples = len(idx)
        if n_samples < 2:
            rows.append(
                {
                    "class": cluster,
                    "n_samples": n_samples,
                    "n_pairs": 0,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "q25": np.nan,
                    "q75": np.nan,
                }
            )
            continue
        block = dist_matrix[np.ix_(idx, idx)]
        vals = block[np.triu_indices_from(block, k=1)]
        rows.append(
            {
                "class": cluster,
                "n_samples": n_samples,
                "n_pairs": int(len(vals)),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)),
            }
        )
    return pd.DataFrame(rows)


def load_partitions_from_tables(
    tables_dir: Path, ids: list[str]
) -> OrderedDict[float, np.ndarray]:
    files = sorted(tables_dir.glob("partition_*_assignments.csv"))
    parts: OrderedDict[float, np.ndarray] = OrderedDict()
    for path in files:
        m = _RAW_NAME_RE.match(path.name)
        if m is None:
            continue
        th = float(m.group("cutoff"))
        df = pd.read_csv(path)
        if "id" not in df.columns or "cluster" not in df.columns:
            continue
        map_row = dict(zip(df["id"].astype(str), df["cluster"].astype(str)))
        labels = np.array([map_row[i] for i in ids], dtype=object)
        parts[th] = labels
    return OrderedDict(sorted(parts.items(), key=lambda x: x[0]))


def render_annotated_partitions_pdf(
    z: np.ndarray,
    ids: list[str],
    partitions: OrderedDict[float, np.ndarray],
    corrected_labels: dict[str, str] | None,
    support_dict: dict[frozenset, float] | None,
    bootstrap_cutoff: float,
    corrected_bar_width: float,
    show_partitioning_bars: bool,
    figure_width: float | None,
    out_path: Path,
) -> None:
    from otuformer.cli.cluster import _plot_upgma_partition_tree_panel

    _plot_upgma_partition_tree_panel(
        z,
        ids,
        partitions,
        out_path,
        support_dict=support_dict,
        bootstrap_cutoff=bootstrap_cutoff,
        corrected_labels=corrected_labels,
        corrected_bar_width=corrected_bar_width,
        show_partitioning_bars=show_partitioning_bars,
        figure_width=figure_width,
    )


def parse_bootstrap_support_from_newick(
    newick_path: Path, ids: list[str]
) -> dict[frozenset, float]:
    text = newick_path.read_text(encoding="utf-8").strip()
    if text == "":
        return {}

    s = text
    n = len(s)
    idx = 0
    support: dict[frozenset, float] = {}

    def skip_ws(i: int) -> int:
        while i < n and s[i].isspace():
            i += 1
        return i

    def read_token(i: int) -> tuple[str, int]:
        start = i
        while i < n and s[i] not in ":,();":
            i += 1
        return s[start:i].strip(), i

    def skip_branch_len(i: int) -> int:
        if i < n and s[i] == ":":
            i += 1
            while i < n and s[i] not in ",();":
                i += 1
        return i

    def parse_node(i: int) -> tuple[set[str], int]:
        i = skip_ws(i)
        if i < n and s[i] == "(":
            i += 1
            left, i = parse_node(i)
            i = skip_ws(i)
            if i >= n or s[i] != ",":
                raise ValueError("Malformed Newick: expected ','")
            i += 1
            right, i = parse_node(i)
            i = skip_ws(i)
            if i >= n or s[i] != ")":
                raise ValueError("Malformed Newick: expected ')'")
            i += 1

            label, i = read_token(i)
            clade = left | right
            if label != "":
                try:
                    support[frozenset(clade)] = float(label)
                except ValueError:
                    pass

            i = skip_branch_len(i)
            return clade, i

        name, i = read_token(i)
        if name == "":
            raise ValueError("Malformed Newick: empty leaf name")
        i = skip_branch_len(i)
        return {name}, i

    clade, idx = parse_node(0)
    idx = skip_ws(idx)
    if idx < n and s[idx] == ";":
        idx += 1
    idx = skip_ws(idx)
    if idx != n:
        raise ValueError("Malformed Newick: trailing tokens")

    id_set = set(ids)
    if not set(clade).issubset(id_set):
        return {}
    return support
