"""Alpha diversity metrics from OTU assignment tables."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _is_number(value: object) -> bool:
    try:
        float(str(value).strip())
        return True
    except ValueError:
        return False


def detect_otu_table_header(raw: pd.DataFrame, has_header: bool = False) -> bool:
    if raw.empty or raw.shape[1] < 2:
        raise ValueError("OTU table must have at least 2 columns.")
    if has_header:
        return True
    first_row = raw.iloc[0, 1:]
    any_non_numeric = any(
        not pd.isna(v) and str(v).strip() != "" and not _is_number(v) for v in first_row
    )
    if any_non_numeric:
        return True
    raise ValueError("OTU IDs required: provide header row or --otu-table-has-header.")


def parse_otu_table(raw: pd.DataFrame, has_header: bool) -> pd.DataFrame:
    """Parse a raw OTU table DataFrame into a sample×OTU count matrix.

    Row 0 is the header (sample name + OTU IDs in columns 2..N).
    Subsequent rows are samples with abundance values.
    """
    if not detect_otu_table_header(raw, has_header=has_header):
        raise ValueError("OTU table requires header row.")
    if len(raw) < 2:
        raise ValueError("OTU table must have a header row and at least one data row.")
    otu_ids = raw.iloc[0, 1:].astype(str).str.strip().tolist()
    if any(otu == "" for otu in otu_ids):
        raise ValueError("OTU header cells (columns 2..N) must be non-empty.")
    data = raw.iloc[1:, :].copy()
    data.columns = ["sample"] + otu_ids
    data["sample"] = data["sample"].astype(str)
    for otu in otu_ids:
        data[otu] = pd.to_numeric(data[otu], errors="coerce").fillna(0).astype(int)
    data = data.set_index("sample")
    return data


def normalize_assignments(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize assignment column names: accept id|image, cluster, optional sample."""
    cols = {c.strip().lower(): c for c in df.columns}
    if "id" in cols and "image" in cols:
        raise ValueError("Assignments cannot contain both id and image columns.")
    if "id" in cols:
        df = df.rename(columns={cols["id"]: "id"})
    elif "image" in cols:
        df = df.rename(columns={cols["image"]: "id"})
    else:
        raise ValueError("Assignments must include id or image column.")
    if "cluster" not in cols:
        raise ValueError("Assignments must include cluster column.")
    df = df.rename(columns={cols["cluster"]: "cluster"})
    if "sample" in cols:
        df = df.rename(columns={cols["sample"]: "sample"})
    return df


def has_valid_samples(assignments: pd.DataFrame) -> bool:
    """Return True if 'sample' column exists and no value is NaN or empty after trimming."""
    if "sample" not in assignments.columns:
        return False
    col = assignments["sample"]
    if col.isna().any():
        return False
    cleaned = col.astype(str).str.strip()
    return bool(cleaned.ne("").all())


def sanitize_sample_name(name: str) -> str:
    """Normalize a sample name: collapse whitespace, replace / and \\."""
    cleaned = re.sub(r"\s+", "_", name.strip())
    return cleaned.replace("/", "_").replace("\\", "_")


def dedupe_sample_names(names: list[str]) -> list[str]:
    """Append numeric suffixes for duplicate names."""
    seen: dict[str, int] = {}
    output = []
    for name in names:
        count = seen.get(name, 0) + 1
        seen[name] = count
        output.append(name if count == 1 else f"{name}_{count}")
    return output


def split_assignments_by_sample(
    assignments: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Group assignments by the 'sample' column, returning a dict of DataFrames."""
    grouped = {}
    for sample, chunk in assignments.groupby("sample"):
        grouped[str(sample)] = chunk.copy()
    return grouped


def build_per_sample_paths(samples: list[str], out_dir: Path) -> dict[str, Path]:
    """Return {sample: path} with sanitized, deduplicated filenames."""
    sanitized = [sanitize_sample_name(s) for s in samples]
    deduped = dedupe_sample_names(sanitized)
    return {name: out_dir / f"{name}.csv" for name in deduped}


def build_diversity_tables_from_otu_table(
    otu_table: pd.DataFrame,
    min_values: list[int],
    tree_newick_path: Optional[Path],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compute global and per-sample diversity tables from an OTU count matrix.

    Returns (global_table, {sample: per_sample_table}).
    """
    global_counts = otu_table.sum(axis=0)
    global_assignments = pd.DataFrame(
        {"cluster": global_counts.index.repeat(global_counts.values.astype(int))}
    )
    global_table = diversity_table(global_assignments, min_values, tree_newick_path)
    per_sample = {}
    for sample, row in otu_table.iterrows():
        assignments = pd.DataFrame(
            {"cluster": row.index.repeat(row.values.astype(int))}
        )
        per_sample[str(sample)] = diversity_table(
            assignments, min_values, tree_newick_path
        )
    return global_table, per_sample


def filter_by_min_abundance(
    assignments: pd.DataFrame, min_abundance: int
) -> pd.DataFrame:
    if min_abundance <= 0:
        return assignments.copy()
    counts = assignments["cluster"].value_counts()
    keep = counts[counts >= min_abundance].index
    return assignments[assignments["cluster"].isin(keep)].copy()


def _abundance_vector(assignments: pd.DataFrame) -> np.ndarray:
    return np.array(sorted(assignments["cluster"].value_counts().values, reverse=True))


def _fisher_alpha(s: int, n: int, max_iter: int = 100, tol: float = 1e-6) -> float:
    if n <= 0 or s <= 0:
        return 0.0
    alpha = s / max(math.log(1 + n), 1e-12)
    for _ in range(max_iter):
        f = alpha * math.log(1 + n / alpha) - s
        df = math.log(1 + n / alpha) - n / (n + alpha)
        if abs(df) < 1e-12:
            break
        alpha_new = alpha - f / df
        if alpha_new <= 0:
            alpha_new = alpha / 2
        if abs(alpha_new - alpha) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new
    return float(alpha)


def compute_alpha_diversity(assignments: pd.DataFrame) -> dict[str, float]:
    counts = _abundance_vector(assignments)
    n = int(counts.sum())
    s = int(len(counts))
    if n == 0 or s == 0:
        return {
            "Richness": 0.0,
            "Chao1": 0.0,
            "ACE": 0.0,
            "Margalef": 0.0,
            "Menhinick": 0.0,
            "Shannon": 0.0,
            "Simpson": 0.0,
            "InverseSimpson": 0.0,
            "Brillouin": 0.0,
            "Fisher_alpha": 0.0,
            "Pielou_J": 0.0,
            "Heip_E": 0.0,
            "Berger_Parker": 0.0,
            "Hill_q0": 0.0,
            "Hill_q1": 0.0,
            "Hill_q2": 0.0,
        }

    props = counts / n
    shannon = -float(np.sum(props * np.log(props + 1e-300)))
    simpson = float(1.0 - np.sum(props**2))
    inv_simpson = float(1.0 / np.sum(props**2)) if np.sum(props**2) > 0 else 0.0
    pielou_j = shannon / math.log(s) if s > 1 else 0.0
    heip_e = (math.exp(shannon) - 1) / (s - 1) if s > 1 else 0.0
    berger_parker = float(counts[0] / n)
    margalef = (s - 1) / math.log(n) if n > 1 else 0.0
    menhinick = s / math.sqrt(n) if n > 0 else 0.0

    n1 = int((counts == 1).sum())
    n2 = int((counts == 2).sum())
    chao1 = (
        float(s + (n1 * (n1 - 1)) / (2 * (n2 + 1)))
        if n2 > 0
        else float(s + n1 * (n1 - 1) / 2)
    )

    rare = counts[counts <= 10]
    s_rare = len(rare)
    s_abund = s - s_rare
    n_rare = int(rare.sum())
    f1 = int((counts == 1).sum())
    if n_rare > 0 and s_rare > 0:
        c_ace = 1 - f1 / n_rare
        if c_ace > 0 and n_rare > 1:
            gamma_sq = max(
                0.0,
                s_rare
                / c_ace
                * sum(k * (k - 1) * (counts == k).sum() for k in range(1, 11))
                / (n_rare * (n_rare - 1))
                - 1,
            )
            ace = float(s_abund + s_rare / c_ace + f1 * gamma_sq / c_ace)
        else:
            ace = float(s)
    else:
        ace = float(s)

    brillouin = (
        (math.lgamma(n + 1) - sum(math.lgamma(int(c) + 1) for c in counts)) / n
        if n > 0
        else 0.0
    )
    fisher_alpha = _fisher_alpha(s, n)

    hill_q0 = float(s)
    hill_q1 = float(math.exp(shannon))
    hill_q2 = float(inv_simpson)

    return {
        "Richness": float(s),
        "Chao1": chao1,
        "ACE": ace,
        "Margalef": margalef,
        "Menhinick": menhinick,
        "Shannon": shannon,
        "Simpson": simpson,
        "InverseSimpson": inv_simpson,
        "Brillouin": brillouin,
        "Fisher_alpha": fisher_alpha,
        "Pielou_J": pielou_j,
        "Heip_E": heip_e,
        "Berger_Parker": berger_parker,
        "Hill_q0": hill_q0,
        "Hill_q1": hill_q1,
        "Hill_q2": hill_q2,
    }


def build_mpd_inputs(
    otu_ids: list[str], counts: list[int]
) -> tuple[list[str], np.ndarray]:
    return otu_ids, np.array(counts, dtype=int)


def compute_mpd_from_counts(
    otu_ids: list[str], counts: np.ndarray, tree_newick_path: Path
) -> dict[str, float]:
    import warnings

    try:
        from skbio import TreeNode
        from skbio.diversity import alpha_diversity
        from skbio.diversity.alpha import phydiv
    except Exception as exc:
        raise ImportError("scikit-bio required for MPD computation") from exc

    tree = TreeNode.read(str(tree_newick_path), convert_underscores=False)
    tip_names = {tip.name for tip in tree.tips()}
    taxa_set = set(otu_ids)

    missing_from_tree = taxa_set - tip_names
    if missing_from_tree:
        warnings.warn(f"Taxa missing from tree: {missing_from_tree}; dropped for MPD.")

    overlap = [oid for oid in otu_ids if oid in tip_names]
    if not overlap:
        return {
            "MPD": float("nan"),
            "MPD_w": float("nan"),
            "PD_richness_norm": float("nan"),
        }
    overlap_counts = np.array([counts[otu_ids.index(oid)] for oid in overlap])
    tree_pruned = tree.shear(overlap)

    faith_pd = alpha_diversity(
        "faith_pd",
        overlap_counts.reshape(1, -1),
        taxa=overlap,
        tree=tree_pruned,
    )
    mpd_w = phydiv(
        overlap_counts,
        taxa=overlap,
        tree=tree_pruned,
        rooted=True,
        weight=True,
        validate=False,
    )
    richness = len(overlap)
    pd_richness_norm = float(faith_pd[0]) / richness if richness > 0 else float("nan")

    return {
        "MPD": float(faith_pd[0]),
        "MPD_w": float(mpd_w),
        "PD_richness_norm": pd_richness_norm,
    }


def compute_mpd(assignments: pd.DataFrame, tree_newick_path: Path) -> dict[str, float]:
    try:
        from skbio import TreeNode
        from skbio.diversity import alpha_diversity
        from skbio.diversity.alpha import phydiv
    except Exception as exc:
        raise ImportError("scikit-bio required for MPD computation") from exc

    if "id" in assignments.columns:
        tip_ids = list(assignments["id"].value_counts().index)
        counts_arr = np.array(assignments["id"].value_counts().values, dtype=int)
    else:
        cluster_counts = assignments["cluster"].value_counts()
        tip_ids = list(cluster_counts.index)
        counts_arr = np.array([cluster_counts.get(o, 0) for o in tip_ids], dtype=int)
    try:
        return compute_mpd_from_counts(tip_ids, counts_arr, tree_newick_path)
    except Exception:
        return {
            "MPD": float("nan"),
            "MPD_w": float("nan"),
            "PD_richness_norm": float("nan"),
        }


def diversity_table(
    assignments: pd.DataFrame,
    min_abundances: list[int],
    tree_newick_path: Optional[Path] = None,
) -> pd.DataFrame:
    records: dict[str, dict[str, float]] = {}
    for min_ab in min_abundances:
        filtered = filter_by_min_abundance(assignments, min_ab)
        col = f"min_abundance_{min_ab}"
        if len(filtered) == 0:
            records[col] = {}
            continue
        metrics = compute_alpha_diversity(filtered)
        if tree_newick_path is not None:
            try:
                phylo_metrics = compute_mpd(filtered, tree_newick_path)
                metrics.update(phylo_metrics)
            except Exception:
                metrics["MPD"] = float("nan")
                metrics["MPD_w"] = float("nan")
                metrics["PD_richness_norm"] = float("nan")
        records[col] = metrics
    df = pd.DataFrame(records)
    df.index.name = "index"
    return df
