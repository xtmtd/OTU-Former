"""Expert correction write-back for partition assignments."""

from __future__ import annotations

import pandas as pd


def apply_corrections(
    assignments: pd.DataFrame, corrections: pd.DataFrame
) -> pd.DataFrame:
    result = assignments.copy()
    if len(corrections) == 0:
        return result
    corr_map = dict(
        zip(corrections["id"].astype(str), corrections["corrected_cluster"])
    )
    mask = result["id"].astype(str).isin(corr_map)
    result.loc[mask, "cluster"] = result.loc[mask, "id"].astype(str).map(corr_map)
    return result


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
