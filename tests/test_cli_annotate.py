from __future__ import annotations

from pathlib import Path

import pandas as pd
import otuformer.delineation.annotate as annotate_mod
from typer.testing import CliRunner

from otuformer.cli.main import app


runner = CliRunner()


def _write_cluster_like_layout(base: Path) -> tuple[Path, Path, Path, Path]:
    runs_dir = base / "runs"
    cluster_dir = runs_dir / "cluster"
    tables_dir = cluster_dir / "UPGMA" / "partitions" / "tables"
    logs_dir = cluster_dir / "logs"
    distance_dir = cluster_dir / "distance_statistics"
    extract_dir = runs_dir / "extract3"
    annotate_dir = runs_dir / "annotate"

    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    distance_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    annotate_dir.mkdir(parents=True, exist_ok=True)

    ids = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
    raw_df = pd.DataFrame(
        {
            "id": ids,
            "cluster": ["OTU_1", "OTU_1", "OTU_2", "OTU_2"],
            "sample": ["S1", "S1", "S2", "S2"],
        }
    )
    raw_path = tables_dir / "partition_0.2_assignments.csv"
    raw_df.to_csv(raw_path, index=False)

    raw_df.to_csv(tables_dir / "partition_0.25_assignments.csv", index=False)

    emb_df = pd.DataFrame(
        {
            "id": ids,
            "dim_0": [0.1, 0.1, 0.9, 0.8],
            "dim_1": [0.1, 0.2, 0.8, 0.9],
        }
    )
    emb_path = extract_dir / "embeddings.csv"
    emb_df.to_csv(emb_path, index=False)

    distance_df = pd.DataFrame(
        {
            "id": ids,
            "img1.jpg": [0.0, 0.1, 0.8, 0.9],
            "img2.jpg": [0.1, 0.0, 0.7, 0.8],
            "img3.jpg": [0.8, 0.7, 0.0, 0.2],
            "img4.jpg": [0.9, 0.8, 0.2, 0.0],
        }
    )
    distance_df.set_index("id").to_csv(distance_dir / "distance_matrix.csv")

    log_text = "\n".join(
        [
            "[2026-04-01 08:56:19] Parameters:",
            "[2026-04-01 08:56:19] {",
            '[2026-04-01 08:56:19]   "distance": "cosine",',
            '[2026-04-01 08:56:19]   "embeddings": "runs/extract3/embeddings.csv",',
            '[2026-04-01 08:56:19]   "pca_whitening": false,',
            '[2026-04-01 08:56:19]   "pca_components": 256,',
            '[2026-04-01 08:56:19]   "local_scaling": false,',
            '[2026-04-01 08:56:19]   "local_k": 0,',
            '[2026-04-01 08:56:19]   "local_k_strategy": "adaptive"',
            "[2026-04-01 08:56:19] }",
        ]
    )
    (logs_dir / "cluster.log").write_text(log_text, encoding="utf-8")

    corr_path = annotate_dir / "correction.csv"
    pd.DataFrame(
        {
            "id": ["img2.jpg", "img3.jpg"],
            "cluster": ["OTU_3", "OTU_3"],
            "sample": ["S1", "S2"],
        }
    ).to_csv(corr_path, index=False)

    return raw_path, corr_path, annotate_dir / "out", emb_path


def test_annotate_rejects_nonstandard_raw_assignments_path(tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_text("id,cluster\na,OTU_1\n", encoding="utf-8")
    corr = tmp_path / "corr.csv"
    corr.write_text("id,cluster\na,OTU_2\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "annotate",
            "--raw-assignments",
            str(raw),
            "--corrections",
            str(corr),
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "partition_<cutoff>_assignments.csv" in result.output


def test_annotate_happy_path_outputs(tmp_path):
    raw, corr, out_dir, emb_path = _write_cluster_like_layout(tmp_path)
    result = runner.invoke(
        app,
        [
            "annotate",
            "--raw-assignments",
            str(raw),
            "--corrections",
            str(corr),
            "--embeddings",
            str(emb_path),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0

    assert (out_dir / "partition_0.2_assignments.csv").exists()
    assert (out_dir / "partition_0.2_assignments_changed_only.csv").exists()
    assert (out_dir / "otu_table.csv").exists()
    assert (out_dir / "pairwise_distance_summary_intra-class.csv").exists()
    assert (out_dir / "UPGMA_tree_partitions_annotated.pdf").exists()
    assert (out_dir / "annotation_summary.json").exists()


def test_annotate_strict_mode_fails_when_correction_id_missing(tmp_path):
    raw, corr, out_dir, emb_path = _write_cluster_like_layout(tmp_path)
    corr_df = pd.read_csv(corr)
    corr_df.loc[0, "id"] = "unknown.jpg"
    corr_df.to_csv(corr, index=False)

    result = runner.invoke(
        app,
        [
            "annotate",
            "--raw-assignments",
            str(raw),
            "--corrections",
            str(corr),
            "--embeddings",
            str(emb_path),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code != 0
    assert "not found in --raw-assignments" in result.output


def test_annotate_without_embeddings_skips_intra_summary(tmp_path):
    raw, corr, out_dir, _emb_path = _write_cluster_like_layout(tmp_path)
    (
        tmp_path / "runs" / "cluster" / "distance_statistics" / "distance_matrix.csv"
    ).unlink()
    result = runner.invoke(
        app,
        [
            "annotate",
            "--raw-assignments",
            str(raw),
            "--corrections",
            str(corr),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0
    assert not (out_dir / "pairwise_distance_summary_intra-class.csv").exists()
    assert not (out_dir / "UPGMA_tree_partitions_annotated.pdf").exists()


def test_annotate_partitioning_bar_flag_defaults_off_and_can_enable(
    tmp_path, monkeypatch
):
    raw, corr, out_dir, emb_path = _write_cluster_like_layout(tmp_path)
    calls: list[bool] = []

    def _fake_render(*, out_path, show_partitioning_bars, **kwargs):
        calls.append(bool(show_partitioning_bars))
        out_path.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(annotate_mod, "render_annotated_partitions_pdf", _fake_render)

    default_run = runner.invoke(
        app,
        [
            "annotate",
            "--raw-assignments",
            str(raw),
            "--corrections",
            str(corr),
            "--embeddings",
            str(emb_path),
            "--out-dir",
            str(out_dir / "default"),
        ],
    )
    assert default_run.exit_code == 0

    enabled_run = runner.invoke(
        app,
        [
            "annotate",
            "--raw-assignments",
            str(raw),
            "--corrections",
            str(corr),
            "--embeddings",
            str(emb_path),
            "--show-partitioning-bars",
            "--out-dir",
            str(out_dir / "enabled"),
        ],
    )
    assert enabled_run.exit_code == 0
    assert calls == [False, True]
