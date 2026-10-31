"""extract command - embeddings extraction."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import typer

from otuformer.cli import SIZE_EXAMPLES, _parse_size, docs_url, format_user_command
from otuformer.cli.constraints import (
    validate_eval_transform,
    validate_extract_data_options,
    validate_size_option,
)
from otuformer.constants import EVAL_TRANSFORM_CHOICES

app = typer.Typer(
    help=(
        "Extract embeddings from images.\n\n"
        "Runs a trained model (PyTorch checkpoint or ONNX) over an image directory and\n"
        "produces a CSV with embedding vectors. Supports multiple token modes (CLS,\n"
        "patch-topk, attention-pool) and optional UMAP visualization.\n\n"
        "Quick example:\n\n"
        "  otuformer extract --checkpoint runs/finetune/best.pt --input-images-dir ./images\n"
        "  otuformer extract --checkpoint runs/finetune/best.pt --input-images-dir ./images --label-csv labels.csv --token-mode attention-pool\n"
        "  otuformer extract --onnx-path runs/export/encoder.onnx --input-images-dir ./images\n"
        "\nDocs:\n"
        f"  {docs_url('extract')}\n"
    )
)


_EVAL_TRANSFORM_CHOICES = EVAL_TRANSFORM_CHOICES


def _validate_extract_options(
    *, eval_transform: str, topk_patches: int = 20, attention_pooling_epochs: int = 20
) -> None:
    """Reject extract options outside their accepted values before any output.

    Uses Typer-native validation instead of a Click-specific choice type so
    Typer builds commands the same way whether Click is installed or vendored
    by Typer. The size guards mirror otuformer.embedding.extractor; running them
    here keeps a bad --topk-patches/--attention-pooling-epochs proposal from
    clearing an overwritten --out-dir before the core rejects it.
    """
    try:
        validate_eval_transform(eval_transform)
        validate_size_option(topk_patches, stage="--topk-patches")
        validate_size_option(
            attention_pooling_epochs, stage="--attention-pooling-epochs"
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None


@app.callback(invoke_without_command=True)
def extract(
    ctx: typer.Context,
    checkpoint: Path | None = typer.Option(
        None,
        "--checkpoint",
        help="Path to pretrain or finetune checkpoint. Required unless --onnx-path is provided.",
    ),
    input_images_dir: Path = typer.Option(
        ..., "--input-images-dir", help="Image directory or parent directory."
    ),
    out_dir: Path = typer.Option(
        Path("runs/extract"), "--out-dir", help="Output directory."
    ),
    model_name: str = typer.Option(
        "vit_tiny_patch16_224", "--model-name", help="timm backbone."
    ),
    extract_size: str = typer.Option(
        "auto",
        "--extract-size",
        help=(
            "Resize/crop size for extraction. 'auto' uses the checkpoint's "
            "recorded training size. "
            f"{SIZE_EXAMPLES}"
        ),
    ),
    eval_transform: str = typer.Option(
        "center-crop",
        "--eval-transform",
        help=(
            "Evaluation preprocessing protocol: center-crop (Resize + CenterCrop) "
            "or whole-specimen-pad (aspect-preserving square padding). "
            "Default: center-crop."
        ),
    ),
    use_projector_output: bool = typer.Option(
        False,
        "--use-projector-output",
        help=(
            "Use SSL projector output for SSL checkpoints, or the ArcFace embedding "
            "for fine-tune checkpoints; useful for fine-tuned-class retrieval and "
            "closed-set clustering."
        ),
    ),
    use_student: bool = typer.Option(
        False,
        "--use-student",
        help=(
            "Load student weights instead of teacher (EMA model). "
            "Default loads teacher, which gives better embeddings for downstream "
            "analysis (DINO/iBOT convention). No effect on finetune checkpoints."
        ),
    ),
    token_mode: str = typer.Option(
        "cls",
        "--token-mode",
        help=(
            "Embedding token mode. cls=raw CLS token (the comparison baseline for "
            "unseen classes, cross-dataset transfer, and general morphology); "
            "patch-topk=top-K patch tokens + PCA; attention-pool=learned query pooling "
            "over patch tokens."
        ),
    ),
    topk_patches: int = typer.Option(
        20,
        "--topk-patches",
        help="Top-K patch tokens for patch-topk mode (common: 10/20/30).",
    ),
    attention_pooling_type: str = typer.Option(
        "lightweight",
        "--attention-pooling-type",
        help="Attention pooling type. Choices: lightweight | multihead | gated.",
    ),
    attention_pooling_epochs: int = typer.Option(
        20,
        "--attention-pooling-epochs",
        help=(
            "Epochs to finetune attention query when no cached pooling checkpoint is found."
        ),
    ),
    label_csv: Path | None = typer.Option(
        None,
        "--label-csv",
        help=(
            "CSV with a required 'image' column plus optional 'label' and any other "
            "metadata. Supplying it selects the shared, strictly validated data path; "
            "it is the selector, the metadata source, and the metrics/UMAP label "
            "source. In attention-pool mode it must include labels when cached "
            "pooling weights are missing."
        ),
    ),
    coco_instances: Path | None = typer.Option(
        None,
        "--coco-instances",
        help="COCO instance annotations (boxes, polygons, RLE).",
    ),
    coco_keypoints: Path | None = typer.Option(
        None,
        "--coco-keypoints",
        help="COCO keypoint annotations.",
    ),
    coco_panoptic: Path | None = typer.Option(
        None,
        "--coco-panoptic",
        help=(
            "COCO panoptic JSON. Requires --panoptic-masks-dir."
        ),
    ),
    panoptic_masks_dir: Path | None = typer.Option(
        None,
        "--panoptic-masks-dir",
        help=(
            "Root for panoptic segment-ID PNGs, resolved from each annotation's "
            "file_name as written. Requires --coco-panoptic."
        ),
    ),
    metrics_sample_size: int = typer.Option(
        10000,
        "--metrics-sample-size",
        help="Max samples for metrics and UMAP evaluation (<=0 means no cap).",
    ),
    # UMAP options (mirrors pretrain/finetune)
    umap_n_neighbors: int = typer.Option(
        15, "--umap-n-neighbors", help="UMAP n_neighbors."
    ),
    umap_min_dist: float = typer.Option(0.1, "--umap-min-dist", help="UMAP min_dist."),
    umap_metric: str = typer.Option(
        "cosine",
        "--umap-metric",
        help="Distance metric for UMAP projection. Common choices: cosine, euclidean.",
    ),
    visualize_class_number: int = typer.Option(
        20,
        "--visualize-class-number",
        help="Max classes to show in UMAP plot.",
    ),
    disable_umap: bool = typer.Option(
        False,
        "--disable-umap",
        help="Skip UMAP generation even when --label-csv is provided.",
    ),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size."),
    num_workers: int = typer.Option(4, "--num-workers", help="DataLoader workers."),
    device: str = typer.Option(
        "auto", "--device", help="Device: auto | cpu | cuda | mps."
    ),
    onnx_path: Path | None = typer.Option(
        None,
        "--onnx-path",
        help=(
            "Path to exported ONNX model for inference. "
            "When provided, uses ONNX Runtime instead of PyTorch. "
            "Only supports token-mode 'cls' (default). "
            "Overrides --checkpoint, --model-name, --use-student, and --device."
        ),
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed."),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Clear an existing non-empty output directory."
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    _validate_extract_options(
        eval_transform=eval_transform,
        topk_patches=topk_patches,
        attention_pooling_epochs=attention_pooling_epochs,
    )

    try:
        validate_extract_data_options(
            label_csv=label_csv,
            coco_instances=coco_instances,
            coco_keypoints=coco_keypoints,
            coco_panoptic=coco_panoptic,
            panoptic_masks_dir=panoptic_masks_dir,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None

    new_data = _new_data_inputs(
        label_csv=label_csv,
        coco_instances=coco_instances,
        coco_keypoints=coco_keypoints,
        coco_panoptic=coco_panoptic,
        panoptic_masks_dir=panoptic_masks_dir,
    )
    if new_data:
        reader_inputs = dict(new_data)
        new_params = dict(
            checkpoint=str(checkpoint) if checkpoint is not None else "",
            input_images_dir=str(input_images_dir),
            out_dir=str(out_dir),
            overwrite=overwrite,
            model_name=model_name,
            extract_size=extract_size,
            eval_transform=eval_transform,
            use_projector_output=use_projector_output,
            use_student=use_student,
            token_mode=token_mode,
            topk_patches=topk_patches,
            attention_pooling_type=attention_pooling_type,
            attention_pooling_epochs=attention_pooling_epochs,
            metrics_sample_size=metrics_sample_size,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            umap_metric=umap_metric,
            visualize_class_number=visualize_class_number,
            disable_umap=disable_umap,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            seed=seed,
            onnx_path=str(onnx_path) if onnx_path is not None else "",
            label_csv=str(label_csv) if label_csv is not None else "",
            coco_instances=str(coco_instances) if coco_instances is not None else "",
            coco_keypoints=str(coco_keypoints) if coco_keypoints is not None else "",
            coco_panoptic=str(coco_panoptic) if coco_panoptic is not None else "",
            panoptic_masks_dir=(
                str(panoptic_masks_dir) if panoptic_masks_dir is not None else ""
            ),
        )
        settings = dict(new_params)
        # One CSV is the selector and the metadata source on this path.
        settings["manifest"] = label_csv
        settings["label_csv"] = label_csv
        settings["checkpoint"] = checkpoint
        settings["onnx_path"] = onnx_path
        settings["extract_size"] = _parse_size(extract_size, stage="--extract-size")
        settings["cli_command"] = format_user_command(ctx, new_params, "extract")
        _run_new_data_path(
            settings=settings, inputs=reader_inputs, params=new_params
        )
        return

    from otuformer.cli_schema import (
        check_inputs_survive_cleanup,
        check_output_safety,
        input_identities,
        verify_inputs_unchanged,
    )
    from otuformer.embedding.extractor import extract_embeddings
    from otuformer.utils.io import prepare_output_dir, write_csv

    # --overwrite must never remove the label CSV this run reads, nor clear the
    # image root itself. An --out-dir *inside* the image root stays supported.
    from otuformer.data.paths import is_within

    # The entry the user typed is what must survive: `--input-images-dir out/via`
    # stays unusable once cleanup removes `out/via`, even though the directory it
    # pointed at is untouched. Both are recorded, so the post-check sees either loss.
    # The model inputs are read by this run too. Enumerated before anything is
    # cleared, with the same helpers the shared path uses, so a checkpoint, graph,
    # external tensor or reusable pooling file inside the output directory is
    # refused instead of deleted.
    model_inputs: list[Path] = []
    if checkpoint is not None:
        model_inputs.append(Path(checkpoint))
    if onnx_path is not None:
        model_inputs.extend(_onnx_input_paths(Path(onnx_path)))
    if token_mode == "attention-pool":
        model_inputs.extend(
            _pool_candidate_paths(
                out_dir,
                {
                    "attention_pooling_type": attention_pooling_type,
                    "checkpoint": checkpoint,
                },
            )
        )
    written_inputs = [
        input_images_dir,
        *([label_csv] if label_csv is not None else []),
        *model_inputs,
    ]
    identities = input_identities(written_inputs)
    # The written entries are checked, not only their targets, so an input reached
    # through a link inside the output directory is refused before anything is
    # cleared. An --out-dir inside the image root stays supported, because checking
    # the entry there finds nothing inside the output directory.
    try:
        if label_csv is not None:
            check_output_safety(
                out_dir, protected_paths=[label_csv], overwrite=overwrite
            )
        if overwrite:
            check_inputs_survive_cleanup(out_dir, written_inputs)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None
    out_target = Path(out_dir).expanduser().resolve()
    images_root = Path(input_images_dir).expanduser().resolve()
    if is_within(out_target, images_root):
        raise typer.BadParameter(
            f"--out-dir {out_target} would clear the image root {images_root}; "
            "choose a directory outside it"
        )
    prepare_output_dir(out_dir, overwrite=overwrite)
    try:
        verify_inputs_unchanged(identities)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None

    from otuformer.utils.logging import TeeLogger

    tee = TeeLogger(out_dir / "logs" / "extract.log")
    original_stderr = sys.stderr
    sys.stdout = tee
    sys.stderr = tee
    try:
        params = dict(
            checkpoint=str(checkpoint) if checkpoint is not None else "",
            input_images_dir=str(input_images_dir),
            out_dir=str(out_dir),
            overwrite=overwrite,
            model_name=model_name,
            extract_size=extract_size,
            eval_transform=eval_transform,
            use_projector_output=use_projector_output,
            use_student=use_student,
            token_mode=token_mode,
            topk_patches=topk_patches,
            attention_pooling_type=attention_pooling_type,
            attention_pooling_epochs=attention_pooling_epochs,
            label_csv=str(label_csv) if label_csv is not None else "",
            metrics_sample_size=metrics_sample_size,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            umap_metric=umap_metric,
            visualize_class_number=visualize_class_number,
            disable_umap=disable_umap,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            seed=seed,
            onnx_path=str(onnx_path) if onnx_path is not None else "",
        )
        if checkpoint is not None:
            attention_pooling_checkpoint = (
                out_dir
                / "checkpoints"
                / f"{checkpoint.stem}.{attention_pooling_type}_pooling.pth"
            )
        else:
            attention_pooling_checkpoint = (
                out_dir / "checkpoints" / f"model.{attention_pooling_type}_pooling.pth"
            )
        params["attention_pooling_checkpoint"] = str(attention_pooling_checkpoint)
        cli_command = format_user_command(ctx, params, "extract")
        print(f"Command: {cli_command}")
        print("Parameters:")
        print(json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True))
        print("-" * 80)

        # Enable MPS fallback for ops not yet supported on MPS
        if device in {"mps", "auto"}:
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        df = extract_embeddings(
            checkpoint_path=checkpoint,
            images_dir=input_images_dir,
            model_name=model_name,
            extract_size=_parse_size(extract_size, stage="--extract-size"),
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
            use_projector_output=use_projector_output,
            use_student=use_student,
            token_mode=token_mode,
            topk_patches=topk_patches,
            attention_pooling_type=attention_pooling_type,
            attention_pooling_epochs=attention_pooling_epochs,
            attention_train_csv=label_csv,
            seed=seed,
            extract_csv=label_csv,
            attention_pooling_checkpoint_path=attention_pooling_checkpoint,
            onnx_path=onnx_path,
            eval_transform=eval_transform,
        )
        out_path = out_dir / "embeddings.csv"
        write_csv(df, out_path)
        print(f"Embeddings saved to: {out_path} ({len(df)} samples)")

        if label_csv is not None:
            import numpy as np
            import pandas as pd

            from otuformer.embedding.evaluator import (
                compute_clustering_metrics,
                compute_knn_accuracy,
                compute_linear_probing_metrics,
                compute_map,
                compute_recall_at_k,
                run_umap,
            )

            label_df = pd.read_csv(label_csv)
            if "image" not in label_df.columns:
                raise ValueError("--label-csv must contain an 'image' column")

            # Align labels to embedding rows by filename (id column).
            emb_ids = df["id"].tolist()
            emb_id_set = set(emb_ids)
            label_df = label_df[label_df["image"].isin(emb_id_set)].copy()
            duplicate_images = (
                label_df["image"][label_df["image"].duplicated()].unique().tolist()
            )
            if duplicate_images:
                raise ValueError(
                    "--label-csv contains duplicate 'image' rows: "
                    f"{', '.join(str(name) for name in duplicate_images[:5])}"
                )
            label_indexed = label_df.set_index("image")
            valid_ids = [x for x in emb_ids if x in label_indexed.index]
            emb_mask = df["id"].isin(set(valid_ids))
            dim_cols = [c for c in df.columns if c.startswith("dim_")]
            embeddings = df.loc[emb_mask, dim_cols].values
            labels = (
                label_indexed.loc[df.loc[emb_mask, "id"].tolist(), "label"].values
                if "label" in label_df.columns
                else None
            )

            n = len(embeddings)
            if metrics_sample_size > 0 and n > metrics_sample_size:
                rng = np.random.default_rng(seed)
                # Sorted indices: trainer._maybe_subsample_for_metrics applies the
                # same seeded, sorted subsample order so both paths score the same
                # rows in the same order.
                idx = np.sort(rng.choice(n, size=metrics_sample_size, replace=False))
                embeddings = embeddings[idx]
                if labels is not None:
                    labels = labels[idx]
                print(
                    f"[Info] Visualization subsample: {len(idx)}/{n} samples (seed={seed})"
                )

            if labels is not None and len(np.unique(labels)) >= 2:
                print(f"[Metrics] Computing metrics on {len(embeddings)} samples ...")
                metrics: dict[str, object] = {}

                try:
                    metrics.update(compute_recall_at_k(embeddings, labels))
                except Exception as e:
                    print(f"[Warning] Recall@K failed: {e}")
                try:
                    metrics.update(compute_knn_accuracy(embeddings, labels))
                except Exception as e:
                    print(f"[Warning] kNN accuracy failed: {e}")
                try:
                    probing = compute_linear_probing_metrics(embeddings, labels)
                    metrics["Linear_Probing_Acc"] = probing["Linear_Probing_Acc"]
                    metrics["Linear_Probing_Balanced_Acc"] = probing[
                        "Linear_Probing_Balanced_Acc"
                    ]
                except Exception as e:
                    print(f"[Warning] Linear probing failed: {e}")
                try:
                    metrics["mAP"] = compute_map(embeddings, labels)
                except Exception as e:
                    print(f"[Warning] mAP failed: {e}")
                try:
                    metrics.update(compute_clustering_metrics(embeddings, labels))
                except Exception as e:
                    print(f"[Warning] Clustering metrics failed: {e}")

                metrics_path = out_dir / "metrics.csv"
                metrics_rows = [
                    {"metric": k, "value": "" if v is None else v}
                    for k, v in metrics.items()
                ]
                pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
                print(f"[Metrics] Results saved to: {metrics_path}")
                for k, v in metrics.items():
                    if v != "" and v is not None:
                        try:
                            print(f"  {k}: {float(v):.4f}")
                        except (TypeError, ValueError):
                            print(f"  {k}: {v}")
            elif labels is not None:
                print("[Info] Skipping supervised metrics: label-csv contains fewer than two classes.")

            if disable_umap:
                pass
            elif len(embeddings) >= 10:
                umap_path = out_dir / "umap.pdf"
                print(f"[UMAP] Generating UMAP projection -> {umap_path}")
                try:
                    run_umap(
                        embeddings,
                        labels,
                        umap_path,
                        n_neighbors=umap_n_neighbors,
                        min_dist=umap_min_dist,
                        metric=umap_metric,
                        max_classes=visualize_class_number,
                    )
                    print(f"[UMAP] Saved to: {umap_path}")
                except Exception as e:
                    print(f"[Warning] UMAP failed: {e}")
            else:
                print("[Info] Skipping UMAP: fewer than 10 samples after sampling.")

    except Exception:
        traceback.print_exc(file=tee)
        raise
    finally:
        sys.stdout = tee.terminal
        sys.stderr = original_stderr
        tee.close()


# --- shared v2 data path --------------------------------------------------

class _RecordedFailure(Exception):
    """A failure whose run record was already written by the caller."""


# Metric-family name to the key a failed family is recorded under.
_FAMILY_KEY = {
    "recall_at_k": "Recall@K",
    "knn_accuracy": "kNN_Accuracy",
    "linear_probing": "Linear_Probing",
    "map": "mAP",
    "clustering": "Clustering",
}


def _failed_outcome(family: str, error: BaseException):
    from otuformer.embedding.evaluator import MetricOutcome

    return MetricOutcome(
        None, "failed", f"{family} metric family raised {type(error).__name__}", {"family": family}
    )


def _new_data_inputs(*, label_csv=None, **values) -> dict[str, Path]:
    """Reader inputs for the shared path; the CSV is the ``manifest`` selector."""
    merged = {"manifest": label_csv, **values}
    return {name: Path(value) for name, value in merged.items() if value is not None}


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _artifact_entry(out_dir: Path, path: Path, rows: int | None = None) -> dict:
    import hashlib

    entry = {
        "path": Path(path).relative_to(out_dir).as_posix(),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "bytes": Path(path).stat().st_size,
    }
    if rows is not None:
        entry["rows"] = int(rows)
    return entry


def _evaluate_records(
    bundle,
    frame,
    *,
    sample_size: int,
    seed: int,
    enable_visualization: bool,
    disable_umap: bool,
    out_dir: Path,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    visualize_class_number: int,
) -> tuple[dict, dict]:
    """Metrics and optional UMAP derived from the parsed records, never a CSV."""
    import numpy as np
    import pandas as pd

    from otuformer.embedding.evaluator import (
        _compute_clustering_metrics_result,
        _compute_knn_accuracy_result,
        _compute_linear_probing_metrics_result,
        _compute_map_result,
        _compute_recall_at_k_result,
        run_umap,
    )

    dim_cols = [column for column in frame.columns if column.startswith("dim_")]
    embeddings = frame[dim_cols].to_numpy()
    raw_labels = [record.metadata.get("label", "") for record in bundle.records]
    total = len(embeddings)

    sampled = list(range(total))
    excluded: list[int] = []
    if sample_size > 0 and total > sample_size:
        rng = np.random.default_rng(seed)
        sampled = np.sort(rng.choice(total, size=sample_size, replace=False)).tolist()
        excluded = sorted(set(range(total)) - set(sampled))
        print(
            f"[Info] Visualization subsample: {len(sampled)}/{total} samples (seed={seed})"
        )

    feats = embeddings[sampled]
    labels = np.array([raw_labels[index] for index in sampled])
    labelled_mask = np.array([label != "" for label in labels])
    labelled_feats = feats[labelled_mask]
    labelled = labels[labelled_mask]

    outcomes: dict = {}
    skipped_reason = None
    if len(labelled) == 0:
        skipped_reason = "no nonempty labels in the evaluated rows"
    elif len(np.unique(labelled)) < 2:
        skipped_reason = "fewer than two label classes in the evaluated rows"
    else:
        print(f"[Metrics] Computing metrics on {len(labelled)} labelled samples ...")
        # One optional metric family must never discard a finished extraction: a
        # failing family is recorded as failed and the others still run.
        families = (
            ("recall_at_k", _compute_recall_at_k_result),
            ("knn_accuracy", _compute_knn_accuracy_result),
            ("linear_probing", _compute_linear_probing_metrics_result),
            ("map", lambda x, y: {"mAP": _compute_map_result(x, y)}),
            ("clustering", _compute_clustering_metrics_result),
        )
        for family, compute in families:
            try:
                outcomes.update(compute(labelled_feats, labelled))
            except Exception as exc:  # noqa: BLE001 - optional evaluation
                print(f"[Warning] {family} metrics failed: {exc}")
                outcome = _failed_outcome(family, exc)
                outcomes[_FAMILY_KEY[family]] = outcome

    metrics_path = None
    if outcomes:
        metrics_path = out_dir / "metrics.csv"
        rows = [
            {"metric": name, "value": "" if outcome.value is None else outcome.value}
            for name, outcome in outcomes.items()
        ]
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
        print(f"[Metrics] Results saved to: {metrics_path}")
        for name, outcome in outcomes.items():
            if outcome.value is not None:
                print(f"  {name}: {float(outcome.value):.4f}")

    umap_path = None
    if not enable_visualization:
        visualization = {
            "status": "not_applicable",
            "reason": "no CSV selector; UMAP is only generated for CSV-selected runs",
        }
    elif disable_umap:
        visualization = {"status": "not_applicable", "reason": "disabled by --disable-umap"}
    elif len(feats) < 10:
        visualization = {
            "status": "not_applicable",
            "reason": "fewer than 10 samples after sampling",
        }
        print("[Info] Skipping UMAP: fewer than 10 samples after sampling.")
    else:
        umap_path = out_dir / "umap.pdf"
        print(f"[UMAP] Generating UMAP projection -> {umap_path}")
        try:
            run_umap(
                feats,
                labels if bool(labelled_mask.any()) else None,
                umap_path,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                metric=umap_metric,
                max_classes=visualize_class_number,
            )
            visualization = {"status": "written", "path": umap_path.name}
        except Exception as exc:  # noqa: BLE001 - visualization is optional
            print(f"[Warning] UMAP failed: {exc}")
            umap_path = None
            visualization = {"status": "failed", "reason": str(exc)[:200]}

    from otuformer.data.provenance import serialize_metric_outcomes

    evaluation = serialize_metric_outcomes(outcomes)
    evaluation["samples_considered"] = total
    evaluation["sampled_rows"] = len(sampled)
    evaluation["evaluated_rows"] = int(labelled_mask.sum())
    evaluation["evaluated_image_keys"] = [
        bundle.records[index].image_key for index in np.array(sampled)[labelled_mask]
    ]
    evaluation["unlabeled_image_keys"] = [
        bundle.records[index].image_key for index in np.array(sampled)[~labelled_mask]
    ]
    evaluation["excluded_by_sampling"] = [
        bundle.records[index].image_key for index in excluded
    ]
    evaluation["seed"] = int(seed)
    if skipped_reason is not None:
        evaluation["skipped"] = {"status": "not_applicable", "reason": skipped_reason}
    evaluation["visualization"] = {"umap": visualization}
    return metrics_path, evaluation


def _pool_candidate_paths(out_dir: Path, settings: dict) -> list[Path]:
    """Attention-pool files this run could load instead of writing its own.

    The loader searches the output directory *and* the backbone checkpoint's
    directory, with tagged variants of both, so any existing pooling checkpoint in
    either place is a reusable input. Enumerated by pattern because the tagged name
    depends on a backbone that is not loaded yet at protection time.
    """
    pooling_type = settings["attention_pooling_type"]
    checkpoint = settings["checkpoint"]
    directories = [out_dir / "checkpoints"]
    if checkpoint is not None:
        directories.append(Path(checkpoint).expanduser().parent)
    candidates: list[Path] = []
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(f"*{pooling_type}_pooling.pth")):
            if not path.is_file():
                continue
            for entry in (path, path.resolve()):
                if entry not in candidates:
                    candidates.append(entry)
    return candidates


def _onnx_input_paths(onnx_path: Path) -> list[Path]:
    """The graph plus the external tensor files it currently references."""
    from otuformer.embedding.extractor import _onnx_external_files

    paths = [Path(onnx_path)]
    paths.extend(_onnx_external_files(Path(onnx_path)))
    return [path for path in paths if path.exists()]


def _run_new_data_path(*, settings: dict, inputs: dict[str, Path], params: dict) -> None:
    """Orchestrate the shared data path: safety, validation, extraction, publication."""

    import shutil

    from otuformer.cli_schema import (
        check_output_safety,
        default_protected_paths,
        input_identities,
        verify_inputs_unchanged,
    )
    from otuformer.data.provenance import (
        build_data_records,
        build_run_record,
        build_validation_record,
    )
    from otuformer.data.reader import read_dataset
    from otuformer.data.records import DataBundle
    from otuformer.embedding.extractor import _extract_embeddings_result
    from otuformer.utils.io import prepare_output_dir, write_csv, write_json_atomic
    from otuformer.utils.logging import TeeLogger

    out_dir = Path(settings["out_dir"])
    images_dir = Path(settings["input_images_dir"])
    started_at = _utc_now()
    params = dict(params)
    params["data_path"] = "shared_records"
    params["csv_parsing"] = (
        "strict" if (settings.get("manifest") or settings.get("label_csv")) else "none"
    )

    # Every file this run reads is protected before anything can be cleared,
    # including inputs derived from an explicitly passed path.
    protected = [*default_protected_paths(), images_dir, *inputs.values()]
    for key in ("checkpoint", "onnx_path"):
        value = settings.get(key)
        if value is not None:
            protected.append(Path(value))
    if settings.get("onnx_path") is not None:
        protected.extend(_onnx_input_paths(Path(settings["onnx_path"])))
    if settings["token_mode"] == "attention-pool":
        protected.extend(_pool_candidate_paths(out_dir, settings))
    # Snapshot the inputs before the check, so the baseline is the state the check
    # judged rather than whatever it looks like afterwards.
    identities = input_identities(protected)
    try:
        conditions = check_output_safety(
            out_dir, protected_paths=protected, overwrite=settings["overwrite"]
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None
    for condition in conditions:
        print(f"[Info] {condition}")

    # The check above is decided before anything is cleared. Whatever a cleanup did
    # to those paths -- including a directory entry swapped by another process in
    # between -- has to be visible before the run continues.
    prepare_output_dir(out_dir, overwrite=settings["overwrite"])
    try:
        verify_inputs_unchanged(identities)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None

    tee = TeeLogger(out_dir / "logs" / "extract.log")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = tee
    sys.stderr = tee

    run_path = out_dir / "run_provenance.json"
    staging = out_dir / ".staging"
    created: list[Path] = []
    empty_bundle = DataBundle(images_dir=images_dir, records=())
    # Latest real state; a later failure must keep it rather than claim nothing ran.
    state: dict = {"bundle": empty_bundle, "execution": None, "evaluation": None}

    def record(bundle, status, *, execution=None, evaluation=None, artifacts=(), failure=None):
        return build_run_record(
            requested=params,
            bundle=bundle,
            execution=execution,
            evaluation=evaluation or {},
            status=status,
            started_at=started_at,
            ended_at=None if status == "running" else _utc_now(),
            artifacts=list(artifacts),
            failure=failure,
        )

    try:
        print(
            f"[Info] data path: shared_records (strict CSV parsing: "
            f"{params['csv_parsing']})"
        )
        print(f"Command: {settings['cli_command']}")
        print("Parameters:")
        print(json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True))
        print("-" * 80)
        write_json_atomic(record(empty_bundle, "running"), run_path)
        created.append(run_path)

        bundle = read_dataset(
            images_dir,
            manifest=inputs.get("manifest"),
            coco_instances=inputs.get("coco_instances"),
            coco_keypoints=inputs.get("coco_keypoints"),
            coco_panoptic=inputs.get("coco_panoptic"),
            panoptic_masks_dir=inputs.get("panoptic_masks_dir"),
        )
        state["bundle"] = bundle
        validation_path = out_dir / "data_validation.json"
        write_json_atomic(build_validation_record(bundle), validation_path)
        created.append(validation_path)
        print(
            f"[Info] Selected {len(bundle.records)} image(s); "
            f"selection rule: {bundle.selection_rule}"
        )
        if not bundle.validation.valid:
            errors = bundle.validation.by_severity("error")
            print(f"[Error] data validation failed with {len(errors)} error(s):")
            for issue in errors[:10]:
                print(f"  - {issue.code}: {issue.message}")
            failure = {
                "status": "failed",
                "kind": "data_validation",
                "error_count": len(errors),
                "codes": sorted({issue.code for issue in errors}),
            }
            write_json_atomic(
                record(
                    bundle,
                    "failed",
                    failure=failure,
                    artifacts=[_artifact_entry(out_dir, validation_path)],
                ),
                run_path,
            )
            raise _RecordedFailure()

        pool_path = out_dir / "checkpoints" / "model.pooling.pth"
        if settings["checkpoint"] is not None:
            pool_path = (
                out_dir
                / "checkpoints"
                / f"{Path(settings['checkpoint']).stem}."
                f"{settings['attention_pooling_type']}_pooling.pth"
            )
        result = _extract_embeddings_result(
            checkpoint_path=settings["checkpoint"],
            images_dir=images_dir,
            model_name=settings["model_name"],
            extract_size=settings["extract_size"],
            batch_size=settings["batch_size"],
            device=settings["device"],
            num_workers=settings["num_workers"],
            use_projector_output=settings["use_projector_output"],
            use_student=settings["use_student"],
            token_mode=settings["token_mode"],
            topk_patches=settings["topk_patches"],
            attention_pooling_type=settings["attention_pooling_type"],
            attention_pooling_epochs=settings["attention_pooling_epochs"],
            attention_train_csv=None,
            seed=settings["seed"],
            extract_csv=None,
            attention_pooling_checkpoint_path=pool_path,
            onnx_path=settings["onnx_path"],
            eval_transform=settings["eval_transform"],
            records=bundle,
        )
        frame = result.frame
        state["execution"] = result
        # A pooling checkpoint written by this run is an output, not an input. It is
        # registered as created (so a failure removes it) and published in the final
        # provenance, which is the only record of which reusable weights this run
        # produced. A checkpoint reused from disk is never added here.
        written_pool = (result.effective.get("attention_pooling") or {}).get(
            "written_file"
        )
        pooling_created = bool(written_pool) and Path(written_pool).is_file()
        if pooling_created:
            created.append(Path(written_pool))

        metrics_path, evaluation = _evaluate_records(
            bundle,
            frame,
            sample_size=settings["metrics_sample_size"],
            seed=settings["seed"],
            # Metrics/UMAP are CSV-selected behaviour; COCO-only input never opts in.
            enable_visualization=(
                settings.get("manifest") is not None
                or settings.get("label_csv") is not None
            ),
            disable_umap=settings["disable_umap"],
            out_dir=out_dir,
            umap_n_neighbors=settings["umap_n_neighbors"],
            umap_min_dist=settings["umap_min_dist"],
            umap_metric=settings["umap_metric"],
            visualize_class_number=settings["visualize_class_number"],
        )
        state["evaluation"] = evaluation
        for produced in (metrics_path, out_dir / "umap.pdf"):
            if produced is not None and Path(produced).exists():
                created.append(Path(produced))

        # Stage both final files, check their pairing, then promote together.
        staging.mkdir(parents=True, exist_ok=True)
        staged_embeddings = staging / "embeddings.csv"
        staged_mapping = staging / "data_records.jsonl"
        write_csv(frame, staged_embeddings)
        mapping = build_data_records(bundle, frame)
        with staged_mapping.open("w", encoding="utf-8") as handle:
            for row in mapping:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        embeddings_path = out_dir / "embeddings.csv"
        mapping_path = out_dir / "data_records.jsonl"
        os.replace(staged_embeddings, embeddings_path)
        created.append(embeddings_path)
        os.replace(staged_mapping, mapping_path)
        created.append(mapping_path)
        shutil.rmtree(staging, ignore_errors=True)

        final_files = [
            path
            for path in (
                embeddings_path,
                mapping_path,
                validation_path,
                metrics_path,
                out_dir / "umap.pdf" if Path(out_dir / "umap.pdf").exists() else None,
                Path(written_pool) if pooling_created else None,
            )
            if path is not None and Path(path).exists()
        ]
        artifacts = [
            _artifact_entry(
                out_dir,
                path,
                rows=len(frame) if path in (embeddings_path, mapping_path) else None,
            )
            for path in final_files
        ]
        # Close the log before publishing "completed": a log failure must be able
        # to fail the run, and the record must not outlive the final state.
        sys.stdout, sys.stderr = original_stdout, original_stderr
        tee.close()
        write_json_atomic(
            record(
                bundle,
                "completed",
                execution=result,
                evaluation=evaluation,
                artifacts=artifacts,
            ),
            run_path,
        )
        print(f"Embeddings saved to: {embeddings_path} ({len(frame)} samples)")
        print(f"Row mapping saved to: {mapping_path}")
        print(f"Run provenance saved to: {run_path}")
        print(f"Validation report saved to: {validation_path}")
        return
    except BaseException as error:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        if isinstance(error, _RecordedFailure):
            _cleanup_created(created)
            raise typer.Exit(code=1) from None
        interrupted = isinstance(error, KeyboardInterrupt)
        status = "interrupted" if interrupted else "failed"
        failure = {
            "status": status,
            "kind": type(error).__name__,
            "message": str(error)[:500],
        }
        try:
            # Keep what actually ran: an empty bundle here would claim the
            # extraction never happened.
            write_json_atomic(
                record(
                    state["bundle"],
                    status,
                    execution=state["execution"],
                    evaluation=state["evaluation"],
                    failure=failure,
                ),
                run_path,
            )
        except Exception as record_error:  # pragma: no cover - defensive
            print(f"[Error] could not write the {status} run record: {record_error}")
        _cleanup_created(created)
        if not isinstance(error, typer.Exit):
            print(f"[Error] {status}: {error}")
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        sys.stdout, sys.stderr = original_stdout, original_stderr
        try:
            tee.close()
        except Exception as close_error:  # pragma: no cover - defensive
            print(f"[Warning] log close failed: {close_error}")


def _cleanup_created(paths: list[Path], *, keep_provenance: bool = True) -> None:
    """Remove only destinations created by this invocation.

    Best effort per entry: one failing removal must not stop the rest of the
    rollback, and must never replace the error that triggered it.
    """
    failed: list[Path] = []
    for path in paths:
        if keep_provenance and path.name in {"run_provenance.json", "data_validation.json"}:
            continue
        try:
            Path(path).unlink(missing_ok=True)
        except OSError as error:
            failed.append(Path(path))
            print(f"[Warning] could not remove {path}: {error}")
    if failed:
        print(
            f"[Warning] rollback left {len(failed)} file(s) in place: "
            + ", ".join(str(path) for path in failed)
        )
