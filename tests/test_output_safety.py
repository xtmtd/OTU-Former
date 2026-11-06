"""Cleanup must preserve every input dependency before any output is changed."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from typer.testing import CliRunner

from otuformer.cli.main import app
from otuformer.utils.io import prepare_output_dir


@pytest.mark.parametrize("layout", ["direct", "link", "chain", "directory_target", "dotdot", "target_dotdot", "cycle", "long_chain"])
def test_cleanup_preserves_input_dependencies(tmp_path, layout):
    out = tmp_path / "out"
    out.mkdir()
    keep = out / "keep.txt"
    keep.write_text("keep")
    images = tmp_path / "images"
    images.mkdir()
    pixels = tmp_path / "pixels"
    pixels.mkdir()
    real = (out if layout in {"direct", "link", "dotdot", "target_dotdot"} else pixels) / "real.png"
    real.write_bytes(b"pixels")
    entry = images / "entry.png"
    if layout == "direct":
        entry = real
    elif layout == "link":
        entry.symlink_to(real)
    elif layout == "chain":
        (out / "middle.png").symlink_to(real)
        entry.symlink_to(out / "middle.png")
    elif layout == "directory_target":
        (out / "middle").symlink_to(pixels, target_is_directory=True)
        (tmp_path / "bridge").symlink_to(out / "middle", target_is_directory=True)
        entry.symlink_to(tmp_path / "bridge" / "real.png")
    elif layout in {"dotdot", "target_dotdot"}:
        (out / "sub").mkdir()
        (images / "jump").symlink_to(out / "sub", target_is_directory=True)
        written = images / "jump" / ".." / "real.png"
        if layout == "dotdot":
            entry = written
        else:
            entry.symlink_to(written)
    else:
        links = [images / f"link_{i}" for i in range(65)]
        for link, target in zip(links, links[1:]):
            link.symlink_to(target)
        links[-1].symlink_to(links[0] if layout == "cycle" else real)
        entry = links[0]

    with pytest.raises(ValueError):
        prepare_output_dir(out, overwrite=True, protected_paths=[entry])
    assert keep.read_text() == "keep"
    assert real.read_bytes() == b"pixels"


def test_cleanup_allows_output_below_input_root(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    real = images / "real.png"
    real.write_bytes(b"pixels")
    out = images / "run"
    out.mkdir()
    (out / "old.txt").write_text("old")
    prepare_output_dir(out, overwrite=True, protected_paths=[images, real])
    assert real.read_bytes() == b"pixels"
    assert not (out / "old.txt").exists()


@pytest.fixture
def checkpoint(tmp_path):
    from otuformer.training.model import OTUFormerEncoder

    model = OTUFormerEncoder(out_dim=16, pretrained=False, img_size=32)
    source = tmp_path / "source.pth"
    torch.save({"model_state_dict": model.state_dict(), "config": {"model_name": "vit_tiny_patch16_224", "out_dim": 16, "image_size": 32}}, source)
    return source


@pytest.mark.parametrize("reference", ["absolute", "symlink", "chain", "directory_target", "dotdot"])
def test_extract_refuses_to_delete_selected_images(tmp_path, checkpoint, reference):
    images = tmp_path / "images"
    images.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    keep = out / "keep.txt"
    keep.write_text("keep")
    real = out / "real.png"
    Image.new("RGB", (32, 32)).save(real)
    Image.new("RGB", (32, 32)).save(images / "safe.png")
    entry = images / "entry.png"
    if reference == "absolute":
        ref = str(real)
    elif reference == "dotdot":
        (out / "sub").mkdir()
        (images / "jump").symlink_to(out / "sub", target_is_directory=True)
        ref = "jump/../real.png"
    else:
        if reference == "chain":
            (out / "middle.png").symlink_to(real)
            entry.symlink_to(out / "middle.png")
        elif reference == "directory_target":
            pixels = tmp_path / "pixels"
            pixels.mkdir()
            real.rename(pixels / "real.png")
            real = pixels / "real.png"
            (out / "middle").symlink_to(pixels, target_is_directory=True)
            (tmp_path / "bridge").symlink_to(out / "middle", target_is_directory=True)
            entry.symlink_to(tmp_path / "bridge" / "real.png")
        else:
            entry.symlink_to(real)
        ref = "entry.png"
    manifest = tmp_path / "labels.csv"
    pd.DataFrame({"image": [ref], "label": ["A"]}).to_csv(manifest, index=False)
    argv = ["extract", "--checkpoint", str(checkpoint), "--input-images-dir", str(images), "--out-dir", str(out), "--overwrite", "--disable-umap", "--device", "cpu", "--num-workers", "0"]
    if reference in {"absolute", "dotdot"}:
        argv += ["--label-csv", str(manifest)]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 2, (repr(result.exception), result.output)
    assert keep.read_text() == "keep"
    assert real.exists()


def test_extract_missing_model_preserves_old_output(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    keep = out / "keep.txt"
    keep.write_text("keep")
    result = CliRunner().invoke(app, ["extract", "--input-images-dir", str(images), "--out-dir", str(out), "--overwrite"])
    assert result.exit_code == 2
    assert "checkpoint" in result.output
    assert keep.read_text() == "keep"


@pytest.mark.parametrize("command,options,name", [
    ("export", ["--checkpoint"], "source.pth"),
    ("cluster", ["--embeddings"], "embeddings.csv"),
    ("annotate", ["--raw-assignments", "--corrections"], "assignments.csv"),
    ("diversity", ["--assignments"], "assignments.csv"),
    ("cam", ["--checkpoint"], "source.pth"),
])
def test_downstream_inputs_survive_overwrite(tmp_path, checkpoint, command, options, name):
    out = tmp_path / "out"
    out.mkdir()
    source = out / name
    source.write_bytes(checkpoint.read_bytes() if name.endswith(".pth") else b"image,cluster,sample\na.png,A,S1\n")
    argv = [command, "--out-dir", str(out), "--overwrite"]
    for option in options:
        argv += [option, str(source)]
    if command == "cam":
        images = tmp_path / "images"
        images.mkdir()
        argv += ["--images-dir", str(images), "--device", "cpu"]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 2, (repr(result.exception), result.output)
    assert source.exists()
    assert any(message in " ".join(result.output.replace("│", " ").split()) for message in ("holds the input", "contains protected input", "is reached through"))


@pytest.mark.parametrize("command", ["pretrain", "finetune"])
@pytest.mark.parametrize("manifest_kind", ["training", "validation"])
def test_training_selected_images_survive_cleanup(tmp_path, checkpoint, monkeypatch, command, manifest_kind):
    from otuformer.training import trainer

    monkeypatch.setattr(trainer, f"run_{command}", lambda *args, **kwargs: None)
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (32, 32)).save(images / "safe.png")
    out = tmp_path / "out"
    out.mkdir()
    real = out / "pixels.png"
    Image.new("RGB", (32, 32)).save(real)
    (images / "linked.png").symlink_to(real)
    train = tmp_path / "train.csv"
    pd.DataFrame({"image": ["linked.png" if manifest_kind == "training" else "safe.png"], "label": ["A"]}).to_csv(train, index=False)
    validation = tmp_path / "validation.csv"
    pd.DataFrame({"image": ["linked.png"], "label": ["A"]}).to_csv(validation, index=False)
    argv = [command, "--input-images-dir", str(images), "--train-data", str(train), "--out-dir", str(out), "--overwrite", "--device", "cpu"]
    if command == "finetune":
        argv += ["--checkpoint", str(checkpoint)]
    if manifest_kind == "validation":
        argv += ["--visualize-data", str(validation)]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert any(message in " ".join(result.output.replace("│", " ").split()) for message in ("holds the input", "contains protected input", "is reached through"))
    assert real.exists()
    if manifest_kind == "validation":
        result = CliRunner().invoke(app, argv + ["--disable-embedding-metrics"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert not real.exists()


@pytest.mark.parametrize("csv_selection", [False, True])
def test_extract_allows_safe_nested_output_and_respects_selection(tmp_path, checkpoint, csv_selection):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (32, 32)).save(images / "safe.png")
    # Strict records retain their existing non-overlapping output policy.
    out = tmp_path / "run" if csv_selection else images / "run"
    out.mkdir()
    old = out / "old.txt"
    old.write_text("old")
    argv = ["extract", "--checkpoint", str(checkpoint), "--input-images-dir", str(images), "--out-dir", str(out), "--overwrite", "--disable-umap", "--device", "cpu", "--num-workers", "0"]
    if csv_selection:
        Image.new("RGB", (32, 32)).save(out / "unselected.png")
        (images / "unselected.png").symlink_to(out / "unselected.png")
        selected = tmp_path / "selected.csv"
        pd.DataFrame({"image": ["safe.png"]}).to_csv(selected, index=False)
        argv += ["--label-csv", str(selected)]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert len(pd.read_csv(out / "embeddings.csv")) == 1
    assert not old.exists()
    assert (images / "safe.png").is_file()


def test_extract_protects_a_batch_directory_link_target(tmp_path, checkpoint):
    images = tmp_path / "images"
    images.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    pixels = tmp_path / "pixels"
    pixels.mkdir()
    Image.new("RGB", (32, 32)).save(pixels / "real.png")
    (out / "middle").symlink_to(pixels, target_is_directory=True)
    (images / "batch").symlink_to(out / "middle", target_is_directory=True)
    result = CliRunner().invoke(app, ["extract", "--checkpoint", str(checkpoint), "--input-images-dir", str(images), "--out-dir", str(out), "--overwrite"])
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert any(message in " ".join(result.output.replace("│", " ").split()) for message in ("holds the input", "contains protected input", "is reached through")), result.output
    assert (out / "middle").is_symlink()


@pytest.mark.parametrize("tagged", [False, True])
def test_extract_protects_reusable_attention_pool(tmp_path, checkpoint, tagged):
    images = tmp_path / "images"
    images.mkdir()
    out = tmp_path / "out"
    cache_dir = out / "checkpoints"
    cache_dir.mkdir(parents=True)
    cache = cache_dir / ("source.prefix1-reg0-lightweight_pooling.pth" if tagged else "source.lightweight_pooling.pth")
    cache.write_bytes(b"existing cached pool")
    result = CliRunner().invoke(app, ["extract", "--checkpoint", str(checkpoint), "--input-images-dir", str(images), "--out-dir", str(out), "--overwrite", "--token-mode", "attention-pool"])
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert any(message in " ".join(result.output.replace("│", " ").split()) for message in ("holds the input", "contains protected input", "is reached through"))
    assert cache.read_bytes() == b"existing cached pool"


def test_cam_protects_actual_selected_images(tmp_path, checkpoint):
    images = tmp_path / "images"
    images.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    real = out / "real.png"
    Image.new("RGB", (32, 32)).save(real)
    (images / "linked.png").symlink_to(real)
    result = CliRunner().invoke(app, ["cam", "--checkpoint", str(checkpoint), "--images-dir", str(images), "--out-dir", str(out), "--overwrite"])
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert any(message in " ".join(result.output.replace("│", " ").split()) for message in ("holds the input", "contains protected input", "is reached through"))
    assert real.exists()


@pytest.mark.parametrize("use_embeddings", [False, True])
def test_annotate_protects_inferred_log_input(tmp_path, use_embeddings):
    cluster = tmp_path / "cluster"
    tables = cluster / "UPGMA" / "partitions" / "tables"
    tables.mkdir(parents=True)
    raw = tables / "partition_0.1_assignments.csv"
    raw.write_text("id,cluster\na,A\n")
    corrections = tmp_path / "corrections.csv"
    corrections.write_text("id,cluster\na,A\n")
    out = tmp_path / "out"
    out.mkdir()
    log = out / "cluster.log"
    log.write_text("original log")
    (cluster / "logs").mkdir()
    (cluster / "logs" / "cluster.log").symlink_to(log)
    argv = ["annotate", "--raw-assignments", str(raw), "--corrections", str(corrections), "--out-dir", str(out), "--overwrite"]
    if use_embeddings:
        embeddings = tmp_path / "embeddings.csv"
        embeddings.write_text("id,dim_0\na,1\n")
        argv += ["--embeddings", str(embeddings)]
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert any(message in " ".join(result.output.replace("│", " ").split()) for message in ("holds the input", "contains protected input", "is reached through"))
    assert log.read_text() == "original log"


@pytest.mark.parametrize("kind", ["initializer", "constant", "subgraph", "sparse", "function"])
def test_extract_protects_every_onnx_external_tensor(tmp_path, kind):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    images = tmp_path / "images"
    images.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    payload = out / "weights.bin"
    payload.write_bytes(np.array([1.0], dtype=np.float32).tobytes())
    tensor = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="weights")
    tensor.ClearField("raw_data")
    tensor.data_location = TensorProto.EXTERNAL
    entry = tensor.external_data.add()
    entry.key, entry.value = "location", "out/weights.bin"
    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])]
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 32, 32])
    graph = helper.make_graph([], "test", inputs, [output])
    if kind == "initializer":
        graph.initializer.append(tensor)
    elif kind == "constant":
        graph.node.append(helper.make_node("Constant", [], ["weights"], value=tensor))
    elif kind == "subgraph":
        nested = helper.make_graph([], "nested", [], [], initializer=[tensor])
        graph.node.append(helper.make_node("If", ["condition"], ["branch"], then_branch=nested, else_branch=nested))
    elif kind == "sparse":
        indices = numpy_helper.from_array(np.array([0], dtype=np.int64))
        graph.sparse_initializer.append(helper.make_sparse_tensor(tensor, indices, [1]))
    model = helper.make_model(graph)
    if kind == "function":
        function = model.functions.add()
        function.name, function.domain = "Custom", "test"
        function.attribute_proto.append(helper.make_attribute("default_weight", tensor))
    model_path = tmp_path / "model.onnx"
    onnx.save_model(model, str(model_path))
    result = CliRunner().invoke(app, ["extract", "--onnx-path", str(model_path), "--input-images-dir", str(images), "--out-dir", str(out), "--overwrite"])
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert any(message in " ".join(result.output.replace("│", " ").split()) for message in ("holds the input", "contains protected input", "is reached through"))
    assert payload.exists()


def test_extract_reuses_pool_with_image_only_selector(tmp_path, checkpoint):
    import shutil

    images = tmp_path / "images"
    images.mkdir()
    for name in ("a", "b"):
        Image.new("RGB", (32, 32)).save(images / f"{name}.png")
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"image": ["a.png", "b.png"], "label": ["A", "B"]}).to_csv(labels, index=False)
    base = ["extract", "--checkpoint", str(checkpoint), "--input-images-dir", str(images), "--device", "cpu", "--num-workers", "0", "--batch-size", "2", "--disable-umap", "--token-mode", "attention-pool", "--attention-pooling-epochs", "1"]
    first = tmp_path / "first"
    result = CliRunner().invoke(app, base + ["--label-csv", str(labels), "--out-dir", str(first)])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    cache = next((first / "checkpoints").glob("*.pth"))
    shutil.copy2(cache, checkpoint.parent / cache.name)
    pd.DataFrame({"image": ["a.png", "b.png"]}).to_csv(labels, index=False)
    second = tmp_path / "second"
    result = CliRunner().invoke(app, base + ["--label-csv", str(labels), "--out-dir", str(second)])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert len(pd.read_csv(second / "embeddings.csv")) == 2


def test_schema_refuses_input_reached_through_an_output_entry(tmp_path):
    from otuformer.cli_schema import check_output_safety

    out = tmp_path / "out"
    out.mkdir()
    target = tmp_path / "source.csv"
    target.write_text("image,label\na,A\n")
    (out / "middle.csv").symlink_to(target)
    entry = tmp_path / "entry.csv"
    entry.symlink_to(out / "middle.csv")
    with pytest.raises(ValueError):
        check_output_safety(out, protected_paths=[entry], overwrite=True)
    assert (out / "middle.csv").is_symlink()


def test_finetune_protects_accepted_pseudo_images(tmp_path, checkpoint, monkeypatch):
    from otuformer.training import trainer

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (32, 32)).save(images / "expert.png")
    out = tmp_path / "out"
    out.mkdir()
    real = out / "pseudo.png"
    Image.new("RGB", (32, 32)).save(real)
    (images / "pseudo.png").symlink_to(real)
    train = tmp_path / "train.csv"
    pd.DataFrame({"image": ["expert.png"], "label": ["A"]}).to_csv(train, index=False)
    rows = [{"image": "expert.png", "label": "A", "source": "expert"}, {"image": "pseudo.png", "label": "A", "source": "known-pseudo"}]
    monkeypatch.setattr(trainer, "prepare_pseudo_round", lambda **kwargs: {
        "source_config": {"loss": "arcface", "long_tail": "none"},
        "source_class_labels": ["A"], "ssl_checkpoint": str(checkpoint),
        "rows": rows, "accepted": rows[1:], "diagnostics": [], "summary": {},
        "class_labels": ["A"], "source_checkpoint": str(checkpoint),
    })
    monkeypatch.setattr(trainer, "run_finetune", lambda *args, **kwargs: None)
    result = CliRunner().invoke(app, ["finetune", "--pseudo-label-from", str(checkpoint), "--train-data", str(train), "--input-images-dir", str(images), "--out-dir", str(out), "--overwrite"])
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert any(message in " ".join(result.output.replace("│", " ").split()) for message in ("holds the input", "contains protected input", "is reached through"))
    assert real.exists()
