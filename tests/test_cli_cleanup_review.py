"""Regression checks for cleanup review feedback."""

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from otuformer.cli.main import app


@pytest.mark.parametrize("with_embeddings", [False, True])
@pytest.mark.parametrize("layout", ["short", "nonstandard", "missing_log"])
def test_annotate_invalid_context_preserves_output(tmp_path, monkeypatch, with_embeddings, layout):
    monkeypatch.chdir(tmp_path)
    raw = Path("raw.csv" if layout == "short" else "partition_0.1_assignments.csv")
    if layout == "missing_log":
        raw = Path("cluster/UPGMA/partitions/tables") / raw
        raw.parent.mkdir(parents=True)
    raw.write_text("id,cluster\na,A\n")
    Path("corrections.csv").write_text("id,cluster\na,B\n")
    out = Path("out")
    out.mkdir()
    sentinel = out / "old.txt"
    sentinel.write_text("keep")
    argv = ["annotate", "--raw-assignments", str(raw), "--corrections", "corrections.csv", "--out-dir", str(out), "--overwrite"]
    if with_embeddings:
        Path("embeddings.csv").write_text("id,dim_0\na,1\n")
        argv += ["--embeddings", "embeddings.csv"]
    result = CliRunner().invoke(app, argv)
    assert sentinel.exists(), (result.output, repr(result.exception))
    assert sentinel.read_text() == "keep"
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert isinstance(result.exception, SystemExit)
    assert sorted(p.name for p in out.iterdir()) == ["old.txt"]


@pytest.mark.parametrize("phylo", [False, True])
@pytest.mark.parametrize("save_tree", [False, True])
def test_diversity_protects_embeddings_only_when_read(tmp_path, phylo, save_tree):
    assignments = tmp_path / "assignments.csv"
    assignments.write_text("id,cluster\na,A\nb,B\n")
    out = tmp_path / "out"
    out.mkdir()
    embeddings = out / "embeddings.csv"
    embeddings.write_text("id,dim_0,dim_1\na,1,0\nb,0,1\n")
    argv = ["diversity", "--assignments", str(assignments), "--embeddings", str(embeddings), "--min-abundance", "1", "--out-dir", str(out), "--overwrite"]
    if phylo:
        argv.append("--phylo")
    if save_tree:
        argv.append("--save-nj-tree")
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == (2 if phylo else 0), (result.output, repr(result.exception))
    assert embeddings.exists() == phylo
    assert (out / "diversity_indices.csv").exists() == (not phylo)


def test_diversity_builds_phylogeny_without_saving_tree(tmp_path):
    assignments = tmp_path / "assignments.csv"
    assignments.write_text("id,cluster\na,A\nb,B\nc,C\n")
    embeddings = tmp_path / "embeddings.csv"
    pd.DataFrame({"id": ["a", "b", "c"], "dim_0": [1., 0., -1.], "dim_1": [0., 1., 0.]}).to_csv(embeddings, index=False)
    out = tmp_path / "out"
    out.mkdir()
    ignored_tree = out / "unused.nwk"
    ignored_tree.write_text("not a valid tree")
    result = CliRunner().invoke(app, ["diversity", "--assignments", str(assignments), "--embeddings", str(embeddings), "--tree", str(ignored_tree), "--phylo", "--min-abundance", "1", "--out-dir", str(out), "--overwrite"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    table = pd.read_csv(out / "diversity_indices.csv", index_col=0)
    assert pd.notna(table.loc["MPD", "min_abundance_1"])
    assert not (out / "NJ_OTU.nwk").exists()
    assert not ignored_tree.exists()


def test_cleanup_guards_reject_unverifiable_chains_consistently(tmp_path):
    from otuformer.cli_schema import check_output_safety
    from otuformer.utils.io import prepare_output_dir

    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("keep")
    entry = tmp_path / "loop"
    entry.symlink_to(entry)
    for guard in (prepare_output_dir, check_output_safety):
        with pytest.raises(ValueError, match="cannot verify output safety.*more than 40 symbolic links"):
            guard(out, overwrite=True, protected_paths=[entry])
        assert sentinel.read_text() == "keep"


def test_schema_traverses_each_input_once(tmp_path, monkeypatch):
    from otuformer import cli_schema
    from otuformer.utils import paths

    out = tmp_path / "out"
    out.mkdir()
    source = tmp_path / "source.csv"
    source.write_text("image,label\na,A\n")
    entry = tmp_path / "entry.csv"
    entry.symlink_to(source)
    calls = []
    real_walk = paths.input_dependency_positions

    def observed_walk(path):
        calls.append(path)
        return real_walk(path)

    monkeypatch.setattr(cli_schema, "input_dependency_positions", observed_walk)
    cli_schema.check_output_safety(out, overwrite=True, protected_paths=[entry])
    assert calls == [entry]


@pytest.mark.parametrize("module_name", ["otuformer.utils.io", "otuformer.cli_schema"])
def test_modules_have_one_module_level_path_walker_import(module_name):
    import ast
    import importlib

    module = importlib.import_module(module_name)
    tree = ast.parse(Path(module.__file__).read_text())
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "otuformer.utils.paths"
        for alias in node.names
        if alias.name == "input_dependency_positions"
    ]
    assert len(imports) == 1
    assert imports[0] in tree.body


def test_io_cleanup_guard_retains_actionable_overwrite_error(tmp_path):
    from otuformer.utils.io import prepare_output_dir

    out = tmp_path / "out"
    out.mkdir()
    source = out / "input.csv"
    source.write_text("preserve input")
    with pytest.raises(ValueError) as error:
        prepare_output_dir(out, overwrite=True, protected_paths=[source])
    assert str(error.value) == (
        f"--overwrite would clear {out}, which holds the input {out}; "
        "choose another --out-dir or drop --overwrite."
    )
    assert source.read_text() == "preserve input"


@pytest.mark.parametrize("overwrite", [False, True])
def test_schema_rejects_output_below_protected_input_root(tmp_path, overwrite):
    from otuformer.cli_schema import check_output_safety

    root = tmp_path / "images"
    out = root / "run"
    out.mkdir(parents=True)
    sentinel = out / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(ValueError) as error:
        check_output_safety(out, protected_paths=[root], overwrite=overwrite)
    assert str(error.value) == (
        f"output directory {out.resolve()} is inside protected input {root.resolve()}"
    )
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("layout", ["direct", "relay", "root"])
def test_schema_cleanup_dependency_error_is_precise(tmp_path, layout):
    from otuformer.cli_schema import check_output_safety

    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("keep")
    entry = sentinel if layout == "direct" else out
    if layout == "relay":
        source = tmp_path / "source.csv"
        source.write_text("image,label\na,A\n")
        relay = out / "relay.csv"
        relay.symlink_to(source)
        entry = tmp_path / "entry.csv"
        entry.symlink_to(relay)
    with pytest.raises(ValueError) as error:
        check_output_safety(out, overwrite=True, protected_paths=[entry])
    assert str(error.value) == (
        f"output directory {out.resolve()} contains protected input "
        f"dependency {out.resolve()}, which {entry} is reached through"
    )
    assert sentinel.read_text() == "keep"
