from pathlib import Path

from click.testing import CliRunner

from pkasso import cli


class Mol:
    def GetProp(self, name):
        return {"_Name": "state0", "Probability": "1.0", "net_charge": "0"}[name]


class Scan:
    def __init__(self):
        self.exported_macro_pkas = None
        self.saved_sdf = None
        self.exported_scan = None

    def export_macro_pkas(self, file):
        self.exported_macro_pkas = file

    def print_macro_pkas(self):
        pass

    def plot_scan(self):
        return object()

    def plot_mols(self, size_x, size_y):
        return object()

    def export_scan(self, file, fig_scan, fig_mols):
        self.exported_scan = file

    def save_sdf(self, file):
        self.saved_sdf = file


def mock_protonate(captured):
    def protonate(*args, **kwargs):
        captured.update(kwargs)
        return ["C"], [Mol()]

    return protonate


def test_short_help_option_is_available_for_root_and_subcommands():
    runner = CliRunner()

    for args in (["-h"], ["single", "-h"], ["batch", "-h"], ["scan", "-h"]):
        result = runner.invoke(cli.cli, args)

        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "--help" in result.output


def test_txt_out_and_cutoff_export_are_common_options():
    runner = CliRunner()

    for command, cutoff_default in (("single", 0.2), ("batch", 1.0), ("scan", 0.2)):
        result = runner.invoke(cli.cli, [command, "--help"])

        assert result.exit_code == 0
        assert "--txt-out" in result.output
        assert "--cutoff-export" in result.output
        assert f"default: {cutoff_default}" in result.output


def test_uppercase_ph_alias_is_equivalent_for_single(monkeypatch):
    captured = {}

    monkeypatch.setattr(cli, "protonate", mock_protonate(captured))

    result = CliRunner().invoke(
        cli.cli,
        ["single", "--smiles", "C", "--pH", "5.5"],
    )

    assert result.exit_code == 0
    assert captured["pH"] == 5.5
    assert captured["cutoff_export"] == 0.2


def test_uppercase_ph_alias_is_equivalent_for_batch(monkeypatch, tmp_path):
    captured = {}
    smi = tmp_path / "mols.smi"
    smi.write_text("C mol\n", encoding="utf-8")

    monkeypatch.setattr(cli, "read_smi", lambda _path: {"mol": "C"})
    monkeypatch.setattr(cli, "protonate", mock_protonate(captured))
    monkeypatch.setattr(cli, "save_sdf", lambda *_args: None)

    result = CliRunner().invoke(
        cli.cli,
        ["batch", "--smi", str(smi), "--pH", "5.5"],
    )

    assert result.exit_code == 0
    assert captured["pH"] == 5.5
    assert captured["cutoff_export"] == 1.0


def test_repeated_option_is_rejected():
    result = CliRunner().invoke(
        cli.cli,
        ["single", "--smiles", "C", "--ph", "5", "--ph", "6"],
    )

    assert result.exit_code != 0
    assert "cannot be specified multiple times" in result.output


def test_mixed_aliases_are_rejected_as_repeated_option():
    result = CliRunner().invoke(
        cli.cli,
        ["single", "--smiles", "C", "--ph", "5", "--pH", "6"],
    )

    assert result.exit_code != 0
    assert "cannot be specified multiple times" in result.output


def test_repeated_name_is_rejected():
    result = CliRunner().invoke(
        cli.cli,
        ["single", "--smiles", "C", "--name", "first", "--name", "second"],
    )

    assert result.exit_code != 0
    assert "cannot be specified multiple times" in result.output


def test_max_tautomers_conflicts_with_no_tautomer_search():
    result = CliRunner().invoke(cli.cli, ["single", "--smiles", "C", "--max-tautomers", "5", "--no-tautomer-search"])

    assert result.exit_code != 0
    assert "--max-tautomers cannot be used with --no-tautomer-search." in result.output


def test_num_confs_conflicts_with_no_tautomer_search():
    result = CliRunner().invoke(cli.cli, ["single", "--smiles", "C", "--num-confs", "2", "--no-tautomer-search"])

    assert result.exit_code != 0
    assert "--num-confs cannot be used with --no-tautomer-search." in result.output


def test_common_option_conflicts_apply_to_batch_and_scan():
    runner = CliRunner()

    batch = runner.invoke(cli.cli, ["batch", "--smi", "mols.smi", "--no-tautomer-search", "--num-confs", "2"])
    scan = runner.invoke(cli.cli, ["scan", "--smiles", "C", "--no-tautomer-search", "--num-confs", "2"])

    assert batch.exit_code != 0
    assert "--num-confs cannot be used with --no-tautomer-search." in batch.output
    assert scan.exit_code != 0
    assert "--num-confs cannot be used with --no-tautomer-search." in scan.output


def test_max_tautomers_and_num_confs_can_be_used_together(monkeypatch):
    captured = {}

    monkeypatch.setattr(cli, "protonate", mock_protonate(captured))

    result = CliRunner().invoke(cli.cli, ["single", "--smiles", "C", "--max-tautomers", "5", "--num-confs", "2"])

    assert result.exit_code == 0
    assert captured["max_tautomers"] == 5
    assert captured["num_confs"] == 2


def test_single_forwards_name_and_sizes_output_to_columns(monkeypatch):
    captured = {}

    monkeypatch.setattr(cli, "protonate", mock_protonate(captured))

    result = CliRunner().invoke(
        cli.cli,
        ["single", "--smiles", "C", "--name", "actual_name"],
    )

    assert result.exit_code == 0
    assert captured["name"] == "actual_name"

    lines = result.output.splitlines()
    expected_width = len("Microstate") + len("SMILES") + 31
    assert lines[0] == "-" * expected_width
    assert lines[1] == "actual_name | pH: 7.0".center(expected_width)
    assert lines[3] == "-" * expected_width


def test_single_writes_microstate_table_to_txt_out(monkeypatch):
    monkeypatch.setattr(cli, "protonate", mock_protonate({}))

    with CliRunner().isolated_filesystem():
        result = CliRunner().invoke(
            cli.cli,
            ["single", "--smiles", "C", "--name", "single_name", "--txt-out", "overview.txt"],
        )
        txt_output = Path("overview.txt").read_text(encoding="utf-8")

    assert result.exit_code == 0
    assert txt_output == result.output
    assert "single_name | pH: 7.0" in txt_output
    assert "state0" in txt_output


def test_write_microstate_tables_uses_common_width(tmp_path):
    txt_out = tmp_path / "overview.txt"

    cli.write_microstate_tables(
        txt_out,
        [
            ("short", 7.0, ["C"], [Mol()]),
            ("a_much_longer_molecule_name", 7.5, ["CCCCCCCCCCCC"], [Mol()]),
        ],
    )

    separator_widths = {
        len(line)
        for line in txt_out.read_text(encoding="utf-8").splitlines()
        if line.startswith("-")
    }
    assert len(separator_widths) == 1


def test_batch_writes_all_microstate_tables_to_common_txt_out(monkeypatch):
    monkeypatch.setattr(cli, "read_smi", lambda _path: {"first": "C", "second": "N"})
    monkeypatch.setattr(cli, "protonate", mock_protonate({}))
    monkeypatch.setattr(cli, "save_sdf", lambda *_args: None)

    with CliRunner().isolated_filesystem():
        Path("mols.smi").write_text("C mol\n", encoding="utf-8")
        result = CliRunner().invoke(
            cli.cli,
            ["batch", "--smi", "mols.smi", "--ph", "6.5", "--txt-out", "overview.txt"],
        )
        txt_output = Path("overview.txt").read_text(encoding="utf-8")

    assert result.exit_code == 0
    assert result.output == txt_output
    assert "first | pH: 6.5" in txt_output
    assert "second | pH: 6.5" in txt_output
    assert txt_output.count("Microstate") == 2


def test_batch_writes_one_combined_sdf_by_default(monkeypatch):
    saved = []

    def protonate(*_args, **_kwargs):
        return ["C", "[CH3+]"], [Mol(), Mol()]

    monkeypatch.setattr(cli, "read_smi", lambda _path: {"first": "C", "second": "N"})
    monkeypatch.setattr(cli, "protonate", protonate)
    monkeypatch.setattr(cli, "save_sdf", lambda mols, path: saved.append((tuple(mols), path)))

    with CliRunner().isolated_filesystem():
        Path("mols.smi").write_text("C first\n", encoding="utf-8")
        result = CliRunner().invoke(cli.cli, ["batch", "--smi", "mols.smi"])
        individual_folder_exists = Path("batch_individual_sdfs").exists()

    assert result.exit_code == 0
    assert [(len(mols), path) for mols, path in saved] == [
        (4, Path("molecules_batch.sdf")),
    ]
    assert individual_folder_exists is False


def test_batch_individual_sdfs_use_safe_unique_filenames(monkeypatch):
    saved = []

    monkeypatch.setattr(cli, "read_smi", lambda _path: {"../escape": "C", "..?escape": "N"})
    monkeypatch.setattr(cli, "protonate", mock_protonate({}))
    monkeypatch.setattr(cli, "save_sdf", lambda mols, path: saved.append((mols, path)))

    with CliRunner().isolated_filesystem():
        Path("mols.smi").write_text("C first\n", encoding="utf-8")
        result = CliRunner().invoke(
            cli.cli,
            [
                "batch",
                "--smi",
                "mols.smi",
                "--individual-sdfs",
                "--sdf-combined",
                "all_microstates.sdf",
            ],
        )

    assert result.exit_code == 0
    assert [path for _, path in saved] == [
        Path("all_microstates.sdf"),
        Path("batch_individual_sdfs/0000_escape.sdf"),
        Path("batch_individual_sdfs/0001_escape.sdf"),
    ]


def test_batch_help_explains_combined_sdf_and_cutoff():
    output = CliRunner().invoke(cli.cli, ["batch", "--help"]).output

    assert "--individual-sdfs / --no-individual-sdfs" in output
    assert "molecules_batch.sdf" in output
    assert "all exported microstates for all input" in output
    assert "multiple records when multiple microstates" in output
    assert "--cutoff-export" in output


def test_output_paths_must_be_distinct(tmp_path):
    smi = tmp_path / "molecules.smi"
    smi.write_text("C molecule\n", encoding="utf-8")
    output = tmp_path / "same.out"

    result = CliRunner().invoke(
        cli.cli,
        [
            "batch",
            "--smi",
            str(smi),
            "--sdf-combined",
            str(output),
            "--txt-out",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert "--txt-out must not use the same path as --sdf-combined" in result.output


def test_scan_sanitizes_name_for_default_output_paths(monkeypatch):
    scan = Scan()
    monkeypatch.setattr(cli, "scan_pH", lambda *_args, **_kwargs: scan)

    result = CliRunner().invoke(
        cli.cli,
        ["scan", "--smiles", "C", "--name", "../escape"],
    )

    assert result.exit_code == 0
    assert scan.exported_macro_pkas == Path("escape_macro_pkas.out")
    assert scan.exported_scan == Path("escape_scan.svg")
    assert scan.saved_sdf == Path("escape_mols_scan.sdf")


def test_explicit_ignored_options_are_rejected(tmp_path):
    gpu = CliRunner().invoke(cli.cli, ["single", "--smiles", "C", "--gpu"])
    model_folder = CliRunner().invoke(
        cli.cli,
        ["single", "--smiles", "C", "--unipka-model-folder", str(tmp_path)],
    )
    path_out = CliRunner().invoke(
        cli.cli,
        ["batch", "--smi", "mols.smi", "--path-out", "individual"],
    )

    assert "--gpu requires --model mixed." in gpu.output
    assert "--unipka-model-folder requires --model mixed." in model_folder.output
    assert "--path-out requires --individual-sdfs." in path_out.output


def test_individual_sdf_filename_has_safe_length():
    filename = cli._safe_sdf_filename(0, "a" * 1000)

    assert len(filename) <= 255


def test_cutoff_states_must_be_at_least_one():
    result = CliRunner().invoke(cli.cli, ["single", "--smiles", "C", "--cutoff-states", "0"])

    assert result.exit_code != 0
    assert "--cutoff-states must be >= 1." in result.output


def test_cutoff_states_allows_one(monkeypatch):
    captured = {}

    monkeypatch.setattr(cli, "protonate", mock_protonate(captured))

    result = CliRunner().invoke(cli.cli, ["single", "--smiles", "C", "--cutoff-states", "1"])

    assert result.exit_code == 0
    assert captured["cutoff_states"] == 1


def test_default_model_leaves_model_selection_to_python_interface(monkeypatch):
    captured = {}

    monkeypatch.setattr(cli, "protonate", mock_protonate(captured))

    result = CliRunner().invoke(cli.cli, ["single", "--smiles", "C"])

    assert result.exit_code == 0
    assert "model" not in captured


def test_mixed_model_passes_fixed_fold_and_unipka_options(monkeypatch, tmp_path):
    captured = {}

    monkeypatch.setattr(cli, "protonate", mock_protonate(captured))

    result = CliRunner().invoke(
        cli.cli,
        [
            "single",
            "--smiles",
            "C",
            "--model",
            "mixed",
            "--unipka-model-folder",
            str(tmp_path),
            "--nthreads",
            "4",
            "--gpu",
        ],
    )

    assert result.exit_code == 0
    assert captured["model"] == {
        "molgpka": {},
        "unipka": {
            "folds": 1,
            "model_dir": tmp_path,
            "gpu": True,
        },
    }
    assert captured["nthreads"] == 4


def test_molgpka_model_receives_top_level_nthreads(monkeypatch):
    captured = {}

    monkeypatch.setattr(cli, "protonate", mock_protonate(captured))

    result = CliRunner().invoke(
        cli.cli,
        ["single", "--smiles", "C", "--model", "molgpka", "--nthreads", "3"],
    )

    assert result.exit_code == 0
    assert captured["nthreads"] == 3
    assert "model" not in captured


def test_mixed_model_uses_unipkainfer_default_model_dir_and_cpu(monkeypatch):
    captured = {}

    monkeypatch.setattr(cli, "protonate", mock_protonate(captured))

    result = CliRunner().invoke(cli.cli, ["single", "--smiles", "C", "--model", "mixed"])

    assert result.exit_code == 0
    assert captured["model"]["unipka"] == {
        "folds": 1,
        "gpu": False,
    }
    assert captured["nthreads"] == 0


def test_mixed_model_options_apply_to_batch(monkeypatch):
    captured = {}

    monkeypatch.setattr(cli, "read_smi", lambda _path: {"mol": "C"})
    monkeypatch.setattr(cli, "protonate", mock_protonate(captured))
    monkeypatch.setattr(cli, "save_sdf", lambda *_args: None)

    with CliRunner().isolated_filesystem():
        Path("mols.smi").write_text("C mol\n", encoding="utf-8")
        result = CliRunner().invoke(
            cli.cli,
            ["batch", "--smi", "mols.smi", "--model", "mixed", "--no-gpu"],
        )

    assert result.exit_code == 0
    assert captured["model"]["unipka"]["gpu"] is False


def test_batch_smi_must_be_an_existing_file(tmp_path):
    missing = CliRunner().invoke(
        cli.cli,
        ["batch", "--smi", str(tmp_path / "missing.smi")],
    )
    directory = CliRunner().invoke(
        cli.cli,
        ["batch", "--smi", str(tmp_path)],
    )

    assert missing.exit_code != 0
    assert "--smi: must be an existing file" in missing.output
    assert directory.exit_code != 0
    assert "--smi: must be an existing file" in directory.output


def test_batch_njobs_must_not_be_zero():
    result = CliRunner().invoke(
        cli.cli,
        ["batch", "--smi", "mols.smi", "--njobs", "0"],
    )

    assert result.exit_code != 0
    assert "--njobs must not be 0." in result.output


def test_ph_values_must_be_finite():
    runner = CliRunner()

    for args, error in (
        (["single", "--smiles", "C", "--ph", "nan"], "--ph must be finite."),
        (["batch", "--smi", "mols.smi", "--ph", "nan"], "--ph must be finite."),
        (["scan", "--smiles", "C", "--min-ph", "nan"], "--min-ph must be finite."),
        (["scan", "--smiles", "C", "--max-ph", "nan"], "--max-ph must be finite."),
        (
            ["single", "--smiles", "C", "--cutoff-export", "nan"],
            "--cutoff-export must be >= 0 and <= 1.",
        ),
    ):
        result = runner.invoke(cli.cli, args)

        assert result.exit_code != 0
        assert error in result.output


def test_file_outputs_reject_directories(tmp_path):
    runner = CliRunner()
    regular_file = tmp_path / "not_a_directory"
    regular_file.write_text("", encoding="utf-8")

    for args in (
        ["single", "--smiles", "C", "--sdf-out", str(tmp_path)],
        ["single", "--smiles", "C", "--txt-out", str(tmp_path)],
        ["batch", "--smi", "mols.smi", "--txt-out", str(tmp_path)],
        ["scan", "--smiles", "C", "--fig-out", str(tmp_path)],
        ["scan", "--smiles", "C", "--sdf-out", str(tmp_path)],
        ["scan", "--smiles", "C", "--pkas-out", str(tmp_path)],
        ["scan", "--smiles", "C", "--txt-out", str(tmp_path)],
    ):
        result = runner.invoke(cli.cli, args)

        assert result.exit_code != 0
        assert "is a directory" in result.output

    result = runner.invoke(
        cli.cli,
        ["batch", "--smi", "mols.smi", "--path-out", str(regular_file)],
    )

    assert result.exit_code != 0
    assert "is a file" in result.output


def test_mixed_model_options_apply_to_scan(monkeypatch):
    captured = {}

    class Scan:
        def export_macro_pkas(self, file):
            pass

        def print_macro_pkas(self):
            pass

        def plot_scan(self):
            return None

        def plot_mols(self, size_x, size_y):
            return None

        def export_scan(self, fig_out, fig_scan, fig_mols):
            pass

        def save_sdf(self, sdf_out):
            pass

    def mock_scan_ph(*args, **kwargs):
        captured.update(kwargs)
        return Scan()

    monkeypatch.setattr(cli, "scan_pH", mock_scan_ph)

    with CliRunner().isolated_filesystem():
        result = CliRunner().invoke(
            cli.cli,
            ["scan", "--smiles", "C", "--model", "mixed", "--nthreads", "2"],
        )

    assert result.exit_code == 0
    assert captured["nthreads"] == 2
    assert "nthreads" not in captured["model"]["unipka"]


def test_scan_writes_all_ph_tables_to_txt_out(monkeypatch):
    captured = {}

    class MoleculeOutput:
        smiles = ("C",)
        mols = (Mol(),)

    class TextScan(Scan):
        pHs = [7.0, 7.25]

        def molecule_at(self, pH):
            return MoleculeOutput()

    def scan_ph(*args, **kwargs):
        captured.update(kwargs)
        return TextScan()

    monkeypatch.setattr(cli, "scan_pH", scan_ph)

    with CliRunner().isolated_filesystem():
        result = CliRunner().invoke(
            cli.cli,
            [
                "scan",
                "--smiles",
                "C",
                "--min-ph",
                "7.0",
                "--max-ph",
                "7.25",
                "--txt-out",
                "overview.txt",
            ],
        )
        txt_output = Path("overview.txt").read_text(encoding="utf-8")

    assert result.exit_code == 0
    assert captured["output_molecules_from_scan"] is True
    assert "molecule | pH: 7.0" in txt_output
    assert "molecule | pH: 7.25" in txt_output
    assert txt_output.count("Microstate") == 2


def test_scan_uses_common_cutoff_export_default_and_skips_molecule_output_without_txt_out(monkeypatch):
    captured = {}
    scan = Scan()

    def scan_ph(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return scan

    monkeypatch.setattr(cli, "scan_pH", scan_ph)

    result = CliRunner().invoke(cli.cli, ["scan", "--smiles", "C", "--name", "2014", "--pkas-out", "2014_pkas.txt"])

    assert result.exit_code == 0
    assert captured["args"][0] == "C"
    assert captured["name"] == "2014"
    assert captured["cutoff_export"] == 0.2
    assert captured["output_molecules_from_scan"] is False
    assert scan.exported_macro_pkas.name == "2014_pkas.txt"
