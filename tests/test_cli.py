from pathlib import Path

from click.testing import CliRunner

from pkasso import cli


class Mol:
    def GetProp(self, name):
        return {"_Name": "state0", "Probability": "1.0", "net_charge": "0"}[name]


def mock_protonate(captured):
    def protonate(*args, **kwargs):
        captured.update(kwargs)
        return ["C"], [Mol()]

    return protonate

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
            "folds": 3,
            "model_dir": tmp_path,
            "nthreads": 4,
            "gpu": True,
        },
    }


def test_mixed_model_defaults_to_packaged_data_and_cpu(monkeypatch):
    captured = {}

    monkeypatch.setattr(cli, "protonate", mock_protonate(captured))

    result = CliRunner().invoke(cli.cli, ["single", "--smiles", "C", "--model", "mixed"])

    assert result.exit_code == 0
    assert captured["model"]["unipka"] == {
        "folds": 3,
        "model_dir": Path(cli.__file__).resolve().parent / "data",
        "nthreads": 0,
        "gpu": False,
    }


def test_mixed_model_options_apply_to_batch(monkeypatch):
    captured = {}

    monkeypatch.setattr(cli, "read_smi", lambda _path: {"mol": "C"})
    monkeypatch.setattr(cli, "protonate", mock_protonate(captured))
    monkeypatch.setattr(cli, "save_sdf", lambda *_args: None)

    with CliRunner().isolated_filesystem():
        result = CliRunner().invoke(
            cli.cli,
            ["batch", "--smi", "mols.smi", "--model", "mixed", "--no-gpu"],
        )

    assert result.exit_code == 0
    assert captured["model"]["unipka"]["gpu"] is False


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
    assert captured["model"]["unipka"]["nthreads"] == 2
