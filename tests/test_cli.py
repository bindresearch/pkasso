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


def test_scan_pkas_out_without_cutoff_export(monkeypatch):
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
    assert "cutoff_export" not in captured
    assert scan.exported_macro_pkas.name == "2014_pkas.txt"
