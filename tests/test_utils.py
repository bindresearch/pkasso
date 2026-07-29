import pytest

from pkasso.utils import read_smi


def test_read_smi_skips_blank_lines(tmp_path):
    smi = tmp_path / "molecules.smi"
    smi.write_text("\n C first\n   \nN\n\nO third\n", encoding="utf-8")

    assert read_smi(smi) == {
        "first": "C",
        "molecule0": "N",
        "third": "O",
    }


def test_read_smi_rejects_duplicate_names(tmp_path):
    smi = tmp_path / "molecules.smi"
    smi.write_text("C repeated\nN repeated\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"Duplicate molecule name 'repeated' on line 2",
    ):
        read_smi(smi)


def test_read_smi_rejects_duplicate_generated_key(tmp_path):
    smi = tmp_path / "molecules.smi"
    smi.write_text("C molecule0\nN\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"Duplicate molecule name 'molecule0' on line 2",
    ):
        read_smi(smi)


def test_read_smi_rejects_invalid_or_empty_input(tmp_path):
    smi = tmp_path / "molecules.smi"
    smi.write_text("not_smiles bad\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Invalid SMILES 'not_smiles' on line 1"):
        read_smi(smi)

    smi.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="No molecules found"):
        read_smi(smi)
