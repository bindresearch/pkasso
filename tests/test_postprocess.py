import pytest
import numpy as np
from rdkit import Chem

import pkasso.postprocess as postprocess
from pkasso.postprocess import Molecule, Scan, combine_results


def test_combine_results_exports_frequency_sigmas():
    mol = Chem.MolFromSmiles("C")
    mols_lib = {"1": mol}

    molecule = combine_results(
        "methane",
        ["1"],
        [0.5],
        mols_lib,
        {"1": 0},
        [0.1],
    )

    assert molecule.freqs == pytest.approx((1.0,))
    assert molecule.freqs_sigmas == pytest.approx((0.2,))
    assert molecule.microstates[0].freq_sigma == pytest.approx(0.2)
    assert molecule.mols[0].GetProp("Probability_sigma") == "0.1"


def test_scan_exposes_uncertainty_curves_and_plots_them():
    mol = Chem.MolFromSmiles("C")
    mol.SetProp("_Name", "1 (+0)")
    pHs = np.array([6.0, 7.0], dtype=np.float64)
    scan = Scan(
        name="scan",
        indices=[1],
        state_strs_relevant=["1"],
        mols_relevant=[mol],
        sfreqs_relevant=[np.array([0.8, 0.2], dtype=np.float64)],
        sfreqs_relevant_sigmas=[np.array([0.05, 0.03], dtype=np.float64)],
        pHs=pHs,
        net_charges=np.array([0.2, 0.8], dtype=np.float64),
        net_charge_sigmas=np.array([0.02, 0.04], dtype=np.float64),
        sfreqs_not_relevant=[np.array([0.2, 0.8], dtype=np.float64)],
        sfreqs_not_relevant_sigmas=[np.array([0.05, 0.03], dtype=np.float64)],
        pkas_macro={0: 6.5},
        pkas_macro_sigmas={0: 0.2},
    )

    fig = scan.plot_scan(fontsize=14)

    assert scan.sfreqs_relevant_sigmas[0].tolist() == pytest.approx([0.05, 0.03])
    assert scan.net_charge_sigmas.tolist() == pytest.approx([0.02, 0.04])
    assert scan.pkas_macro_sigmas == {0: 0.2}
    assert fig.axes[1].texts[0].get_fontsize() == 14


def test_scan_plot_mols_delegates_to_draw_mols(monkeypatch):
    mol = Chem.MolFromSmiles("[CH3:1][OH:2]")
    mol.SetProp("_Name", "1 (+0)")
    scan = Scan(
        name="scan",
        indices=[1],
        state_strs_relevant=["1"],
        mols_relevant=[mol],
        sfreqs_relevant=[],
        sfreqs_relevant_sigmas=[],
        pHs=np.array([], dtype=np.float64),
        net_charges=np.array([], dtype=np.float64),
        net_charge_sigmas=np.array([], dtype=np.float64),
        sfreqs_not_relevant=[],
        sfreqs_not_relevant_sigmas=[],
        pkas_macro={},
    )
    sentinel = object()
    calls = []

    def fake_draw_mols(mols, **kwargs):
        calls.append((mols, kwargs))
        return sentinel

    monkeypatch.setattr(postprocess, "draw_mols", fake_draw_mols)

    assert scan.plot_mols(size_x=300, size_y=225, molsPerRow=2, fontsize=11) is sentinel
    assert calls == [
        (
            scan.mols_relevant,
            {
                "subImgSize": (300, 225),
                "max_cols": 2,
                "show_probability": False,
                "fontsize": 11,
            },
        )
    ]
    assert all(atom.GetAtomMapNum() == 0 for atom in mol.GetAtoms())


def test_scan_molecule_at_supports_exact_and_tolerant_lookup():
    molecule = Molecule("scan", ())
    scan = Scan(
        name="scan",
        indices=[],
        state_strs_relevant=[],
        mols_relevant=[],
        sfreqs_relevant=[],
        sfreqs_relevant_sigmas=[],
        pHs=np.array([7.5], dtype=np.float64),
        net_charges=np.array([0.0], dtype=np.float64),
        net_charge_sigmas=np.array([0.0], dtype=np.float64),
        sfreqs_not_relevant=[],
        sfreqs_not_relevant_sigmas=[],
        pkas_macro={},
        molecules={7.5: molecule},
    )

    assert scan.molecules[7.5] is molecule
    assert scan.molecule_at(7.5) is molecule
    assert scan.molecule_at(7.500000001) is molecule

    with pytest.raises(KeyError, match="No scanned pH found"):
        scan.molecule_at(7.6)

    with pytest.raises(ValueError, match="non-negative"):
        scan.molecule_at(7.5, tolerance=-1.0)


def test_scan_molecule_at_lazily_materializes_and_caches_output():
    distribution = object()
    molecule = Molecule("scan", ())
    calls = []
    scan = Scan(
        name="scan",
        indices=[],
        state_strs_relevant=[],
        mols_relevant=[],
        sfreqs_relevant=[],
        sfreqs_relevant_sigmas=[],
        pHs=np.array([7.5], dtype=np.float64),
        net_charges=np.array([0.0], dtype=np.float64),
        net_charge_sigmas=np.array([0.0], dtype=np.float64),
        sfreqs_not_relevant=[],
        sfreqs_not_relevant_sigmas=[],
        pkas_macro={},
        microstate_distributions={7.5: distribution},
        _molecule_factory=lambda value: calls.append(value) or molecule,
    )

    assert scan.molecules == {}
    assert scan.molecule_at(7.500000001) is molecule
    assert scan.molecule_at(7.5) is molecule
    assert calls == [distribution]
    assert scan.molecules == {7.5: molecule}
