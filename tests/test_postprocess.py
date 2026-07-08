import pytest
import numpy as np
from rdkit import Chem

from pkasso.postprocess import Scan, combine_results


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
    pHs = np.array([6.0, 7.0], dtype=np.float64)
    scan = Scan(
        name="scan",
        indices=[1],
        state_strs_relevant=["1"],
        mols_relevant=[],
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

    fig = scan.plot_scan()

    assert scan.sfreqs_relevant_sigmas[0].tolist() == pytest.approx([0.05, 0.03])
    assert scan.net_charge_sigmas.tolist() == pytest.approx([0.02, 0.04])
    assert scan.pkas_macro_sigmas == {0: 0.2}
    assert fig is not None
