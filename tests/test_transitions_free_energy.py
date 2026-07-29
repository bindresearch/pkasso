import numpy as np
import pytest

from pkasso import transitions


def test_calc_proton_count_offsets_from_state_vectors():
    state_vecs = [
        np.array([1, 1]),
        np.array([0, 1]),
        np.array([2, 1]),
        np.array([0, 2]),
    ]

    offsets = transitions.calc_proton_count_offsets(state_vecs)

    assert offsets.tolist() == [0.0, -1.0, 1.0, 0.0]


def test_calc_state_pH_dependent_free_energies_adds_shifted_ph_term():
    standard_free_energies = np.array([5.0, 5.0, 5.0])
    state_vecs = [
        np.array([1]),
        np.array([0]),
        np.array([2]),
    ]

    free_energies = transitions.calc_state_pH_dependent_free_energies(
        standard_free_energies,
        state_vecs,
        pH=7.0,
        target_mean=6.0,
    )

    assert free_energies.tolist() == pytest.approx(
        [
            5.0,
            5.0 - np.log(10),
            5.0 + np.log(10),
        ]
    )
