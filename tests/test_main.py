import pytest
import numpy as np

from autoprot import main

from rdkit import Chem

@pytest.mark.parametrize(
    ("smiles_raw","net_charge"),
    [
        (r"CCCCC(=O)O", 0),
        (r"CCCCC(=O)[O-]", 0),
        (r"[NH3+]CCCCC", 0),
        (r"[N+](C)(C)(C)(C)", 1),
        (r"CCCCC", 0),
    ],
)
def test_preprocess(smiles_raw,net_charge):

    mol, exclude_indices = main.preprocess(smiles_raw,verbose=False)
    qs = [at.GetFormalCharge() for at in mol.GetAtoms()]
    print(qs)
    for at_idx, q in enumerate(qs):
        print(at_idx, q)
        if q != 0:
            assert at_idx in exclude_indices
    assert Chem.GetFormalCharge(mol) == net_charge


@pytest.mark.parametrize(
    ("pH","pH_band","expected_indices","expected_q_options"),
    [
        (1., 4.5,
         [0,1,2,3],
         np.array([
             [1,0,0,0],
             [1,1,1,1],
             [1,1,1,0]
         ])
         ),
        (8., 4.5,
         [0,1,2,3],
         np.array([
             [1,0,0,1],
             [1,1,1,1],
             [0,1,1,0]
         ])
         ),
        (8., 0.,
         [0,1,2,3],
         np.array([
             [1,0,0,0],
             [1,1,1,1],
             [0,0,1,0]
         ])
         ),
    ],
)
def test_find_candidate_sites(pH,pH_band,expected_indices,expected_q_options):
    base = {
        0: 2.0,
        1: 7.0,
        2: 12.0,}
    acid = {
        0: 4.0,
        3: 12.0
    }
    exclude_indices = []
    indices, q_options = main.find_candidate_sites(
        base, acid, exclude_indices,pH, pH_band=pH_band)
    assert (np.allclose(indices,expected_indices)) and (np.allclose(q_options,expected_q_options))

def test_construct_state_vectors():
    q_options = np.array([
        [1,0],
        [1,1],
        [0,1]
    ])
    cutoff_states = 100
    state_vecs = main.construct_state_vectors(q_options, cutoff_states)
    print(state_vecs)
    assert np.allclose(state_vecs, np.array([
        [0, 1],
        [0, 2],
        [1, 1],
        [1, 2]
    ]))
    cutoff_states = 2
    state_vecs = main.construct_state_vectors(q_options, cutoff_states)
    assert state_vecs == []
    # assert False


@pytest.mark.parametrize(
    ("smiles_raw",'net_charge'),
    [
        (r"NCCCCC", 1),
        (r"NCCCCCN", 2),
        (r"CCCCCO", -1),
        (r"NCCCCCO", 0),
        (r"Nc1ccc(O)cc1", 0),
        (r"Nc1ccc(N)cc1", 2),
    ],
)
def test_construct_mol(smiles_raw,net_charge):
    mol = Chem.MolFromSmiles(smiles_raw)
    symbols = [at.GetSymbol() for at in mol.GetAtoms()]

    indices = []
    state_vec = []

    for at_idx, s in enumerate(symbols):
        if s == 'N':
            indices.append(at_idx)
            state_vec.append(2)
        elif s == 'O':
            indices.append(at_idx)
            state_vec.append(0)
    state_vec = np.array(state_vec)
    mol_cand, _ = main.construct_mol(mol, indices, state_vec)
    assert Chem.GetFormalCharge(mol_cand) == net_charge