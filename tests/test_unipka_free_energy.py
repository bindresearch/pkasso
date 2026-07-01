import pandas as pd
from rdkit import Chem

from pkasso.external.unipka.pka_predictor.free_energy import (
    _aggregate_free_energy_predictions,
    _mol_to_unmapped_smiles,
)


def test_mol_to_unmapped_smiles_removes_atom_maps_and_preserves_charge():
    mol = Chem.MolFromSmiles("[CH3:1][C:2](=O)[O-:3]")

    smiles = _mol_to_unmapped_smiles(mol)

    assert ":" not in smiles
    assert Chem.GetFormalCharge(Chem.MolFromSmiles(smiles)) == -1


def test_aggregate_free_energy_predictions_averages_conformers_by_input_order():
    conformer_results = pd.DataFrame(
        [
            {"molecule_index": 0, "smiles": "CCO", "conformer_free_energy": 1.0},
            {"molecule_index": 0, "smiles": "CCO", "conformer_free_energy": 2.0},
            {"molecule_index": 1, "smiles": "CCN", "conformer_free_energy": -1.0},
            {"molecule_index": 1, "smiles": "CCN", "conformer_free_energy": 1.0},
        ]
    )

    results = _aggregate_free_energy_predictions(conformer_results, n_molecules=2)

    assert results["smiles"].tolist() == ["CCO", "CCN"]
    assert results["standard_free_energy"].tolist() == [1.5, 0.0]
    assert results["n_conformers"].tolist() == [2, 2]
