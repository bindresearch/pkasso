from rdkit import Chem

from pkasso.predict_pka import UnipkaPredictor


def mapped_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return mol


def test_unipka_predictor_carboxylic_acid_site_ids():
    predictor = UnipkaPredictor(mapped_mol("CCCCC(=O)O"))

    assert predictor.pred_acid_ids() == [7]
    assert predictor.pred_base_ids() == [6]
    assert predictor.exclude_sites() == ([], [])
    assert predictor.pred_acid() == {}
    assert predictor.pred_base() == {}
    assert not hasattr(predictor, "model")


def test_unipka_predictor_amine_site_ids():
    predictor = UnipkaPredictor(mapped_mol("NCCCCC"))

    assert predictor.pred_acid_ids() == []
    assert predictor.pred_base_ids() == [1]


def test_unipka_predictor_ammonium_site_ids():
    predictor = UnipkaPredictor(mapped_mol("[NH3+]CCCCC"))

    assert predictor.pred_acid_ids() == [1]
    assert predictor.pred_base_ids() == []
