"""molgpka pka calculations with custom rules"""
# mypy: disable-error-code=no-untyped-call

from pathlib import Path

import torch
from rdkit.Chem.rdchem import Mol
from torch_geometric.data import Data

from .descriptor import MolVectorizer, mol2vec
from .ionization_group import get_ionization_aid
from .net import GCNNet


def load_model(model_file: Path, device: str = "cpu") -> GCNNet:
    """Load molgpka ML torch model."""

    model = GCNNet().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    model.eval()
    return model


def model_pred(mol: Mol, atom_idx: int, model: GCNNet, device: str = "cpu") -> float:
    """Predict pKa with molgpka model."""

    data = mol2vec(mol, atom_idx)
    return model_pred_data(data, model, device=device)


def model_pred_data(data: Data, model: GCNNet, device: str = "cpu") -> float:
    """Predict pKa from precomputed molgpka graph data."""

    with torch.no_grad():
        data = data.to(device)
        pKa = model(data)
        pKa = pKa.cpu().numpy()
        pka: float = pKa[0][0]
    return pka


def predict_acid(mol_h: Mol, model_acid: GCNNet, device: str = "cpu") -> dict[int, float]:
    """Predict acid pKas with molgpka model."""

    acid_idxs = get_ionization_aid(mol_h, "acid")
    acid_res = {}
    vectorizer = MolVectorizer(mol_h)
    for aid in acid_idxs:
        apka = model_pred_data(vectorizer.mol2vec(aid), model_acid, device=device)
        acid_res.update({aid: apka})
    return acid_res


def predict_base(mol_h: Mol, model_base: GCNNet, device: str = "cpu") -> dict[int, float]:
    """Predict base pKas with molgpka model."""

    base_idxs = get_ionization_aid(mol_h, "base")
    base_res = {}
    vectorizer = MolVectorizer(mol_h)
    for aid in base_idxs:
        bpka = model_pred_data(vectorizer.mol2vec(aid), model_base, device=device)
        base_res.update({aid: bpka})
    return base_res
