from pathlib import Path

import torch
from rdkit.Chem.rdchem import Mol

from .descriptor import mol2vec
from .ionization_group import get_ionization_aid
from .net import GCNNet


def load_model(model_file: Path, device: str = "cpu") -> GCNNet:
    """ Load molgpka ML torch model. """

    model= GCNNet().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    model.eval()
    return model

def model_pred(mol: Mol, atom_idx: int, model: GCNNet, device: str = "cpu") -> float:
    """ Predict pKa with molgpka model. """

    data = mol2vec(mol, atom_idx)
    with torch.no_grad():
        data = data.to(device)
        pKa = model(data)
        pKa = pKa.cpu().numpy()
        pka: float = pKa[0][0]
    return pka

def predict_acid(mol: Mol, model_acid: GCNNet, device: str = "cpu"
) -> dict[int, float]:
    """ Predict acid pKas with molgpka model. """

    acid_idxs = get_ionization_aid(mol, "acid")
    acid_res = {}
    for aid in acid_idxs:
        apka = model_pred(mol, aid, model_acid, device=device)
        acid_res.update({aid:apka})
    return acid_res

def predict_base(mol: Mol, model_base: GCNNet, device: str = "cpu"
) -> dict[int, float]:
    """ Predict base pKas with molgpka model. """
  
    base_idxs= get_ionization_aid(mol, "base")
    base_res = {}
    for aid in base_idxs:
        bpka = model_pred(mol, aid, model_base, device=device)
        base_res.update({aid:bpka})
    return base_res

def predict_acid_base(
    mol_h: Mol,
    model_base: GCNNet,
    model_acid: GCNNet,
    device: str = 'cpu',
    verbose: bool = False,
    pred_acid: bool = True,
    pred_base: bool =True,
    ) -> tuple[dict[int, float], dict[int, float]]:
    """ Wrapper for acid-base prediction with molgpka. """

    if pred_base:
        base = predict_base(mol_h,model_base,device=device)

        base_curated = {} # atom mapping

        for at_idx, pka in base.items():
            atom = mol_h.GetAtomWithIdx(at_idx) 
            map_idx = atom.GetAtomMapNum()
            base_curated[map_idx] = pka
        if verbose:
            print('base')
            print(base)
            print('base curated')
            print(base_curated)
        base = base_curated
    else:
        base = {}

    if pred_acid:
        acid = predict_acid(mol_h,model_acid,device=device)
        acid = get_acid_neighbors(mol_h, acid)

        if verbose:
            print('acid heavy')
            print(acid)

        acid_curated = {} # atom mapping

        for at_idx, pka in acid.items():
            atom = mol_h.GetAtomWithIdx(at_idx) 
            map_idx = atom.GetAtomMapNum()
            acid_curated[map_idx] = pka
        if verbose:
            print('acid curated')
            print(acid_curated)
        acid = acid_curated
    else:
        acid = {}
    return base, acid

def get_acid_neighbors(mol_h: Mol, acid: dict[int, float], verbose: bool = False) -> dict[int, float]:
    """ Find heavy-atom neighbour for acidic proton. """
    acid_heavy = {}

    for at_idx, pka in acid.items():
        H_acid = mol_h.GetAtomWithIdx(at_idx)
        for bond in H_acid.GetBonds():
            neighbor = bond.GetOtherAtom(H_acid)
            neighbor_idx = neighbor.GetIdx()
            if verbose:
                print(f'Neighbor to acid H{at_idx}: {neighbor.GetSymbol()}{neighbor.GetIdx()}')
                print(f'pka: {pka}')
            acid_heavy[neighbor_idx] = pka
    return acid_heavy