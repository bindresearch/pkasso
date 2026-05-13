from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from .descriptor import mol2vec
from .ionization_group import get_ionization_aid
from .net import GCNNet

from autoprot.special_cases import match_pattern, oh_ring_sulfonate, has_nplus_base_proximity

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

        # smarts_NN = '[#7]~[#7]'

        pattern_ncncO = Chem.MolFromSmarts('[n;r6][c;r6][nH;r6][c;r6](=O)')
        _, matches_ncncO = match_pattern(mol_h, pattern_ncncO)

        for at_idx, pka in base.items():
            atom = mol_h.GetAtomWithIdx(at_idx)
            
            ncat = has_nplus_base_proximity(at_idx, mol_h, max_distance=5)
            # print(ncat)
            correction = 2.5 * max(0,(ncat-1.))
            # print(correction)
            pka -= correction

            if atom.GetSymbol() == 'N':
                for match in matches_ncncO:
                    if (at_idx in match):
                        pka -= 2.
                        continue

            # pattern = Chem.MolFromSmarts(smarts_NN)
            # _, matches = match_pattern(mol_h, pattern)
            # for match in matches:
            #     if at_idx in match:
            #         pka -= 2.

            # pka -= 2.
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

        pattern_ncS = Chem.MolFromSmarts('[n,nH,n+]~[c,C]~[S,s]')
        _, matches_ncS = match_pattern(mol_h, pattern_ncS)


        smarts_OHcRO = [
            '[O;H1]-[C;R]~[C;R]~[C;R]([O-])',
            '[O;H1]-[C;R]~[C;R]([O-])',
            '[O;H1]-[C;R]([O-])',
        ]

        smarts_ON = '[#7]~[#8]'

        smarts_carbox_close = [
            '[C](=O)([OH])~[#6]~[C](=O)([O-])',
            '[C](=O)([OH])~[#6][#6]~[C](=O)([O-])',
            '[C](=O)([OH])~[#6]=[#6]~[C](=O)([O-])',
        ]

        pattern_ncncO = Chem.MolFromSmarts('[n;r6][c;r6][nH;r6][c;r6](=O)')
        _, matches_ncncO = match_pattern(mol_h, pattern_ncncO)

        acid_curated = {} # atom mapping
        for at_idx, pka in acid.items():
            atom = mol_h.GetAtomWithIdx(at_idx)

            if atom.GetSymbol() == 'N':
                for match in matches_ncS:
                    for match2 in matches_ncncO:
                        if (at_idx in match) and (at_idx not in match2):
                            pka += 1. # 3.
                            continue

            if atom.GetSymbol() == 'O':
                for smarts in smarts_OHcRO:
                    pattern = Chem.MolFromSmarts(smarts)
                    _, matches = match_pattern(mol_h, pattern)
                    for match in matches:
                        if (at_idx in match):
                            pka += 3.#3.
                            continue
                oh_sooo_list = oh_ring_sulfonate(mol_h)
                if at_idx in oh_sooo_list:
                    pka += 4

                for smarts in smarts_carbox_close:
                    pattern = Chem.MolFromSmarts(smarts)
                    _, matches = match_pattern(mol_h, pattern)
                    for match in matches:
                        if (at_idx in match):
                            pka += 2.
                            continue

                pattern = Chem.MolFromSmarts(smarts_ON)
                _, matches = match_pattern(mol_h, pattern)
                for match in matches:
                    if (at_idx in match):
                        pka += 0. # 2. # 3.
                        continue


            map_idx = atom.GetAtomMapNum()
            acid_curated[map_idx] = pka
        # if verbose:
        # print('acid curated')
        # print(acid_curated)
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