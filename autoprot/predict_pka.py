""" Module to abstract pka prediction away from model """

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from .special_cases import has_invalid_amine, has_nplus_base_proximity, has_phosphate, match_smarts, oh_ring_sulfonate
from .external.net import GCNNet
from .external.pka import predict_acid as molgpka_predict_acid
from .external.pka import predict_base as molgpka_predict_base

from abc import ABC, abstractmethod
from typing import Any

# from dataclasses import dataclass

def get_acid_neighbors(mol_h: Mol, acid: dict[int, float]) -> dict[int, float]:
    """ Find heavy-atom neighbour for acidic proton. """
    acid_heavy = {}

    for at_idx, pka in acid.items():
        H_acid = mol_h.GetAtomWithIdx(at_idx)
        for bond in H_acid.GetBonds():
            neighbor = bond.GetOtherAtom(H_acid)
            neighbor_map_idx = neighbor.GetAtomMapNum()
            # if verbose:
                # print(f'Neighbor to acid H{at_idx}: {neighbor.GetSymbol()}{neighbor.GetIdx()}')
                # print(f'pka: {pka}')
            acid_heavy[neighbor_map_idx] = pka
    return acid_heavy

def convert_base_map_idx(mol_h: Mol, base_res: dict[int, float]) -> dict[int, float]:
    base_res_map_ids: dict[int, float] = {}
    for aid, pka in base_res.items():
        atom = mol_h.GetAtomWithIdx(aid)
        map_idx = atom.GetAtomMapNum()
        base_res_map_ids[map_idx] = pka
    return base_res_map_ids

class Predictor(ABC):

    def __init__(self, mol: Mol):
        self.mol = mol
    
    @abstractmethod
    def pred_acid(self, model: Any, device: str = "cpu") -> dict[int, float]:
        """Predict acidic pKa values keyed by atom map index."""
        ...

    @abstractmethod
    def pred_base(self, model: Any, device: str = "cpu") -> dict[int, float]:
        """Predict basic pKa values keyed by atom map index."""
        ...

class MolgpkaPredictor(Predictor):

    def __init__(self, mol: Mol):
        super().__init__(mol)
        self.mol_h = Chem.rdmolops.AddHs(Chem.Mol(mol))
        self.atom_indices = [atom.GetIdx() for atom in mol.GetAtoms()]
        self.qs = np.array([at.GetFormalCharge() for at in mol.GetAtoms()])

    def pred_acid(
        self,
        model: GCNNet,
        device: str = "cpu",
    ) -> dict[int, float]:
        
        acid = molgpka_predict_acid(self.mol_h, model, device=device)
        acid = get_acid_neighbors(self.mol_h, acid)
        matches_ncS = match_smarts(self.mol, '[n,nH,n+]~[c,C]~[S,s]')

        smarts_OHcRO = [
            '[O;H1]-[C;R]~[C;R]~[C;R]([O-])',
            '[O;H1]-[C;R]~[C;R]([O-])',
            '[O;H1]-[C;R]([O-])',
        ]

        smarts_carbox_close = [
            '[C](=O)([OH])~[#6]~[C](=O)([O-])',
            '[C](=O)([OH])~[#6][#6]~[C](=O)([O-])',
            '[C](=O)([OH])~[#6]=[#6]~[C](=O)([O-])',
        ]

        matches_ncncO = match_smarts(self.mol, '[n;r6][c;r6][nH;r6][c;r6](=O)')

        acid_curated = {} # atom mapping

        _, phosphate_groups = has_phosphate(self.mol)

        for at_idx, q in zip(self.atom_indices, self.qs):
            if q != 0.:
                continue
            pka = None
            atom = self.mol.GetAtomWithIdx(at_idx)
            map_idx = atom.GetAtomMapNum()
            
            if map_idx in acid:
                pka = acid[map_idx]

                if atom.GetSymbol() == 'N':
                    for match in matches_ncS:
                        for match2 in matches_ncncO:
                            if (at_idx in match) and (at_idx not in match2):
                                pka += 1. # 3.
                                continue

                if atom.GetSymbol() == 'O':
                    for smarts in smarts_OHcRO:
                        matches = match_smarts(self.mol, smarts)
                        for match in matches:
                            if (at_idx in match):
                                pka += 3.#3.
                                continue
                    oh_sooo_list = oh_ring_sulfonate(self.mol)
                    if at_idx in oh_sooo_list:
                        pka += 4

                    for smarts in smarts_carbox_close:
                        matches = match_smarts(self.mol, smarts)
                        for match in matches:
                            if (at_idx in match):
                                pka += 2.
                                continue

            for _, oh_map_ids in phosphate_groups.items():
                if map_idx in oh_map_ids:

                    other_O_deprotonated = False
                    for atom in self.mol.GetAtoms():
                        o_map_idx = atom.GetAtomMapNum()
                        if (o_map_idx in oh_map_ids) and (o_map_idx != map_idx): 
                            q_other_O = atom.GetFormalCharge() # check if other O deprotonated
                            if q_other_O == -1.:
                                other_O_deprotonated = True
                                break
                    if other_O_deprotonated:
                        pka = 6.5
                    else:
                        pka = 2.0

                    # matches = match_smarts(mol_h, smarts_ON)
                    # for match in matches:
                    #     if (at_idx in match):
                    #         pka += 0. # 2. # 3.
                    #         continue
                    # print(pka)
            if pka is not None:
                if map_idx == 0:
                    raise
                acid_curated[map_idx] = pka
        acid = acid_curated
        return acid

    def pred_base(
        self,
        model: GCNNet,
        device: str = "cpu",
    ) -> dict[int, float]:
        
        base_aid = molgpka_predict_base(self.mol_h, model, device=device)
        base = convert_base_map_idx(self.mol_h, base_aid)
        base_curated = {} # atom mapping

        invalid_amine_map_idx = has_invalid_amine(self.mol)

        matches_ncncO = match_smarts(self.mol, '[n;r6][c;r6][nH;r6][c;r6](=O)')

        smarts_NphenNOO = "Ncccc([N+](=O)[O-])"
        matches_NphenNOO = match_smarts(self.mol,smarts_NphenNOO)

        for at_idx, q in zip(self.atom_indices, self.qs):
            if q != 0.:
                continue
            pka = None
            atom = self.mol.GetAtomWithIdx(at_idx)
            map_idx = atom.GetAtomMapNum()

            # corrections to existing molgpka predictions
            if map_idx in base:
                pka = base[map_idx]
                
                ncat = has_nplus_base_proximity(map_idx, self.mol, max_distance=5)
                correction = 2.5 * max(0,(ncat-1.))
                pka -= correction

                if atom.GetSymbol() == 'N':
                    for match in matches_ncncO:
                        if (at_idx in match):
                            pka -= 2.
                            continue

            # Added pka predictions (independent of molgpka)
            if (atom.GetSymbol() == 'N') and not (atom.GetIsAromatic()):
                for match in matches_NphenNOO:
                    if (at_idx in match):
                        # print('found',smarts_NphenNOO)
                        pka = 2.0

            # none of the added hydrogens should be considered here
            if (map_idx > 0) and (map_idx == invalid_amine_map_idx):
                pka = 10.4

            if pka is not None:
                # print(pka)
                if map_idx == 0:
                    raise
                base_curated[map_idx] = pka

        base = base_curated
        return base

def predict_acid(
    mol: Mol,
    model: Any,
    device: str = "cpu",
    predictor_cls: type[Predictor] = MolgpkaPredictor,
) -> dict[int, float]:
    """Predict acidic pKa values with the selected predictor backend."""

    predictor = predictor_cls(mol)
    return predictor.pred_acid(model, device=device)


def predict_base(
    mol: Mol,
    model: Any,
    device: str = "cpu",
    predictor_cls: type[Predictor] = MolgpkaPredictor,
) -> dict[int, float]:
    """Predict basic pKa values with the selected predictor backend."""

    predictor = predictor_cls(mol)
    return predictor.pred_base(model, device=device)
