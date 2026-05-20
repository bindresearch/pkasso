""" Module to abstract pka prediction away from model """
# mypy: disable-error-code=no-untyped-call

from abc import ABC, abstractmethod
from importlib import resources
from pathlib import Path
from typing import ClassVar

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from .special_cases import (
    add_exclusion,
    has_invalid_amine,
    has_nplus_base_proximity,
    has_phosphate,
    match_smarts,
    oh_ring_sulfonate,
)
from .external.net import GCNNet
from .external.pka import load_model
from .external.pka import predict_acid as molgpka_predict_acid
from .external.pka import predict_base as molgpka_predict_base

pkg_base = resources.files('autoprot')

ROOT = Path(f'{pkg_base}/data')

def get_acid_neighbors(mol_h: Mol, acid: dict[int, float]) -> dict[int, float]:
    """ Find heavy-atom neighbour for acidic proton. """
    acid_heavy = {}

    for at_idx, pka in acid.items():
        H_acid = mol_h.GetAtomWithIdx(at_idx)
        for bond in H_acid.GetBonds():
            neighbor = bond.GetOtherAtom(H_acid)
            neighbor_map_idx = neighbor.GetAtomMapNum()
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

    def __init__(self, mol: Mol, device: str = "cpu"):
        self.mol = mol
        self.device = device
    
    @abstractmethod
    def pred_acid(self) -> dict[int, float]:
        """Predict acidic pKa values keyed by atom map index."""
        ...

    @abstractmethod
    def pred_base(self) -> dict[int, float]:
        """Predict basic pKa values keyed by atom map index."""
        ...

    @abstractmethod
    def exclude_sites(self) -> tuple[list[int], list[int]]:
        """Return base and acid atom map indices to exclude for this backend."""
        ...

class MolgpkaPredictor(Predictor):

    model_file_base: ClassVar[Path] = ROOT / 'weight_base.pth'
    model_file_acid: ClassVar[Path] = ROOT / 'weight_acid.pth'
    _model_cache: ClassVar[dict[tuple[type, str], tuple[GCNNet, GCNNet]]] = {}

    def __init__(self, mol: Mol, device: str = "cpu") -> None:
        super().__init__(mol, device=device)
        self.model_base, self.model_acid = self._load_models(device)
        self.mol_h = Chem.rdmolops.AddHs(Chem.Mol(mol))
        self.atom_indices = [atom.GetIdx() for atom in mol.GetAtoms()]
        self.qs = np.array([at.GetFormalCharge() for at in mol.GetAtoms()])

    @classmethod
    def _load_models(cls, device: str) -> tuple[GCNNet, GCNNet]:
        cache_key = (cls, device)
        if cache_key not in cls._model_cache:
            model_base = load_model(cls.model_file_base, device=device)
            model_acid = load_model(cls.model_file_acid, device=device)
            cls._model_cache[cache_key] = (model_base, model_acid)
        return cls._model_cache[cache_key]

    def pred_acid(self) -> dict[int, float]:
        acid = self._predict_acid_raw()
        return self._curate_acid(acid)

    def exclude_sites(self) -> tuple[list[int], list[int]]:
        return self._exclude_molgpka_sites()

    def _exclude_molgpka_sites(self) -> tuple[list[int], list[int]]:
        """
        Exclude sites where molgpka predictions are not used directly.

        Exclusions act on the q_options level and are tracked separately
        for protonation (base behavior) and deprotonation (acid behavior).
        """

        exclude_acid_indices: set[int] = set()
        exclude_base_indices: set[int] = set()

        smarts_imine = "NC(=N)"
        matches_imine = match_smarts(self.mol, smarts_imine)

        smarts_sulfonamide = "NS(=O)(=O)"


        smarts_Ncnn = 'Nc(n)n'
        smarts_Nccn = 'Nc(c)n'
        smarts_Nccn2 = 'Nccn'
        smarts_nnn = 'nnn'
        smarts_ncnn = 'ncnn'
        smarts_cNO = 'C=NO'
        smarts_NNC = 'N-N=C'
        smarts_carbonyl = '[#7]~[#6X3](=[#8])'

        smarts_ONphos = 'OC=NP(=O)(O)O'
        matches_ONphos = match_smarts(self.mol, smarts_ONphos)
        smarts_ONO = '[O]-[N+]([O-])'
        smarts_ONCO1 = 'O=N-C=O'
        smarts_ONCO2 = 'C=C(N=O)O'

        for at_idx, q in zip(self.atom_indices, self.qs):
            atom = self.mol.GetAtomWithIdx(at_idx)
            map_idx = atom.GetAtomMapNum()

            if q != 0:
                continue
            if atom.GetSymbol() == 'O':
                for mat in matches_ONphos:
                    if atom.GetIdx() in mat:
                        correct_O = False
                        neighbors = atom.GetNeighbors()
                        for nbr in neighbors:
                            if nbr.GetSymbol() == 'C':
                                correct_O = True # O=CN part of the match
                        if correct_O:
                            exclude_acid_indices.add(map_idx)
                for smarts in [
                    smarts_ONO,
                    smarts_ONCO1,
                    smarts_ONCO2
                ]:
                    exclude_acid_indices = add_exclusion(exclude_acid_indices, self.mol, atom, smarts)

            if atom.GetSymbol() == 'N':
                exclude_base_indices = add_exclusion(exclude_base_indices, self.mol, atom, smarts_carbonyl)
                # aromatic n
                if atom.GetIsAromatic():
                    for smarts in [
                        smarts_nnn,
                    ]:
                        exclude_base_indices = add_exclusion(exclude_base_indices, self.mol, atom, smarts)
                    if not any(neigh.GetAtomicNum() == 7 for neigh in atom.GetNeighbors()):
                        exclude_base_indices = add_exclusion(exclude_base_indices, self.mol, atom, smarts_ncnn)
                    
                    # ring Ns contributing to pi system
                    if (atom.GetTotalNumHs() > 0) or (atom.GetDegree() == 3):
                        exclude_base_indices.add(map_idx)
                # non-aromatic N
                else:
                    for smarts in [
                        smarts_sulfonamide,
                        smarts_Ncnn,
                        smarts_Nccn,
                        smarts_Nccn2,
                        smarts_cNO,
                        smarts_NNC
                    ]:
                        exclude_base_indices = add_exclusion(exclude_base_indices, self.mol, atom, smarts)
                    for mat in matches_imine: # ...N-C(=N)...
                        if atom.GetIdx() in mat:
                            accept = True
                            for bond in atom.GetBonds(): # Find the correct of the two Ns
                                if bond.GetBondType() == Chem.BondType.DOUBLE:
                                    accept = False
                            if accept:
                                exclude_base_indices.add(map_idx)

        return sorted(exclude_base_indices), sorted(exclude_acid_indices)

    def _predict_acid_raw(self) -> dict[int, float]:
        """Run molgpka acid prediction and convert results to atom map indices."""

        acid = molgpka_predict_acid(self.mol_h, self.model_acid, device=self.device)
        return get_acid_neighbors(self.mol_h, acid)

    def _curate_acid(self, acid: dict[int, float]) -> dict[int, float]:
        """Apply AutoProt correction rules to raw acid pKa predictions."""

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

            if pka is not None:
                if map_idx == 0:
                    raise
                acid_curated[map_idx] = pka
        acid = acid_curated
        return acid

    def pred_base(self) -> dict[int, float]:
        base = self._predict_base_raw()
        return self._curate_base(base)

    def _predict_base_raw(self) -> dict[int, float]:
        """Run molgpka base prediction and convert results to atom map indices."""

        base_aid = molgpka_predict_base(self.mol_h, self.model_base, device=self.device)
        return convert_base_map_idx(self.mol_h, base_aid)

    def _curate_base(self, base: dict[int, float]) -> dict[int, float]:
        """Apply AutoProt correction rules to raw base pKa predictions."""

        base_curated = {} # atom mapping

        invalid_amine_map_idx = has_invalid_amine(self.mol)

        matches_ncncO = match_smarts(self.mol, '[n;r6][c;r6][nH;r6][c;r6](=O)')

        smarts_NphenNOO = "Ncccc([N+](=O)[O-])"
        matches_NphenNOO = match_smarts(self.mol,smarts_NphenNOO)

        smarts_diphenylamine = 'N(c)c'
        matches_diphenylamine = match_smarts(self.mol,smarts_diphenylamine)

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
                for match in matches_diphenylamine:
                    if (at_idx in match):
                        pka = 1.8

            # none of the added hydrogens should be considered here
            if (map_idx > 0) and (map_idx == invalid_amine_map_idx):
                pka = 10.4

            if pka is not None:
                if map_idx == 0:
                    raise
                base_curated[map_idx] = pka

        base = base_curated
        return base

def predict_acid(
    mol: Mol,
    device: str = "cpu",
    predictor_cls: type[Predictor] = MolgpkaPredictor,
) -> dict[int, float]:
    """Predict acidic pKa values with the selected predictor backend."""

    predictor = predictor_cls(mol, device=device)
    return predictor.pred_acid()

def predict_base(
    mol: Mol,
    device: str = "cpu",
    predictor_cls: type[Predictor] = MolgpkaPredictor,
) -> dict[int, float]:
    """Predict basic pKa values with the selected predictor backend."""

    predictor = predictor_cls(mol, device=device)
    return predictor.pred_base()
