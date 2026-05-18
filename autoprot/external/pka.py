from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
import numpy as np

from .descriptor import mol2vec
from .ionization_group import get_ionization_aid
from .net import GCNNet

from autoprot.special_cases import match_smarts, oh_ring_sulfonate, has_nplus_base_proximity, has_invalid_amine, has_phosphate

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

def predict_acid(mol_h: Mol, model_acid: GCNNet, device: str = "cpu"
) -> dict[int, float]:
    """ Predict acid pKas with molgpka model. """

    acid_idxs = get_ionization_aid(mol_h, "acid")
    acid_res = {}
    for aid in acid_idxs:
        apka = model_pred(mol_h, aid, model_acid, device=device)
        acid_res.update({aid:apka})
    return acid_res

def get_acid_neighbors(mol_h: Mol, acid: dict[int, float], verbose: bool = False) -> dict[int, float]:
    """ Find heavy-atom neighbour for acidic proton. """
    acid_heavy = {}

    for at_idx, pka in acid.items():
        H_acid = mol_h.GetAtomWithIdx(at_idx)
        for bond in H_acid.GetBonds():
            neighbor = bond.GetOtherAtom(H_acid)
            neighbor_map_idx = neighbor.GetAtomMapNum()
            if verbose:
                print(f'Neighbor to acid H{at_idx}: {neighbor.GetSymbol()}{neighbor.GetIdx()}')
                print(f'pka: {pka}')
            acid_heavy[neighbor_map_idx] = pka
    return acid_heavy

def predict_base(mol_h: Mol, model_base: GCNNet, device: str = "cpu"
) -> dict[int, float]:
    """ Predict base pKas with molgpka model. """
  
    base_idxs = get_ionization_aid(mol_h, "base")
    base_res = {}
    for aid in base_idxs:
        atom = mol_h.GetAtomWithIdx(aid)
        map_idx = atom.GetAtomMapNum()

        bpka = model_pred(mol_h, aid, model_base, device=device)
        base_res.update({map_idx:bpka}) # changed!
    return base_res

def predict_acid_base(
    mol: Mol,
    model_base: GCNNet,
    model_acid: GCNNet,
    device: str = 'cpu',
    verbose: bool = False,
    pred_acid: bool = True,
    pred_base: bool = True,
    ) -> tuple[dict[int, float], dict[int, float]]:
    """ Wrapper for acid-base prediction with molgpka. """

    ### TODO: ###### ADD CORRECTION FOR PREVIOUS EXCEPTIONS! ####

        # # Inject phosphate clusters:
        # if self.phosphate_groups:
        #     state_strs_poh, state_freqs_poh, oh_ids_poh = special_cases.calc_phosphate_clusters(
        #             self.phosphate_groups,pH,self.matrix_def,
        #     )
        #     for state_strs, state_freqs, oh_ids in zip(state_strs_poh,state_freqs_poh,oh_ids_poh):
        #         state_strs_clusters.append(state_strs)
        #         state_freqs_clusters.append(state_freqs)
        #         indices_clusters.append(oh_ids)

        # if (self.invalid_amine_map_idx > 0) and (self.invalid_amine_map_idx in self.indices0):
        #     pka = 10.4
        #     ss_lower = 1 # basic, state_str 1 -> 2
        #     state_strs_invalid_amine, state_freqs_invalid_amine = special_cases.calc_single_fixed_pka(
        #         pH,
        #         pka,
        #         ss_lower,
        #         self.matrix_def
        #     )
        #     state_strs_clusters.append(state_strs_invalid_amine)
        #     state_freqs_clusters.append(state_freqs_invalid_amine)
        #     indices_clusters.append([self.invalid_amine_map_idx])

        # if len(self.NphenNOO_indices) > 0:
        #     pka = 2.0
        #     ss_lower = 1 # basic
        #     for map_idx in self.NphenNOO_indices:
        #         state_strs_NphenNOO, state_freqs_NphenNOO = special_cases.calc_single_fixed_pka(
        #             pH,pka,ss_lower,self.matrix_def)
        #         state_strs_clusters.append(state_strs_NphenNOO)
        #         state_freqs_clusters.append(state_freqs_NphenNOO)
        #         indices_clusters.append([map_idx])

    mol_h = Chem.rdmolops.AddHs(Chem.Mol(mol))

    atom_indices = [atom.GetIdx() for atom in mol.GetAtoms()]
    qs = np.array([at.GetFormalCharge() for at in mol.GetAtoms()]) # type: ignore

    if pred_base:
        base = predict_base(mol_h,model_base,device=device)

        base_curated = {} # atom mapping

        invalid_amine_map_idx = has_invalid_amine(mol)

        matches_ncncO = match_smarts(mol, '[n;r6][c;r6][nH;r6][c;r6](=O)')

        smarts_NphenNOO = "Ncccc([N+](=O)[O-])"
        matches_NphenNOO = match_smarts(mol,smarts_NphenNOO)

        for at_idx, q in zip(atom_indices, qs):
            if q != 0.:
                continue
            pka = None
            atom = mol.GetAtomWithIdx(at_idx)
            map_idx = atom.GetAtomMapNum()

            # corrections to existing molgpka predictions
            if map_idx in base:
                pka = base[map_idx]
                
                ncat = has_nplus_base_proximity(map_idx, mol, max_distance=5)
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
                        pka = 2.0

            # none of the added hydrogens should be considered here
            if (map_idx > 0) and (map_idx == invalid_amine_map_idx):
                pka = 10.4

            # pattern = Chem.MolFromSmarts(smarts_NN)
            # _, matches = match_pattern(mol, pattern)
            # for match in matches:
            #     if at_idx in match:
            #         pka -= 2.

            # pka -= 2.
            if pka is not None:
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

        matches_ncS = match_smarts(mol, '[n,nH,n+]~[c,C]~[S,s]')

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

        matches_ncncO = match_smarts(mol, '[n;r6][c;r6][nH;r6][c;r6](=O)')

        acid_curated = {} # atom mapping
        # for at_idx, pka in acid.items():
            # atom = mol_h.GetAtomWithIdx(at_idx)

        phosphate_found, phosphate_groups = has_phosphate(mol)
        
        print('-'*20)
        print(Chem.MolToSmiles(mol))
        print(phosphate_found)
        print(phosphate_groups)

        for at_idx, q in zip(atom_indices, qs):
            if q != 0.:
                continue
            pka = None
            atom = mol.GetAtomWithIdx(at_idx)
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
                        matches = match_smarts(mol, smarts)
                        for match in matches:
                            if (at_idx in match):
                                pka += 3.#3.
                                continue
                    oh_sooo_list = oh_ring_sulfonate(mol)
                    if at_idx in oh_sooo_list:
                        pka += 4

                    for smarts in smarts_carbox_close:
                        matches = match_smarts(mol, smarts)
                        for match in matches:
                            if (at_idx in match):
                                pka += 2.
                                continue

            for p_idx, oh_map_ids in phosphate_groups.items():
                if map_idx in oh_map_ids:
                    # if len(oh_map_ids) == 1:
                    #     pka = 2.0
                    # else:
                    #     pka = 6.5
                    other_O_deprotonated = False
                    for atom in mol.GetAtoms():
                        o_map_idx = atom.GetAtomMapNum()
                        if (o_map_idx in oh_map_ids) and (o_map_idx != map_idx): 
                            q_other_O = atom.GetFormalCharge() # check if other O deprotonated
                            if q_other_O == -1.:
                                other_O_deprotonated = True
                                break
                    if other_O_deprotonated:
                        pka = 6.5 # 2.0
                    else:
                        pka = 2.0 # 6.5

                    # matches = match_smarts(mol_h, smarts_ON)
                    # for match in matches:
                    #     if (at_idx in match):
                    #         pka += 0. # 2. # 3.
                    #         continue
                    print(pka)
            if pka is not None:
                acid_curated[map_idx] = pka
        # if verbose:
        # print('acid curated')
        # print(acid_curated)
        acid = acid_curated
    else:
        acid = {}
    return base, acid
