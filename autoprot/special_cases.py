from .utils import *
from .transitions import calc_state_diffs, calc_freqs_from_states
import numpy as np
from rdkit import Chem

def match_pattern(mol,pattern):
    found = mol.HasSubstructMatch(pattern)
    matches = mol.GetSubstructMatches(pattern)
    return found, matches

def add_exclusions(mol,verbose=False):
    """ Exclusions act on the q_options level. Exclusions are removed from consideration
    for protonation/deprotonation. This is specified separately for acids and bases.
    For example, only a protonation event could be excluded for a given index,
    but not the deprotonation event.
    This does not affect the indices or cluster splitting"""

    q0s = np.array([at.GetFormalCharge() for at in mol.GetAtoms()])

    exclude_acid_indices = []
    exclude_base_indices = []
    mol_h = Chem.rdmolops.AddHs(mol)

    pattern_carbonyl = Chem.MolFromSmarts("NC(=O)")
    found_carbonyl, matches_carbonyl = match_pattern(mol,pattern_carbonyl)
    pattern_imine = Chem.MolFromSmarts("NC(=N)")
    found_imine, matches_imine = match_pattern(mol,pattern_imine)
    pattern_sulfonamide = Chem.MolFromSmarts("NS(=O)(=O)")
    found_sulfonamide, matches_sulfonamide = match_pattern(mol,pattern_sulfonamide)

    for at_idx, q in enumerate(q0s):
        atom = mol_h.GetAtomWithIdx(at_idx) 
        map_idx = atom.GetAtomMapNum()

        if q == 0.:
            # atom = mol_h.GetAtomWithIdx(at_idx)
            if atom.GetSymbol() == 'N':
                if atom.GetIsAromatic():
                    if atom.GetDegree() == 3: # arom. N with lone pair needed for ring
                        exclude_base_indices.append(map_idx)
                else:
                    for match in matches_carbonyl: # ...N-C(=O)...
                        if atom.GetIdx() in match:
                            print('Excluding N next to carbonyl as base')
                            if map_idx not in exclude_base_indices:
                                exclude_base_indices.append(map_idx)
                            print(at_idx, map_idx)
                    for match in matches_imine: # ...N-C(=N)...
                        if atom.GetIdx() in match:
                            accept = True
                            for bond in atom.GetBonds(): # Find the correct of the two Ns
                                if bond.GetBondType() == Chem.BondType.DOUBLE:
                                    accept = False
                            if accept:
                                print('Excluding N next to imine as base')
                                if map_idx not in exclude_base_indices:
                                    exclude_base_indices.append(map_idx)
                                print(at_idx, map_idx)
                    for match in matches_sulfonamide:
                        if atom.GetIdx() in match:
                            print('Excluding N next to sulfonamide as base')
                            if map_idx not in exclude_base_indices:
                                exclude_base_indices.append(map_idx)
                            print(at_idx, map_idx)

    exclude_base_indices = sorted(exclude_base_indices)
    exclude_acid_indices = sorted(exclude_acid_indices)

    return exclude_base_indices, exclude_acid_indices
    
def add_exceptions(mol,verbose=False):
    """ This de-couples indices from other indices and treats them as separate clusters,
    possibly with special rules (e.g. hard-coded phosphates).
    This does not remove (exclude) the (de)protonation per se."""

    except_indices = []

    # Except everything that couldn't be neutralized
    q0s = np.array([at.GetFormalCharge() for at in mol.GetAtoms()])
    for at_idx, q in enumerate(q0s):
        atom = mol.GetAtomWithIdx(at_idx) 
        map_idx = atom.GetAtomMapNum()
        if q != 0.:
            print(f'NOTE: Input molecule is charged at idx {at_idx}, map_idx {map_idx}!')
            if map_idx not in except_indices:
                except_indices.append(map_idx)

    phosphate_found, phosphate_groups = has_phosphate(mol) # returns map indices

    if phosphate_found:
        # ids_phosphate = phosphate_matches(mol)
        if verbose:
            print(f'phosphate ids: {phosphate_groups}')
        for p_idx, oh_ids in phosphate_groups.items():
            oh_ids = sorted(oh_ids)
            # deprotonate_ohs.append(oh_ids[0])
            for map_idx in oh_ids:
                if map_idx not in except_indices:
                    except_indices.append(map_idx)

    return except_indices, phosphate_groups

def has_phosphate(mol):

    pattern = Chem.MolFromSmarts("P(=O)(O)(O)")

    found = mol.HasSubstructMatch(pattern)
    matches = mol.GetSubstructMatches(pattern)

    phosphate_groups = {}

    for match in matches:
        # Find central P of phosphate
        for idx in match:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == "P":
                p_map_idx = atom.GetAtomMapNum()
                if p_map_idx not in phosphate_groups:
                    phosphate_groups[p_map_idx] = []
        # Find protonable O of phosphate
        for idx in match:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == "O" and atom.GetTotalNumHs() > 0:
                oh_map_idx = atom.GetAtomMapNum()
                if oh_map_idx not in phosphate_groups[p_map_idx]:
                    phosphate_groups[p_map_idx].append(oh_map_idx)

    return found, phosphate_groups

def split_exceptions(indices, q_options, except_indices):
    indices_curated = []
    q_options_curated = []
    for map_idx, q_option in zip(indices, q_options):
        if map_idx not in except_indices:
            indices_curated.append(map_idx)
            q_options_curated.append(q_option)
    q_options_curated = np.array(q_options_curated)
    return indices_curated, q_options_curated

def calc_phosphate_clusters(phosphate_groups,pH,matrix_def,
                            verbose=False):
    """ Special treatment of phosphates as separate cluster"""

    state_strs_poh = []
    state_freqs_poh = []
    oh_ids_poh = []
    
    pka1 = 2.0
    pka2 = 6.5

    poh_acid_pkas_single = {
        '0' : [pka1],
        '1' : [pka1],
    }

    base_lib_poh_single = {
        '0': {},
        '1': {},
    }

    poh_acid_pkas_double = {
        '00' : [pka2, pka2],
        '01' : [pka1, pka2],
        '10' : [pka2, pka1],
        '11' : [pka1, pka1],
    }

    base_lib_poh_double = {
        '00': {},
        '01': {},
        '10': {},
        '11': {},
    }

    for p_idx, oh_ids in phosphate_groups.items():
        if len(oh_ids) == 1:
            state_strs = ['0','1']
            state_vecs = [unpack_vec(state_str) for state_str in state_strs]
            poh_acid_pkas = poh_acid_pkas_single
            base_lib_poh = base_lib_poh_single
        elif len(oh_ids) == 2:
            state_strs = ['00','01','10','11']
            state_vecs = [unpack_vec(state_str) for state_str in state_strs]
            poh_acid_pkas = poh_acid_pkas_double
            base_lib_poh = base_lib_poh_double
        else:
            print(f'Did not find protonable O for phosphate {p_idx}')
            continue
        acid_lib_poh = {}
        for key, val in poh_acid_pkas.items():
            acid_lib_poh[key] = {}
            for jdx, oh_id in enumerate(oh_ids):
                acid_lib_poh[key][oh_id] = val[jdx]

        if verbose:
            print(oh_ids)
            print(acid_lib_poh)
        
        ps_all = calc_state_diffs(state_strs, state_vecs, oh_ids, base_lib_poh, acid_lib_poh, 
                                    pH=pH,matrix_def=matrix_def,verbose=verbose)
        
        state_strs, state_freqs = calc_freqs_from_states(state_strs,state_vecs,ps_all,matrix_def)

        state_strs_poh.append(state_strs)
        state_freqs_poh.append(state_freqs)
        oh_ids_poh.append(oh_ids)

    return state_strs_poh, state_freqs_poh, oh_ids_poh