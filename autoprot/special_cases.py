from .utils import *
from .transitions import calc_state_diffs, calc_freqs_from_states
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from typing import Any

def match_pattern(mol,pattern):
    """ Match pattern in rdkit molecule.
    
    Parameters:
    -----------
    mol : Mol
        Rdkit molecule
    pattern: str
        String to match.

    Returns
    -------
    found : bool
        Boolean returning if at least one match was found.
    matches : list[list[int]]
        Atom indices for each match.
    """

    found = mol.HasSubstructMatch(pattern)
    matches = mol.GetSubstructMatches(pattern)
    return found, matches

def add_exclusions(mol: Mol, verbose: bool = False) -> tuple[list[int], list[int]]:
    """ 
    Exclusions act on the q_options level. Exclusions are removed from consideration
    for protonation/deprotonation. This is specified separately for acids and bases.
    For example, only a protonation event could be excluded for a given index,
    but not the deprotonation event.
    This does not affect the indices or cluster splitting/
    
    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Input molecule with atom mapping numbers used to track sites.
    verbose : bool, optional
        If True, print diagnostic information about applied exclusions.

    Returns
    -------
    exclude_base_indices : list[int]
        Atom map indices for which protonation (base behavior) should
        be excluded.
    exclude_acid_indices : list[int]
        Atom map indices for which deprotonation (acid behavior) should
        be excluded.
    """

    q0s = np.array([at.GetFormalCharge() for at in mol.GetAtoms()])

    exclude_acid_indices: list[int] = []
    exclude_base_indices: list[int] = []
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

        if q == 0:
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
    
def add_exceptions(mol: Mol, verbose: bool = False) -> tuple[list[int], dict[int, list[int]]]:
    """
    Identify indices that should be treated as separate protonation clusters.

    Exceptions decouple specific sites from the normal clustering procedure. 
    This is used for cases such as special functional groups
    (e.g., phosphates) that require dedicated treatment.

    Returns
    -------
    except_indices : list[int]
        Atom map indices that should be removed from normal clustering.
    phosphate_groups : dict[int, list[int]]
        Mapping from phosphate P atom map indices to their protonable OH
        atom map indices.
    """

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

def has_phosphate(mol: Mol) -> tuple[bool, dict[int, list[int]]]:
    """
    Detect phosphate groups and their protonable hydroxyl atoms.

    Searches the molecule for phosphate motifs and returns the atom map
    indices of the central phosphorus atoms along with the indices of
    attached protonable oxygen atoms.

    Returns
    -------
    found : bool
        True if at least one phosphate group is detected.
    phosphate_groups : dict[int, list[int]]
        Mapping from phosphate P atom map indices to protonable OH
        atom map indices.
    """

    pattern = Chem.MolFromSmarts("P(=O)(O)(O)")

    found, matches = match_pattern(mol,pattern)

    phosphate_groups: dict[int, list[int]] = {}

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

def split_exceptions(indices: list[int], q_options: np.ndarray, except_indices: list[int]) -> tuple[list[int], np.ndarray]:
    """
    Remove exception indices from candidate protonation sites.

    Filters the list of candidate indices and their corresponding
    protonation options by removing sites that are handled separately
    (e.g., special functional groups).

    Returns
    -------
    indices_curated : list[int]
        Filtered list of atom map indices.
    q_options_curated : np.ndarray
        Protonation option matrix corresponding to the curated indices.
    """

    indices_curated = []
    q_options_curated = []
    for map_idx, q_option in zip(indices, q_options):
        if map_idx not in except_indices:
            indices_curated.append(map_idx)
            q_options_curated.append(q_option)
    q_options_curated_arr = np.array(q_options_curated)
    return indices_curated, q_options_curated_arr

def calc_phosphate_clusters(
    phosphate_groups: dict[int, list[int]],
    pH: float,
    matrix_def: str,
    verbose: bool = False,
) -> tuple[list[list[str]], list[np.ndarray], list[list[int]]]:
    """
    Compute protonation state distributions for phosphate groups.

    Phosphates are treated as independent clusters with predefined
    pKa values. This function enumerates possible protonation states
    of the phosphate OH groups and calculates their equilibrium
    populations at the given pH.

    Returns
    -------
    state_strs_poh : list[list[str]]
        Protonation state encodings for each phosphate cluster.
    state_freqs_poh : list[np.ndarray]
        Corresponding microstate frequencies.
    oh_ids_poh : list[list[int]]
        Atom map indices of protonable OH atoms for each cluster.
    """

    state_strs_poh = []
    state_freqs_poh = []
    oh_ids_poh = []
    
    pka1 = 2.0
    pka2 = 6.5

    poh_acid_pkas_single: dict[str, list[float]] = {
        '0' : [pka1],
        '1' : [pka1],
    }

    base_lib_poh_single: dict[str, Any] = {
        '0': {},
        '1': {},
    }

    poh_acid_pkas_double: dict[str, list[float]] = {
        '00' : [pka2, pka2],
        '01' : [pka1, pka2],
        '10' : [pka2, pka1],
        '11' : [pka1, pka1],
    }

    base_lib_poh_double: dict[str, Any] = {
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
        acid_lib_poh: dict[str, dict[int, float]] = {}
        for key, val in poh_acid_pkas.items():
            acid_lib_poh[key] = {}
            for jdx, oh_id in enumerate(oh_ids):
                acid_lib_poh[key][oh_id] = val[jdx]

        if verbose:
            print(oh_ids)
            print(acid_lib_poh)
        
        ps_all = calc_state_diffs(state_strs, state_vecs, oh_ids, base_lib_poh, acid_lib_poh, 
                                    pH=pH,matrix_def=matrix_def,verbose=verbose)
        
        state_strs_curated, state_freqs = calc_freqs_from_states(state_strs,state_vecs,ps_all,matrix_def)

        state_strs_poh.append(state_strs_curated)
        state_freqs_poh.append(state_freqs)
        oh_ids_poh.append(oh_ids)

    return state_strs_poh, state_freqs_poh, oh_ids_poh