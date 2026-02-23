from rdkit import Chem

from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import RegistrationHash

from .external.pka import predict_acid_base, load_model
from .transition_matrix import *
from .postprocess import *
from .utils import *
from .coupling import *

import numpy as np

import copy
import itertools
import os

from importlib import resources
pkg_base = resources.files('autoprot')

ROOT = f'{pkg_base}/data'

def preprocess(smiles_raw,verbose=False):
    """ 
    Build and preprocess mol.
    Neutralize all molecule charges. If not possible (e.g. quaternary amine),
    add to exclusion list to avoid (de-)protonation attempts. 
    """

    if verbose:
        print('Raw:')
        print(smiles_raw)
    mol = Chem.MolFromSmiles(smiles_raw, sanitize=True)
    # print(mol)
    smiles = Chem.MolToSmiles(mol,canonical=True)
    
    if verbose:
        print('Canonical')
        print(smiles)
    mol = Chem.MolFromSmiles(smiles, sanitize=True)

    if verbose:
        print('Formal charges before cleanup')
        charges = [at.GetFormalCharge() for at in mol.GetAtoms()]
        print(charges)

    mol = rdMolStandardize.Cleanup(mol)

    reionizer = rdMolStandardize.Reionizer()
    mol = reionizer.reionize(mol)

    uncharger = rdMolStandardize.Uncharger()
    mol = uncharger.uncharge(mol)

    if verbose:
        print('Formal charges after cleanup')
        symbols = [at.GetFormalCharge() for at in mol.GetAtoms()]
        print(symbols)

    q0s = np.array([at.GetFormalCharge() for at in mol.GetAtoms()])

    exclude_indices = []
    mol_h = Chem.rdmolops.AddHs(mol)
    for at_idx, q in enumerate(q0s):
        if q != 0.:
            print(f'NOTE: Input molecule is charged at idx {at_idx}!')
            exclude_indices.append(at_idx)
        else:
            atom = mol_h.GetAtomWithIdx(at_idx)
            if atom.GetSymbol() == 'N':
                if atom.GetIsAromatic():
                    if atom.GetDegree() == 3: # arom. N with lone pair needed for ring
                        exclude_indices.append(at_idx)
    
    phosphate_found, phosphate_ohs = has_phosphate(mol)

    # deprotonate_ohs = []

    if phosphate_found:
        # ids_phosphate = phosphate_matches(mol)
        if verbose:
            print(f'phosphate ids: {phosphate_ohs}')
        for p_idx, oh_ids in phosphate_ohs.items():
            oh_ids = sorted(oh_ids)
            # deprotonate_ohs.append(oh_ids[0])
            for at_idx in oh_ids:
                if at_idx not in exclude_indices:
                    exclude_indices.append(at_idx)
    # print(f'deprotonate ids: {deprotonate_ohs}')

    exclude_indices = sorted(exclude_indices)

    smiles = Chem.MolToSmiles(mol,canonical=True)
    if verbose:
        print('Processed:')
        print(smiles)
        print(f'exclude indices: {exclude_indices}')

    return mol, exclude_indices, phosphate_ohs

def has_phosphate(mol):

    # patterns = [
        # Chem.MolFromSmarts("P(=O)(O)(O)"),
        # Chem.MolFromSmarts("P(=O)(O)(O)O"),
        # Chem.MolFromSmarts("P(=O)(O)([O-])"),
        # Chem.MolFromSmarts("P(=O)(O)([O-])O"),
        # Chem.MolFromSmarts("P(=O)([O-])([O-])"),
        # Chem.MolFromSmarts("P(=O)([O-])([O-])O"),
    # ]

    pattern = Chem.MolFromSmarts("P(=O)(O)(O)")

    # found = any(mol.HasSubstructMatch(p) for p in patterns)
    # matches = [mol.GetSubstructMatches(p) for p in patterns]

    found = mol.HasSubstructMatch(pattern)
    matches = mol.GetSubstructMatches(pattern)

    # print(f'matches: ', matches)

    phosphate_ohs = {}

    for match in matches:
        poh_indices = []
        # Find central P of phosphate
        for idx in match:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == "P":
                p_idx = idx
                if p_idx not in phosphate_ohs:
                    phosphate_ohs[p_idx] = []
        # Find protonable O of phosphate
        for idx in match:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == "O" and atom.GetTotalNumHs() > 0:
                if idx not in phosphate_ohs[p_idx]:
                    phosphate_ohs[p_idx].append(idx)

    return found, phosphate_ohs

# def calc_phosphate_ids(mol):#: str) -> bool:

#     patterns = [
#         Chem.MolFromSmarts("P(=O)(O)(O)"),
#         Chem.MolFromSmarts("P(=O)(O)([O-])"),
#         Chem.MolFromSmarts("P(=O)([O-])(O)"),
#         Chem.MolFromSmarts("P(=O)([O-])([O-])")
#     ]

#     return 


def find_candidate_sites(base,acid,exclude_indices,pH,pH_band=6.,
                         verbose=False):
    """ Find possible (de-)protonation sites """

    prot_candidates = list(base.keys())
    deprot_candidates = list(acid.keys())

    indices = list(sorted(set(prot_candidates + deprot_candidates)))
    if verbose:
        print(f'relevant indices: {indices}')

    for idx in exclude_indices:
        if idx in indices:
            indices.remove(idx)
    if verbose:
        print(f'relevant indices (after exclusion): {indices}')

    q_options = np.zeros((3,len(indices))) # deprot=0, stay=1, prot=2
    q_options[1] = 1 # always allow stay
    for rel_idx, at_idx in enumerate(indices):
        if at_idx in prot_candidates:
            if base[at_idx] >= pH - pH_band:
                q_options[2,rel_idx] = 1 # allow protonation
        if at_idx in deprot_candidates:
            if acid[at_idx] <= pH + pH_band:
                q_options[0,rel_idx] = 1 # allow deprotonation

    return indices, q_options

def construct_state_vectors(q_options, cutoff_states, verbose=False):
    """ Enumerate all combinations of state_vectors allowed by q_options """

    # print('Constructing state vectors...')
    # tfs = np.array(list(itertools.product([0,1,2], repeat=len(indices))))
    # tfs = itertools.product([0,1,2], repeat=len(indices))
    
    q_options_nonzero = []
    for rel_idx, qs in enumerate(q_options.T):
        q_col = []
        for q_idx, q in enumerate(qs):
            if q == 1.:
                q_col.append(q_idx)
        if len(q_col) > 0.:
            q_options_nonzero.append(q_col)
    
    N_trial_vecs = np.prod([len(qs) for qs in q_options_nonzero])
    if N_trial_vecs > cutoff_states:
        return []
    else:
        state_vecs = np.array(list(itertools.product(*q_options_nonzero)))#[0,1,2], repeat=len(indices))
        return state_vecs

def construct_mol(mol0, indices, state_vec):
    """ Make single rdkit mol object from 'neutral' mol0 and state vector """

    mol_cand = copy.deepcopy(mol0)
    qs = state_vec - 1
    
    rw = Chem.RWMol(Chem.AddHs(mol_cand))

    for at_idx, q in zip(indices,qs):
        atom = rw.GetAtomWithIdx(at_idx)
        atom.SetFormalCharge(int(q))
        if q == -1:
            for nbr in atom.GetNeighbors():
                if nbr.GetAtomicNum() == 1:
                    rw.RemoveAtom(nbr.GetIdx())
                    break

    mol_cand = Chem.RemoveHs(rw)
    Chem.SanitizeMol(mol_cand)
    smiles_cand = Chem.MolToSmiles(mol_cand, canonical=False)  
    mol_cand = Chem.MolFromSmiles(smiles_cand, sanitize=True)
    return mol_cand, smiles_cand
    
def construct_mols(mol0, state_strs, state_vecs, indices, mols_lib, smiles_lib):
    """ Make rdkit mol objects for all protonation states defined in state_strs """

    for state_str, state_vec in zip(state_strs, state_vecs):
        if state_str not in mols_lib:
            mol_cand, smiles_cand = construct_mol(mol0, indices, state_vec)
            mol_cand.SetProp("_Name",state_str)
            mols_lib[state_str] = mol_cand
            smiles_lib[state_str] = smiles_cand
    return mols_lib, smiles_lib

def calc_charge(pka,pH=7.):
    """ Hendersson-Hasselbalch eq. """

    ppos = 1. / ( 1 + 10**(pH-pka) ) # fraction of more positively charged res
    return ppos

def run_acid_base_calc(state_strs,state_vecs,indices,mols_lib,model_base,model_acid,base_lib,acid_lib,device='cpu',verbose=False):
    """ Add base and acid calculation results for all state_strs into base_lib and acid_lib """
    # print(indices)
    for state_str, state_vec in zip(state_strs,state_vecs):
        if state_str not in base_lib:
            if verbose:
                print(state_str)

            state_vec_base = np.maximum(state_vec,1)
            state_str_base = pack_vec(state_vec_base)

            # print(f'Using base state_str: {state_str_base}')

            mol_base = mols_lib[state_str_base]
            mol_base_h = Chem.rdmolops.AddHs(mol_base)

            base_tmp, _ = predict_acid_base(mol_base_h,model_base,model_acid,device=device,
                                        pred_acid=False,verbose=verbose)
            base = {}
            for at_idx, b in base_tmp.items():
                if at_idx not in indices:
                    continue
                rel_idx = indices.index(at_idx)
                if state_vec[rel_idx] == 1:
                    base[at_idx] = b

            state_vec_acid = np.minimum(state_vec,1)
            state_str_acid = pack_vec(state_vec_acid)

            # print(f'Using acid state_str: {state_str_acid}')

            mol_acid = mols_lib[state_str_acid]
            mol_acid_h = Chem.rdmolops.AddHs(mol_acid)

            _, acid_tmp = predict_acid_base(mol_acid_h,model_base,model_acid,device=device,
                                           pred_base=False,verbose=verbose)
            
            acid = {}
            for at_idx, a in acid_tmp.items():
                if at_idx not in indices:
                    continue
                rel_idx = indices.index(at_idx)
                if state_vec[rel_idx] == 1:
                    acid[at_idx] = a
            # mol = mols_lib[state_str]
            # mol_h = Chem.rdmolops.AddHs(mol)
            # base, acid = predict_acid_base(mol_h,model_base,model_acid,device=device,
            #                                verbose=verbose)
            base_lib[state_str] = base
            acid_lib[state_str] = acid
        # print('base')
        # print(base_lib[state_str])
        # print('acid')
        # print(acid_lib[state_str])
    return base_lib, acid_lib

###################################################################################

# def calc_phosphate_pkas()

# def calc_state_pkas(state_strs, state_vecs, base_lib, acid_lib, indices, pH=7.,
#                 verbose=False):
#     """ Calc state probabilities from acid/base pka values for given pH """
    
#     ps_all = [] # pH specific

#     for state_str, state_vec in zip(state_strs, state_vecs):
#         if verbose:
#             print('='*20)
#         ps_up = np.zeros((len(state_vec))) - 1
#         ps_down = np.zeros((len(state_vec))) - 1
#         base = base_lib[state_str]
#         acid = acid_lib[state_str]
#         for at_idx, pka in base.items():
#             if at_idx not in indices: # Excluded at the start
#                 continue
#             rel_idx = indices.index(at_idx)
#             p_up = calc_charge(pka,pH=pH) # probability for higher + state
#             p_down = 1 - p_up
#             if verbose:
#                 print(f'at_idx:{at_idx} | rel_idx:{rel_idx} | base {pka} up:{p_up:.2f} stay:{p_down:.2f}')
#             if state_vec[rel_idx] <= 1:
#                 ps_up[rel_idx] = p_up
#             else: # already protonated
#                 ps_down[rel_idx] = p_down
#         for at_idx, pka in acid.items():
#             if at_idx not in indices: # Excluded at the start
#                 continue
#             rel_idx = indices.index(at_idx)
#             p_up = calc_charge(pka,pH=pH)
#             p_down = 1. - p_up
#             if verbose:
#                 print(f'at_idx:{at_idx} | rel_idx:{rel_idx} | acid {pka} stay:{p_up:.2f} down:{p_down:.2f}')
#             if state_vec[rel_idx] >= 1:
#                 ps_down[rel_idx] = p_down
#             else: # already deprotonated
#                 ps_up[rel_idx] = p_up

#         ps = np.vstack([ps_up,ps_down]) # (up/down, state_idx)
#         ps_all.append(ps)

#     ps_all = np.array(ps_all)
#     return ps_all

def calc_state_pkas(state_strs, state_vecs, base_lib, acid_lib, indices, pH=7.,matrix_def='dG',
                verbose=False):
    """ Calc state probabilities from acid/base pka values for given pH """

    ps_all = [] # pH specific

    for state_str, state_vec in zip(state_strs, state_vecs):
        if verbose:
            print('='*20)
            print(f'{state_str}')
        ps_up = {}
        ps_down = {}
        # ps_up = np.empty((len(state_vec)))
        # ps_up.fill(np.nan)
        # ps_down = np.empty((len(state_vec)))
        # ps_down.fill(np.nan)
        # ps_up = np.zeros((len(state_vec))) - 1
        # ps_down = np.zeros((len(state_vec))) - 1
        base = base_lib[state_str]
        acid = acid_lib[state_str]
        for at_idx, pka in base.items():
            if at_idx not in indices: # Excluded at the start
                continue
            rel_idx = indices.index(at_idx)
            if matrix_def == 'msm':
                p_up = calc_charge(pka,pH=pH) # probability for higher + state
                p_down = 1 - p_up
            elif matrix_def == 'dG':
                p_up = np.log(10) * (pH - pka) # -ln(p+/p0)
                p_down = np.log(10) * (pka - pH) # -ln(p0/p+)
            else:
                raise
            if verbose:
                print(f'rel_idx:{rel_idx} | at_idx:{at_idx} | base {pka} up:{p_up:.2f} stay:{p_down:.2f}')
            if state_vec[rel_idx] <= 1:
                ps_up[rel_idx] = p_up
            # else:
                # if verbose:
                    # print('already protonated')
            # else: # already protonated
                # ps_down[rel_idx] = p_down
        for at_idx, pka in acid.items():
            if at_idx not in indices: # Excluded at the start
                continue
            rel_idx = indices.index(at_idx)
            if matrix_def == 'msm':
                p_up = calc_charge(pka,pH=pH) # probability for higher + state
                p_down = 1 - p_up
            elif matrix_def == 'dG':
                p_up = np.log(10) * (pH - pka) # -ln(p+/p0)
                p_down = np.log(10) * (pka - pH) # -ln(p0/p+)
            else:
                raise
            if verbose:
                print(f'rel_idx:{rel_idx} | at_idx:{at_idx} | acid {pka} stay:{p_up:.2f} down:{p_down:.2f}')
            if state_vec[rel_idx] >= 1:
                ps_down[rel_idx] = p_down
            # else:
                # if verbose:
                    # print('already deprotonated')
            # else: # already deprotonated
                # ps_up[rel_idx] = p_up
        ps = {
            'up' : ps_up,
            'down' : ps_down,
        }
        # ps = np.vstack([ps_up,ps_down]) # (up/down, state_idx)
        ps_all.append(ps)

    # ps_all = np.array(ps_all)
    return ps_all

#############################################################################################

def coupling_assay(indices,q_options, mol0, mols_lib, smiles_lib, model_base, model_acid, base_lib, acid_lib, coupling_cutoff=1.0, 
                   device='cpu', verbose=False):
    """ Sensitivity analysis. (De-)protonate every site one by one and assess if pKa values of other sites change """

    state_vecs = construct_state_vectors_single(indices, q_options)
    state_strs = calc_state_strs(state_vecs)
    mols_lib, smiles_lib = construct_mols(mol0, state_strs, state_vecs, indices, mols_lib, smiles_lib) # pH independent
    base_lib, acid_lib = run_acid_base_calc(state_strs,state_vecs,indices,mols_lib,model_base,model_acid,base_lib,acid_lib,device=device,verbose=verbose) # pH independent

    state_str0 = state_strs[0]
    base_pka_diffs = {}
    acid_pka_diffs = {}
    for state_str1 in state_strs[1:]:
        base_pka_diffs[state_str1], acid_pka_diffs[state_str1] = compare_pkas(indices, q_options, state_str0, state_str1, base_lib, acid_lib)
    coupling_matrix = construct_coupling_matrix(indices, state_strs, state_vecs, base_pka_diffs, acid_pka_diffs, coupling_cutoff=coupling_cutoff)
    clusters = cluster_coupling_matrix(coupling_matrix)
    return clusters, mols_lib, base_lib, acid_lib

def sort_string(string,ps):
    """ Sort string by custom indices ps """

    s = list(string)
    s = [s[p] for p in ps]
    s = "".join(s)
    return s

def combine_clusters(state_strs_clusters, state_freqs_clusters, indices_clusters, 
                     sfreq_cutoff_individual=0.01,sfreq_cutoff_combined=0.001,verbose=False):
    """ 
    Combine state frequency results from independent pKa clusters in molecule. 
    The combined probability for microstate A in cluster 1 and B in cluster 2 is:
    p(AB) = p(A)p(B) 
    """

    # Cull the state_strs per cluster a bit before combining.
    # This is quite conservative (everything with at least 1% freq in that cluster)

    cluster_state_ids = []

    # print(state_freqs_clusters)

    for state_freqs in state_freqs_clusters:
        cluster_state_ids.append([])
        for s_idx, s_freq in enumerate(state_freqs):
            if s_freq >= sfreq_cutoff_individual:
                cluster_state_ids[-1].append(s_idx)

    # cluster_state_idxs = [list(range(len(state_strs))) for state_strs in state_strs_clusters]
    combinations = list(itertools.product(*cluster_state_ids))
    if verbose:
        print(f'N microstate combinations from clusters: {len(combinations)}')
    # print(state_freqs_clusters)

    state_strs = []
    state_freqs = []
    indices = []
    for indices_cluster in indices_clusters:
        indices.extend(indices_cluster) # NEEDS SORTING SOMEHOW

    ps = np.argsort(indices)
    indices = [indices[p] for p in ps]
    
    for s_idxs in combinations:
        state_str = ''
        state_freq = 1.
        for c_idx, s_idx in enumerate(s_idxs):
            state_str += state_strs_clusters[c_idx][s_idx]
            state_freq *= state_freqs_clusters[c_idx][s_idx]                   
        if state_freq >= sfreq_cutoff_combined:
            state_str = sort_string(state_str,ps)
            state_strs.append(state_str)
            state_freqs.append(state_freq)

    if verbose:
        print(f'N chosen microstate combinations: {len(state_strs)}')
    # Correct freqs for removal of very unlikely states
    state_freqs = np.array(state_freqs)
    state_freqs /= np.sum(state_freqs)

    state_freqs_lib = {}

    for state_str, state_freq in zip(state_strs, state_freqs):
        state_freqs_lib[state_str] = state_freq

    return indices, state_strs, state_freqs_lib

def calc_relevant_states(state_freqs_all, mols_lib, max_states=18,verbose=False):
    """ Reduce number of states to max_states for plotting """

    cutoff = 0.05
    tries = 0
    if len(state_freqs_all.keys()) == 0:
        return 0, [], [], []
    state_strs_relevant = []
    sfreqs_relevant = []
    mols_relevant = []
    pH_argmaxs = []
    sfreqs_not_relevant = []

    for state_str, sfreqs in state_freqs_all.items():
        if np.max(sfreqs) > cutoff:
            state_strs_relevant.append(state_str)
            sfreqs_relevant.append(sfreqs)
            mols_relevant.append(mols_lib[state_str])
            pH_argmaxs.append(np.argmax(sfreqs))
        else:
            sfreqs_not_relevant.append(sfreqs)
    N_relevant_states = len(state_strs_relevant)
    if verbose:
        print(f'Initial N relevant states: {N_relevant_states} with cutoff {cutoff}')
    while N_relevant_states > max_states:
        state_strs_relevant = []
        sfreqs_relevant = []
        mols_relevant = []
        pH_argmaxs = []
        tries += 1
        cutoff += 0.02
        for state_str, sfreqs in state_freqs_all.items():
            if np.max(sfreqs) > cutoff:
                state_strs_relevant.append(state_str)
                sfreqs_relevant.append(sfreqs)
                mols_relevant.append(mols_lib[state_str])
                pH_argmaxs.append(np.argmax(sfreqs))
            else:
                sfreqs_not_relevant.append(sfreqs)
        N_relevant_states = len(state_strs_relevant)

    # SORT HERE
    ps = np.argsort(pH_argmaxs)
    state_strs_relevant = [state_strs_relevant[p] for p in ps]
    sfreqs_relevant = [sfreqs_relevant[p] for p in ps]
    mols_relevant = [mols_relevant[p] for p in ps]
    if verbose:
        print(f'Final N relevant states: {N_relevant_states} with cutoff {cutoff}')
    return N_relevant_states, state_strs_relevant, sfreqs_relevant, mols_relevant, sfreqs_not_relevant

def screen_clusters(indices0, q_options0, mol0, mols_lib, smiles_lib, 
                                            model_base, model_acid, base_lib, acid_lib, cutoff_states, device='cpu',
                                            verbose=False):
    """ Screen fragmentation of molecule into different pKa clusters """
    accept_clusters = False
    coupling_cutoff = 0.0
    while not accept_clusters:
        if verbose:
            print(f'coupling cutoff: {coupling_cutoff}')
        accept_clusters = True
        clusters, mols_lib, base_lib, acid_lib = coupling_assay(indices0, q_options0, mol0, mols_lib, smiles_lib, 
                                            model_base, model_acid, base_lib, acid_lib, coupling_cutoff=coupling_cutoff, device=device,
                                            verbose=verbose)
        for c_idx, cluster in enumerate(clusters):
            q_options = q_options0[:,cluster]
            state_vecs = construct_state_vectors(q_options, cutoff_states)
            N_states = len(state_vecs)
            if N_states > cutoff_states: # This should never happen, as construct_state_vectors should return an empty list in that case.
                raise
            if (N_states == 0):# or (N_states > cutoff_states):
                accept_clusters = False
                coupling_cutoff += 0.2
    if coupling_cutoff > 1.5:
        print(f'Coupling cutoff high: {coupling_cutoff}')
    return clusters, mols_lib, base_lib, acid_lib

def smiles2hash(smiles: str | None) -> str | None:
    if smiles is None:
        return None
    return RegistrationHash.GetMolHash(RegistrationHash.GetMolLayers(Chem.MolFromSmiles(smiles)))

def mol2hash(mol):
    return RegistrationHash.GetMolHash(RegistrationHash.GetMolLayers(mol))

def calc_hashes(state_strs,mols_lib):
    hashes = []
    for state_str in state_strs:
        mol = mols_lib[state_str]
        hash = mol2hash(mol)
        hashes.append(hash)
    return hashes

def calc_symmetry(state_strs, state_freqs_lib, mols_lib,verbose=False):
    state_hashes = calc_hashes(state_strs,mols_lib)
    state_dict = {}

    for state_str, state_hash in zip(state_strs,state_hashes):
        if state_hash in state_dict:
            state_dict[state_hash].append(state_str)
        else:
            state_dict[state_hash] = [state_str]

    if verbose:
        print(state_dict)

    state_strs_symm = []
    state_freqs_symm = []

    for state_hash, state_strs_per_hash in state_dict.items():
        state_strs_sorted = sorted(state_strs_per_hash)
        state_strs_symm.append(state_strs_sorted[0])
        state_freq = 0.
        for state_str in state_strs_per_hash:
            state_freq += state_freqs_lib[state_str]
        state_freqs_symm.append(state_freq)
    
    state_strs = state_strs_symm
    state_freqs = state_freqs_symm

    return state_strs, state_freqs

def run_pipeline(name,smiles_raw,pH_output=7,cutoff_states=4000,device='cpu',
                 pH_band=8.,pHs = np.arange(0,14.1,0.5),
                 path_out='output',path_figs='figures',
                 verbose=False,cutoff_export=0.5,
                 fout_csv='out.csv',append=True,notebook=False,
                 except_optimize_error=False,
                 matrix_def='msm',export_opti_sdf=False):
                #  write_all_relevant=False):
    if verbose:
        print(name)
        print(smiles_raw, flush=True)

    os.makedirs(path_out,exist_ok=True)
    os.makedirs(path_figs,exist_ok=True)

    return_code = 0

    # molgpka ML models
    model_file_base = f'{ROOT}/weight_base.pth'
    model_file_acid = f'{ROOT}/weight_acid.pth'
    model_base = load_model(model_file_base,device=device)
    model_acid = load_model(model_file_acid,device=device)

    base_lib = {}
    acid_lib = {}
    smiles_lib = {}
    mols_lib = {}

    net_charges = []
    state_freqs_all = {}
    freqs_macro_all = []

    mols_frag_lib = {}
    base_frag_lib = {}
    acid_frag_lib = {}

    mol0, exclude_indices, phosphate_ohs = preprocess(smiles_raw,verbose=verbose)
    mol0_h = Chem.rdmolops.AddHs(mol0)

    base0, acid0 = predict_acid_base(mol0_h,model_base,model_acid,device=device,verbose=verbose)
        
    for pH_idx, pH in enumerate(pHs): #,total=len(pHs)):#,total=len(pHs)):
        # print('='*50)
        # print(f'pH: {pH}',flush=True)
        if verbose:
            print('='*50)
            print(f'pH: {pH}',flush=True)
        indices0, q_options0 = find_candidate_sites(base0, acid0, exclude_indices,pH,pH_band=pH_band,verbose=False)
        if verbose:
            print(f'indices0: {indices0}')

        # Screen coupling between residues
        clusters, mols_lib, base_lib, acid_lib = screen_clusters(indices0, q_options0, mol0, mols_lib, smiles_lib, 
                                            model_base, model_acid, base_lib, acid_lib, cutoff_states, device='cpu',
                                            verbose=verbose)
        if verbose:
            print(f'Clusters: {clusters}')
        state_freqs_clusters = []
        state_strs_clusters = []
        indices_clusters = []
        
        for c_idx, cluster in enumerate(clusters):

            smiles_frag_lib = {}

            # atom indices for cluster
            indices = [indices0[c] for c in cluster] # indices0[cluster]
            indices_str = ''
            for id in indices:
                indices_str += f'{id},'
            # indices_str shows what atoms the cluster state_vec and state_str refer to
            indices_str = indices_str[:-1]

            if indices_str not in mols_frag_lib:
                mols_frag_lib[indices_str] = {}
            if indices_str not in base_frag_lib:
                base_frag_lib[indices_str] = {}
            if indices_str not in acid_frag_lib:
                acid_frag_lib[indices_str] = {}
            
            q_options = q_options0[:,cluster]
            state_vecs = construct_state_vectors(q_options, cutoff_states)
            N_states = len(state_vecs)

            state_strs = calc_state_strs(state_vecs)

            mols_frag_lib[indices_str], smiles_frag_lib = construct_mols(
                mol0, state_strs, state_vecs, indices, mols_frag_lib[indices_str], smiles_frag_lib) # pH independent

            base_frag_lib[indices_str], acid_frag_lib[indices_str] = run_acid_base_calc(state_strs,state_vecs,indices,mols_frag_lib[indices_str],model_base,model_acid,
                                                                                        base_frag_lib[indices_str],acid_frag_lib[indices_str],
                                                                                        device=device,verbose=verbose) # pH independent

            ps_all = calc_state_pkas(state_strs, state_vecs, base_frag_lib[indices_str], acid_frag_lib[indices_str], indices, pH=pH,matrix_def=matrix_def,
                                                        verbose=verbose)
            N_states = len(state_vecs)

            if matrix_def == 'msm':
                tmatrix = calc_tmatrix(state_vecs,state_strs,ps_all,N_states)
                state_freqs = calc_state_freqs_sparse(tmatrix)
            elif matrix_def == 'dG':
                # print(state_strs)
                dGmatrix = calc_dGmatrix(state_vecs,state_strs,ps_all,N_states)
                # print(dGmatrix)
                # Fs = calc_Fs(dGmatrix)
                # print(dGmatrix)
                dG_clusters = find_dGclusters(dGmatrix)
                # print(dG_clusters)
                state_strs, dGmatrix = remove_orphans(dG_clusters, state_strs, dGmatrix)
                # print(state_strs, dGmatrix)
                is_connected = check_connectivity(dGmatrix)
                if not is_connected:
                    raise ValueError('Matrix not connected')
                Fs = reconstruct_free_energies_incomplete_half(dGmatrix)
                # print(Fs)
                state_freqs = calc_populations(Fs)
                # print(state_strs)
                # print(Fs)
            # print(state_freqs)
            # state_freqs = calc_state_freqs_sparse(tmatrix)
            
            state_strs_clusters.append(state_strs)
            state_freqs_clusters.append(state_freqs)
            indices_clusters.append(indices)

        # Inject phosphate clusters:
        
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

        for p_idx, oh_ids in phosphate_ohs.items():
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
            
            ps_all = calc_state_pkas(state_strs, state_vecs, base_lib_poh, acid_lib_poh, oh_ids, pH=pH,matrix_def=matrix_def,verbose=verbose)
            N_states = len(state_vecs)
            if matrix_def == 'msm':
                tmatrix = calc_tmatrix(state_vecs,state_strs,ps_all,N_states)
                state_freqs = calc_state_freqs_sparse(tmatrix)
            elif matrix_def == 'dG':
                dGmatrix = calc_dGmatrix(state_vecs,state_strs,ps_all,N_states)
                # print(dGmatrix)
                # Fs = calc_Fs(dGmatrix)
                # print(dGmatrix)
                dG_clusters = find_dGclusters(dGmatrix)
                # print(dG_clusters)
                state_strs, dGmatrix = remove_orphans(dG_clusters, state_strs, dGmatrix)
                # print(state_strs, dGmatrix)
                is_connected = check_connectivity(dGmatrix)
                if not is_connected:
                    raise ValueError('Matrix not connected')
                Fs = reconstruct_free_energies_incomplete_half(dGmatrix)
                # print(Fs)
                state_freqs = calc_populations(Fs)
            # tmatrix = calc_tmatrix(state_vecs, state_strs,ps_all, N_states)
            # state_freqs = calc_state_freqs_sparse(tmatrix)

            state_strs_clusters.append(state_strs)
            state_freqs_clusters.append(state_freqs)
            indices_clusters.append(oh_ids)

        # print(state_strs_clusters)
        # print(state_freqs_clusters)

        indices, state_strs, state_freqs_lib = combine_clusters(
            state_strs_clusters, state_freqs_clusters, indices_clusters, verbose=verbose)

        # print(state_strs)
        # print([state_freqs_lib[state_str] for state_str in state_strs])

        state_vecs = [unpack_vec(state_str) for state_str in state_strs]
        mols_lib, smiles_lib = construct_mols(mol0, state_strs, state_vecs, indices, mols_lib, smiles_lib) # pH independent

        # Symmetry
        state_strs, state_freqs = calc_symmetry(state_strs, state_freqs_lib, mols_lib,verbose=verbose)
        
        # print(state_strs)
        # print(state_freqs)

        # Max freq
        idx_max = np.argmax(state_freqs)
        state_freq_max = np.max(state_freqs)
        state_str_opti = state_strs[idx_max]

        # Net charge as weighted sum over microstate charges
        state_qs = {}
        freqs_macro = {}
        net_charge = 0.       
        for state_str, state_freq in zip(state_strs, state_freqs):
            state_q = Chem.GetFormalCharge(mols_lib[state_str])
            state_qs[state_str] = state_q
            if state_q in freqs_macro:
                freqs_macro[state_q] += state_freq
            else:
                freqs_macro[state_q] = state_freq
            net_charge += state_q * state_freq
        net_charges.append(net_charge)

        freqs_macro_all.append(freqs_macro)


        # Add to results for pH scan
        for state_str, state_freq in zip(state_strs,state_freqs):
            if state_str not in state_freqs_all:
                state_freqs_all[state_str] = np.zeros(len(pHs))
            state_freqs_all[state_str][pH_idx] = state_freq

        # Select states for pH-specific export
        state_strs_export = []
        state_freqs_export = []
        for state_str, state_freq in zip(state_strs, state_freqs):
            if state_freq > cutoff_export * state_freq_max: # Include all high prob states
                state_strs_export.append(state_str)
                state_freqs_export.append(state_freq)

        state_freqs_export = np.array(state_freqs_export)
        ps = np.argsort(state_freqs_export)[::-1] # Sort by highest probability

        state_freqs_export = state_freqs_export[ps]
        state_strs_export = [state_strs_export[p] for p in ps]

        # Output pH-specific results for pH_output
        if pH == pH_output:
            if verbose:
                print(f'Export at pH {pH_output}:',flush=True)
                for e_idx, (state_str, sfreq) in enumerate(zip(state_strs_export, state_freqs_export)):
                    print(e_idx, state_str, sfreq)
            # export_smi(name,state_strs_export,smiles_lib,state_freqs_export,path=path_out,fout_smi=fout_smi,append=append)
            export_csv(name,state_strs_export,smiles_lib,state_freqs_export,state_qs,path=path_out,fout_csv=fout_csv,append=append)
            return_code = export_sdf(name,state_strs_export,mols_lib,path=path_out,except_optimize_error=except_optimize_error)
            # export_smi(name_state,smiles_lib[state_str_opti],path=path_out)
            if export_opti_sdf:
                plot_optimal_state(name,mols_lib[state_strs_export[0]],path=path_figs)
            if verbose:
                print(f'Optimal smiles for pH {pH}: {smiles_lib[state_str_opti]}')

    # Plotting of pH scan
    net_charges = np.round(np.array(net_charges),decimals=4)

    pkas_macro = {}
    pkas_weights = {}

    for pH, freqs_macro in zip(pHs,freqs_macro_all):
        qs_sorted = sorted(freqs_macro.keys())
        # print(qs_sorted)
        # print(pH, freqs_macro)
        for q in qs_sorted:
            if q+1 in qs_sorted:
                freq1 = freqs_macro[q]
                freq2 = freqs_macro[q+1]
                pka_macro = np.log10(freq2/freq1) + pH
                pka_weight = 1./(freq1**2 + freq2**2)
                if q in pkas_macro:
                    pkas_macro[q].append(pka_macro)
                    pkas_weights[q].append(pka_weight)
                else:
                    pkas_macro[q] = [pka_macro]
                    pkas_weights[q] = [pka_weight]
                # print(q, q+1, pka_macro, pka_weight)
    
    pkas_combined = {}

    for q, pkas in pkas_macro.items():
        ws = pkas_weights[q]
        pka_comb = float(np.average(pkas,weights=ws))
        pkas_combined[q] = pka_comb

    for idx, (q, pka) in enumerate(pkas_combined.items()):
        print(f'pKa{idx+1} | {q+1} --> {q} | {pka:.3f}')

    if len(pkas_combined) > 0:
        export_macro_pkas(name,pkas_combined,path=path_out)

    # print(pkas_combined)

    # with open(f'output/{name}_net_charges.txt','w') as f:
        # for pH, net_charge in zip(pHs, net_charges):
            # f.write(f'{pH:.2f} {net_charge:.3f}\n')

    # reduce number of microstates for plotting
    N_relevant_states, state_strs_relevant, sfreqs_relevant, mols_relevant, sfreqs_not_relevant = calc_relevant_states(state_freqs_all, mols_lib, verbose=verbose)

    if N_relevant_states > 0:
        plot_pH_scan(name, indices, state_strs_relevant, sfreqs_relevant, pHs, net_charges, sfreqs_not_relevant, path=path_figs,verbose=verbose)
        plot_relevant_states(name, mols_relevant, path=path_figs,notebook=notebook)
        compose_image(name,N_relevant_states, path=path_figs)
    return return_code
        # if write_all_relevant:
        #     for state_str, mol in zip(state_strs_relevant, mols_relevant):
        #         export_sdf(f'{name}_{state_str}',mol,path=path_out)
        #         export_smi(f'{name}_{state_str}',smiles_lib[state_str],path=path_out)

