from rdkit import Chem

from rdkit.Chem.MolStandardize import rdMolStandardize

from .external.pka import predict_acid_base, load_model
from .transition_matrix import calc_tmatrix, calc_state_freqs_sparse
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

    smiles = Chem.MolToSmiles(mol,canonical=True)
    if verbose:
        print('Processed:')
        print(smiles)    
    
    return mol, exclude_indices

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

def run_acid_base_calc(state_strs,mols_lib,model_base,model_acid,base_lib,acid_lib,device='cpu',verbose=False):
    """ Add base and acid calculation results for all state_strs into base_lib and acid_lib """

    for state_str in state_strs:
        if state_str not in base_lib:
            if verbose:
                print(state_str)

            mol = mols_lib[state_str]
            mol_h = Chem.rdmolops.AddHs(mol)
            base, acid = predict_acid_base(mol_h,model_base,model_acid,device=device,verbose=verbose)
            base_lib[state_str] = base
            acid_lib[state_str] = acid

    return base_lib, acid_lib

###################################################################################

def calc_state_pkas(state_strs, state_vecs, base_lib, acid_lib,indices,pH=7.,
                verbose=False):
    """ Calc state probabilities from acid/base pka values for given pH """
    
    ps_all = [] # pH specific

    for state_str, state_vec in zip(state_strs, state_vecs):
        if verbose:
            print('='*20)
        ps_up = np.zeros((len(state_vec))) - 1
        ps_down = np.zeros((len(state_vec))) - 1
        base = base_lib[state_str]
        acid = acid_lib[state_str]
        for at_idx, pka in base.items():
            if at_idx not in indices: # Excluded at the start
                continue
            rel_idx = indices.index(at_idx)
            p_up = calc_charge(pka,pH=pH) # probability for higher + state
            p_down = 1 - p_up
            if verbose:
                print(f'at_idx:{at_idx} | rel_idx:{rel_idx} | base {pka} up:{p_up:.2f} stay:{p_down:.2f}')
            if state_vec[rel_idx] <= 1:
                ps_up[rel_idx] = p_up
            else: # already protonated
                ps_down[rel_idx] = p_down
        for at_idx, pka in acid.items():
            if at_idx not in indices: # Excluded at the start
                continue
            rel_idx = indices.index(at_idx)
            p_up = calc_charge(pka,pH=pH)
            p_down = 1. - p_up
            if verbose:
                print(f'at_idx:{at_idx} | rel_idx:{rel_idx} | acid {pka} stay:{p_up:.2f} down:{p_down:.2f}')
            if state_vec[rel_idx] >= 1:
                ps_down[rel_idx] = p_down
            else: # already deprotonated
                ps_up[rel_idx] = p_up

        ps = np.vstack([ps_up,ps_down]) # (up/down, state_idx)
        ps_all.append(ps)

    ps_all = np.array(ps_all)
    return ps_all

#############################################################################################

def coupling_assay(indices,q_options, mol0, mols_lib, smiles_lib, model_base, model_acid, base_lib, acid_lib, coupling_cutoff=1.0, 
                   device='cpu', verbose=False):
    """ Sensitivity analysis. (De-)protonate every site one by one and assess if pKa values of other sites change """

    state_vecs = construct_state_vectors_single(indices, q_options)
    state_strs = calc_state_strs(state_vecs)
    mols_lib, smiles_lib = construct_mols(mol0, state_strs, state_vecs, indices, mols_lib, smiles_lib) # pH independent
    base_lib, acid_lib = run_acid_base_calc(state_strs,mols_lib,model_base,model_acid,base_lib,acid_lib,device=device,verbose=verbose) # pH independent

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

def combine_clusters(state_strs_clusters, state_freqs_clusters, indices_clusters, state_freqs_all, pH_idx, pHs, 
                     sfreq_cutoff_individual=0.01,sfreq_cutoff_combined=0.001,verbose=False):
    """ 
    Combine state frequency results from independent pKa clusters in molecule. 
    The combined probability for microstate A in cluster 1 and B in cluster 2 is:
    p(AB) = p(A)p(B) 
    """

    # Cull the state_strs per cluster a bit before combining.
    # This is quite conservative (everything with at least 1% freq in that cluster)

    cluster_state_ids = []
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
        if state_freq < sfreq_cutoff_combined:
            # combination of microstates very unlikely
            continue
        state_str = sort_string(state_str,ps)
        state_strs.append(state_str)
        state_freqs.append(state_freq)

    if verbose:
        print(f'N chosen microstate combinations: {len(state_strs)}')
    # Correct freqs for removal of very unlikely states
    state_freqs = np.array(state_freqs)
    state_freqs /= np.sum(state_freqs)

    for state_str, state_freq in zip(state_strs, state_freqs):
        if state_str not in state_freqs_all:
            state_freqs_all[state_str] = np.zeros(len(pHs))
        state_freqs_all[state_str][pH_idx] = state_freq
    
    return indices, state_strs, state_freqs_all

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

def run_pipeline(name,smiles_raw,pH_output=7,cutoff_states=4000,device='cpu',
                 pH_band=8.,pHs = np.arange(0,14.1,0.5),
                 path_out='output',path_figs='figures',
                 verbose=False,cutoff_export=0.5,
                 fout_csv='out.csv',append=True,notebook=False,
                 except_optimize_error=False):
                #  write_all_relevant=False):
    if verbose:
        print(name)
        print(smiles_raw, flush=True)

    os.makedirs(path_out,exist_ok=True)
    os.makedirs(path_figs,exist_ok=True)

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

    mols_frag_lib = {}
    base_frag_lib = {}
    acid_frag_lib = {}

    mol0, exclude_indices = preprocess(smiles_raw)
    mol0_h = Chem.rdmolops.AddHs(mol0)

    base0, acid0 = predict_acid_base(mol0_h,model_base,model_acid,device=device,verbose=verbose)
        
    for pH_idx, pH in enumerate(pHs): #,total=len(pHs)):#,total=len(pHs)):
        if verbose:
            print('='*50)
            print(f'pH: {pH}',flush=True)
        indices0, q_options0 = find_candidate_sites(base0, acid0, exclude_indices,pH,pH_band=pH_band,verbose=False)

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

            indices = [indices0[c] for c in cluster] # indices0[cluster]
            indices_str = ''
            for id in indices:
                indices_str += f'{id},'
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

            mols_frag_lib[indices_str], smiles_frag_lib = construct_mols(mol0, state_strs, state_vecs, indices, mols_frag_lib[indices_str], smiles_frag_lib) # pH independent
            base_frag_lib[indices_str], acid_frag_lib[indices_str] = run_acid_base_calc(state_strs,mols_frag_lib[indices_str],model_base,model_acid,
                                                                                        base_frag_lib[indices_str],acid_frag_lib[indices_str],device=device) # pH independent

            ps_all = calc_state_pkas(state_strs, state_vecs, base_frag_lib[indices_str], acid_frag_lib[indices_str], indices, pH=pH,
                                                        verbose=False)
            N_states = len(state_vecs)

            tmatrix = calc_tmatrix(state_vecs,state_strs,ps_all,N_states)
            state_freqs = calc_state_freqs_sparse(tmatrix)
            
            state_strs_clusters.append(state_strs)
            state_freqs_clusters.append(state_freqs)
            indices_clusters.append(indices)

        indices, state_strs, state_freqs_all = combine_clusters(
            state_strs_clusters, state_freqs_clusters, indices_clusters, state_freqs_all, pH_idx, pHs,verbose=verbose)

        state_vecs = [unpack_vec(state_str) for state_str in state_strs]
        mols_lib, smiles_lib = construct_mols(mol0, state_strs, state_vecs, indices, mols_lib, smiles_lib) # pH independent
        state_freq_max = 0.
        net_charge = 0.
        
        state_qs = {}

        for state_str, state_freq in state_freqs_all.items():
            if state_freq[pH_idx] > state_freq_max:
                state_str_opti = state_str
                state_freq_max = state_freq[pH_idx]
            state_q = Chem.GetFormalCharge(mols_lib[state_str]) 
            state_qs[state_str] = state_q
            net_charge += state_q * state_freq[pH_idx]

        state_strs_export = []
        state_freqs_export = []

        for state_str, state_freq in state_freqs_all.items():
            if state_freq[pH_idx] > cutoff_export * state_freq_max: # Include all high prob states
                state_strs_export.append(state_str)
                state_freqs_export.append(state_freq[pH_idx])

        state_freqs_export = np.array(state_freqs_export)
        ps = np.argsort(state_freqs_export)[::-1] # Sort by highest probability

        state_freqs_export = state_freqs_export[ps]
        state_strs_export = [state_strs_export[p] for p in ps]

        net_charges.append(net_charge)

        if pH == pH_output:
            if verbose:
                print(f'Export at pH {pH_output}:',flush=True)
                for e_idx, (state_str, sfreq) in enumerate(zip(state_strs_export, state_freqs_export)):
                    print(e_idx, state_str, sfreq)
            # export_smi(name,state_strs_export,smiles_lib,state_freqs_export,path=path_out,fout_smi=fout_smi,append=append)
            export_csv(name,state_strs_export,smiles_lib,state_freqs_export,state_qs,path=path_out,fout_csv=fout_csv,append=append)
            return_code = export_sdf(name,state_strs_export,mols_lib,path=path_out,except_optimize_error=except_optimize_error)
            # export_smi(name_state,smiles_lib[state_str_opti],path=path_out)
            plot_optimal_state(name,mols_lib[state_strs_export[0]],path=path_figs)
            if verbose:
                print(f'Optimal smiles for pH {pH}: {smiles_lib[state_str_opti]}')

    net_charges = np.round(np.array(net_charges),decimals=4)

    # with open(f'output/{name}_net_charges.txt','w') as f:
        # for pH, net_charge in zip(pHs, net_charges):
            # f.write(f'{pH:.2f} {net_charge:.3f}\n')

    # reduce number of microstates for plotting
    N_relevant_states, state_strs_relevant, sfreqs_relevant, mols_relevant, sfreqs_not_relevant = calc_relevant_states(state_freqs_all, mols_lib,verbose=verbose)

    if N_relevant_states > 0:
        plot_pH_scan(name, indices, state_strs_relevant, sfreqs_relevant, pHs, net_charges, sfreqs_not_relevant, path=path_figs,verbose=verbose)
        plot_relevant_states(name, mols_relevant, path=path_figs,notebook=notebook)
        compose_image(name,N_relevant_states, path=path_figs)
    return return_code
        # if write_all_relevant:
        #     for state_str, mol in zip(state_strs_relevant, mols_relevant):
        #         export_sdf(f'{name}_{state_str}',mol,path=path_out)
        #         export_smi(f'{name}_{state_str}',smiles_lib[state_str],path=path_out)

