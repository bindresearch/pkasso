from svgutils.compose import Figure, SVG
import cairosvg # import before anything in rdkit.Chem.Draw, breaks otherwise !!

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import MolsToGridImage, MolToImage, MolToFile

from rdkit.Chem.MolStandardize import rdMolStandardize

import torch
from .external.ionization_group import get_ionization_aid
from .external.descriptor import mol2vec
from .external.net import GCNNet

import numpy as np
from scipy.sparse import csr_matrix

import copy
import itertools
import os
import time

import matplotlib.pyplot as plt
from tqdm import tqdm

def load_model(model_file, device="cpu"):
    model= GCNNet().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    model.eval()
    return model

def model_pred(m2, aid, model, device="cpu"):
    data = mol2vec(m2, aid)
    with torch.no_grad():
        data = data.to(device)
        pKa = model(data)
        pKa = pKa.cpu().numpy()
        pka = pKa[0][0]
    return pka

def predict_acid(mol,model_acid, device="cpu"):
    acid_idxs= get_ionization_aid(mol, acid_or_base="acid")
    acid_res = {}
    for aid in acid_idxs:
        apka = model_pred(mol, aid, model_acid, device=device)
        acid_res.update({aid:apka})
    return acid_res

def predict_base(mol,model_base, device="cpu"):
  
    base_idxs= get_ionization_aid(mol, acid_or_base="base")
    base_res = {}
    for aid in base_idxs:
        bpka = model_pred(mol, aid, model_base, device=device)
        base_res.update({aid:bpka})
    return base_res

def preprocess(smiles_raw,verbose=False):
    if verbose:
        print('Raw:')
        print(smiles_raw)
    mol = Chem.MolFromSmiles(smiles_raw, sanitize=True)
    smiles = Chem.MolToSmiles(mol,canonical=True)
    if verbose:
        print('Canonical')
        print(smiles)
    mol = Chem.MolFromSmiles(smiles, sanitize=True)

    if verbose:
        print('Formal charges before cleanup')
        symbols = [at.GetFormalCharge() for at in mol.GetAtoms()]
        print(symbols)

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
            print('####')
            print(f'WARNING: Input molecule is charged at idx {at_idx}!')
            print('####')
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
    
    return mol, q0s, exclude_indices

def predict_acid_base(mol_h,model_base,model_acid,device='cpu',verbose=False):

    base = predict_base(mol_h,model_base,device=device)
    acid = predict_acid(mol_h,model_acid,device=device)

    if verbose:
        print('base')
        print(base)
        print('acid H')
        print(acid)

    acid = get_acid_neighbors(mol_h, acid)

    if verbose:
        print('acid heavy')
        print(acid)
    return base, acid

def get_acid_neighbors(mol_h, acid, verbose=False):
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

def find_candidate_sites(base,acid,exclude_indices,pH,pH_band=6.,
                         verbose=False):
    prot_candidates = list(base.keys())
    deprot_candidates = list(acid.keys())

    indices = list(sorted(set(prot_candidates + deprot_candidates)))
    if verbose:
        print(f'relevant indices: {indices}')
    # print(exclude_indices)
    for idx in exclude_indices:
        if idx in indices:
            indices.remove(idx)
    if verbose:
        print(f'relevant indices (after exclusion): {indices}')

    ### ADD A PH-DEPENDENT PKA CHECK HERE?

    q_options = np.zeros((3,len(indices))) # deprot=0, stay=1, prot=2
    q_options[1] = 1 # always allow stay
    for rel_idx, at_idx in enumerate(indices):
        if at_idx in prot_candidates:
            if base[at_idx] > pH - pH_band:
                q_options[2,rel_idx] = 1 # allow protonation
        if at_idx in deprot_candidates:
            if acid[at_idx] < pH + pH_band:
                q_options[0,rel_idx] = 1 # allow deprotonation
    # print(q_options)
    return indices, q_options

# def plot_mols(mols):
#     img=Draw.MolsToGridImage(mols,molsPerRow=5,subImgSize=(200,200),returnPNG=False,useSVG=True) # legends=[x.GetProp("_Name") for x in ms],
#     with open(f'test.svg','w') as f:
#         f.write(img.data)
#     cairosvg.svg2pdf(url=f'test.svg',write_to=f'test.pdf')
#     return img

def construct_state_vectors(indices, q_options, verbose=False):
    tfs = np.array(list(itertools.product([0,1,2], repeat=len(indices))))
    # state_vecs = []
    state_vecs = []
    for trial_vec in tfs:
        # print(trial_vec)
        # trial_vec = np.zeros((len(indices)))
        accept = True
        for rel_idx, s in enumerate(trial_vec):
            if s == 1: # uncharged always included
                continue
            elif (s == 0) and (q_options[0][rel_idx] == 0): # don't allow deprotonation
                accept = False
                break
            elif (s == 2) and (q_options[2][rel_idx] == 0): # don't allow protonation
                accept = False
                break
        if accept:
            if verbose:
                print(trial_vec)
            state_vecs.append(trial_vec)
    return state_vecs

def construct_mol(mol0, indices, state_vec):
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

    # mols = []

    for state_str, state_vec in zip(state_strs, state_vecs):
        if state_str not in mols_lib:
            mol_cand, smiles_cand = construct_mol(mol0, indices, state_vec)
            mol_cand.SetProp("_Name",state_str)
            mols_lib[state_str] = mol_cand
            smiles_lib[state_str] = smiles_cand
    return mols_lib, smiles_lib

        # mol_cand = copy.deepcopy(mol0)
        # qs = state_vec - 1
        
        # if verbose:
        #     print('-----')
        #     print(f'qs: {qs}')

        # rw = Chem.RWMol(Chem.AddHs(mol_cand))

        # for at_idx, q in zip(indices,qs):
        #     # print(at_idx, q)
        #     atom = rw.GetAtomWithIdx(at_idx)
        #     atom.SetFormalCharge(int(q))
        #     # print(atom.GetNumImplicitHs())
        #     if q == -1:
        #         if verbose:
        #             print(f'idx: {at_idx}')
        #             print(f'Explicit Hs: {atom.GetNumExplicitHs()}')
        #             print(f'Implicit Hs: {atom.GetNumImplicitHs()}')
        #         h_explicit = atom.GetNumExplicitHs()
        #         h_implicit = atom.GetNumImplicitHs()
                
        #         for nbr in atom.GetNeighbors():
        #             if nbr.GetAtomicNum() == 1:
        #                 rw.RemoveAtom(nbr.GetIdx())
        #                 break
        #     # if h_explicit > 0:
        #     #     atom.SetNumExplicitHs((h_explicit - 1))
        #     # elif h_implicit > 0:
        #     #     atom.SetNumImplicitHs((h_implicit - 1))

        #     # atom.SetNumExplicitHs((atom.GetNumExplicitHs() - 1))
        # mol_cand = Chem.RemoveHs(rw)
        # # mol_cand = rw.GetMol()
        # Chem.SanitizeMol(mol_cand)
        # symbols = [at.GetSymbol() for at in mol_cand.GetAtoms()]
        # charges = [at.GetFormalCharge() for at in mol_cand.GetAtoms()]
        # smiles_cand = Chem.MolToSmiles(mol_cand, canonical=False)
        # if verbose:
        #     print(symbols)
        #     print(charges)
        #     print(smiles_cand)

        # smiles_lib[state_str] = smiles_cand        
        # mol_cand = Chem.MolFromSmiles(smiles_cand, sanitize=True)
        # # AllChem.EmbedMultipleConfs(mol_cand,numConfs=1,randomSeed=np.random.randint(1,1000),useRandomCoords=True)
        # # AllChem.UFFOptimizeMoleculeConfs(mol_cand)
        # mols.append(mol_cand)
        # mols_lib[state_str] = mol_cand
    # return mols, mols_lib, smiles_lib #, smiles_all

def calc_charge(pka,pH=7.):
    ppos = 1. / ( 1 + 10**(pH-pka) ) # fraction of more positively charged res
    dq = 0
    return ppos
    
def pack_vec(state_vec):
    state_str = "".join([str(x) for x in state_vec])
    return state_str

def unpack_vec(state_str):
    state_vec = np.array([int(s) for s in state_str],dtype=int)
    return state_vec

def run_acid_base_calc(state_strs,mols_lib,model_base,model_acid,base_lib,acid_lib,device='cpu',verbose=False):
    # mols_h = []
    # bases = []
    # acids = []

    # n_cached = 0

    for state_str in state_strs:
        if state_str not in base_lib:
            if verbose:
                print(state_str)
            # base = base_lib[state_str]
            # acid = acid_lib[state_str]
            # n_cached += 1
        # else:
            mol = mols_lib[state_str]
            mol_h = Chem.rdmolops.AddHs(mol)
            base, acid = predict_acid_base(mol_h,model_base,model_acid,device=device,verbose=verbose)
            base_lib[state_str] = base
            acid_lib[state_str] = acid
        # bases.append(base)
        # acids.append(acid)
    # print(f'Cached calculations used: {n_cached}')
    return base_lib, acid_lib

###################################################################################

def calc_state_pkas(state_strs, state_vecs, base_lib, acid_lib,indices,pH=7.,
                verbose=False):
    
    ps_all = [] # pH specific

    for state_str, state_vec in zip(state_strs, state_vecs):
    # for state_vec, base, acid in zip(state_vecs, bases, acids):
        if verbose:
            print('='*20)
        # print(qs, base, acid)
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
            # p_down = 1 - p_up # probability for lower + state
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
            else:
                ps_up[rel_idx] = p_up

        ps = np.vstack([ps_up,ps_down]) # (up/down, state_idx)
        ps_all.append(ps)

    ps_all = np.array(ps_all)

    return ps_all

def calc_tmatrix(state_vecs,state_strs,ps_all,N_states):
    """ Transition matrix between molecule protonation states"""

    # print(len(state_strs),len(state_vecs),len(ps_all), N_states)

    tmatrix_raw = [[[] for _ in range(N_states)] for _ in range(N_states)] # N_states x N_states (x duplicate predictions)

    nonzero_entries = []

    for s_idx, state_vec in enumerate(state_vecs):
        # print(f'{s_idx} {state_vec}')
        ps_up = ps_all[s_idx,0]
        ps_down = ps_all[s_idx,1]

        recipes = [
            [ps_up, 1],
            [ps_down, -1]
        ]

        for rec in recipes:
            ps = rec[0]
            dq = rec[1]

            for at_idx, p in enumerate(ps):
                # self.log(at_idx, p)
                if p > -1.:
                    p = float(p)
                    state_target_vec = state_vec.copy()
                    state_target_vec[at_idx] += dq
                    state_target_str = pack_vec(state_target_vec)
                    # print('target', state_target_str)

                    if state_target_str in state_strs:
                        c_target_idx = state_strs.index(state_target_str)
                        # c_target_idx = self.get_s_idx(state_target,self.state_strs)
                        # tmatrix_raw[c_target_idx][s_idx].append(p) # to, from
                        # tmatrix_raw[s_idx][c_target_idx].append(1-p)
                        tmatrix_raw[s_idx][c_target_idx].append(p) # from, to; row-stochastic
                        tmatrix_raw[c_target_idx][s_idx].append(1-p)
                        nonzero_entries.append([s_idx,c_target_idx])
                        nonzero_entries.append([c_target_idx,s_idx])
                    # else:
                        # print(f'Transition from {state_strs[s_idx]} to {state_target_str} ignored (target out of original pKa range)',flush=True)
                        # raise

    tmatrix_mean = np.zeros((N_states,N_states)) # N_states x N_states, average over predictions per ij

    for idx, jdx in nonzero_entries:
        tmatrix_mean[idx,jdx] = np.mean(tmatrix_raw[idx][jdx])
    # for idx, row in enumerate(tmatrix_raw):
        # for jdx, col in enumerate(row):
            # if len(col) > 0:
                # tmatrix_mean[idx,jdx] = np.mean(col)

    tmatrix = tmatrix_mean.copy()

    # for idx, col in enumerate(tmatrix_mean.T):
    for idx, row in enumerate(tmatrix_mean):
        tmatrix[idx,idx] = np.prod(-1.*row+1) # probability not to transition

    # Normalized tmatrix (probabilities from one state to all other states sums to 1)
    tmatrix_norm = []
    for row in tmatrix:
        # print(row)
        tmatrix_norm.append(row / np.sum(row))
    
    tmatrix_norm = np.array(tmatrix_norm)
    # tmatrix = np.sum(tmatrix,axis=1)

    return tmatrix_norm

def simulate_state_traj(state_vecs,tmatrix,nsteps=100000):
    """ Simulate evolution according to transition matrix """

    # print(f'Len state_vecs: {len(state_vecs)}',flush=True)
    choice0 = np.random.randint(len(state_vecs))

    traj = [choice0]
    traj_states = [state_vecs[choice0]]

    for t in range(1,nsteps):
        # trans = tmatrix.T[traj[t-1]]
        trans = tmatrix[traj[t-1]]
        choice = np.random.choice(np.arange(len(state_vecs)),p=trans)
        # self.log(traj[t-1], trans, choice)
        traj.append(choice)
        traj_states.append(state_vecs[choice])

    traj_states = np.array(traj_states)
    traj = np.array(traj)

    return traj_states, traj

def calc_optimal_state(state_freqs,state_strs,qs_all,verbose=False):
    # state_vecs_mean = np.mean(traj_states,axis=0)

    # state_freqs = np.zeros((N_states))

    # for idx in range(N_states):
        # state_freqs[idx] = len(np.where(traj == idx)[0])
    # state_freqs /= np.sum(state_freqs)
    # self.state_freqs *= 100
    # self.log(self.state_freqs)

    net_charge = 0.
    for s_idx, state_freq in enumerate(state_freqs):
        qs = qs_all[s_idx]
        net_charge += (np.sum(qs) * state_freq)

    if verbose:
        print('State | State (relevant) | Occupancy')
        for state_str, freq in zip(state_strs, state_freqs):
            print(f'{state_str} | {freq*100:.2f}%')

    idx_max = np.argmax(state_freqs)

    # state_vec_opti = state_vecs[idx_max]
    state_str_opti = state_strs[idx_max]

    # self.state_vec_opti = self.reconstruct_full_state(self.unpack_vec(self.states_str[idx_max]))
    # self.state_str_opti = self.pack_vec(self.state_vec_opti)

    if verbose:
        print(f'Most likely state: {state_str_opti} at {state_freqs[idx_max]*100:.2f}%')
    return state_str_opti, net_charge

############################################################################

def get_relevant_states(state_strs, state_freqs_all, mols_lib, cutoff=0.05):
    state_strs_relevant = []
    sfreqs = state_freqs_all.T
    sfreqs_relevant = []
    mols_relevant = []
    for idx, (state_str, sfreq) in enumerate(zip(state_strs,sfreqs)):
        if np.max(sfreq) > cutoff:
            state_strs_relevant.append(state_str)
            sfreqs_relevant.append(sfreq)
            s_idx = state_strs.index(state_str)
            mols_relevant.append(mols_lib[state_str])

    return state_strs_relevant, sfreqs_relevant, mols_relevant

def plot_pH_scan(name, indices, state_strs_relevant, sfreqs_relevant, pHs, net_charges, cmap=plt.cm.Spectral,
                 fsave=''):
    cmap = plt.cm.Spectral

    print(f'Indices: {indices}')
    px = 1/plt.rcParams['figure.dpi']

    fig, ax = plt.subplots(2,1,figsize=(700*px,500*px),height_ratios=[0.6,0.4])

    for idx, (state_str, sfreq) in enumerate(zip(state_strs_relevant,sfreqs_relevant)):
        color = cmap(idx/(len(state_strs_relevant)-1))
        ax[0].plot(pHs,sfreq*100,label=state_str,color=color)
    if len(state_strs_relevant) > 10:
        ax[0].legend(ncol=2,fontsize=6)
    else:
        ax[0].legend(ncol=1,fontsize=8)
    ax[0].set(xlabel='pH',ylabel='Distribution %')
    ax[0].set(xlim=(pHs[0],pHs[-1]))
    ax[0].set_title(name)
    ax[0].set_xticks(np.arange(pHs[0],pHs[-1],1))
    ax[0].grid(alpha=0.3)

    ax[1].plot(pHs,net_charges,color='black')
    ax[1].set(xlabel='pH', ylabel='Net charge')
    ax[1].set(xlim=(pHs[0],pHs[-1]))
    ax[1].grid(alpha=0.3)
    ax[1].set_xticks(np.arange(pHs[0],pHs[-1],1))

    fig.tight_layout()
    if fsave != '':
        fig.savefig(fsave)
    plt.close()
    # return fig

def export_sdf(name,mol_h,path='output'):
    AllChem.EmbedMolecule(mol_h, randomSeed=1, useRandomCoords=True)
    AllChem.UFFOptimizeMolecule(mol_h)
    with Chem.SDWriter(f'{path}/{name}.sdf') as f:
        f.write(mol_h)

def export_smi(name,smiles,path='output'):
    with open(f'{path}/{name}.smi','w') as f:
        f.write(f'{smiles} {name}\n')

# from rdkit.Chem.Draw import rdMolDraw2D

# fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# # RDKit molecules
# Draw.MolsToGridImage(
#     mols_relevant,
#     molsPerRow=3,
#     subImgSize=(200, 200),
#     legends=[x.GetProp("_Name") for x in mols_relevant],
#     ax=ax[0]
# )

def plot_relevant_states(name, mols_relevant,path='figures'):
# def plot_relevant_states(name, mols_relevant):
    for mol in mols_relevant: tmp=AllChem.Compute2DCoords(mol)
    
    img=Draw.MolsToGridImage(mols_relevant,molsPerRow=4,subImgSize=(150,150),legends=[x.GetProp("_Name") for x in mols_relevant],returnPNG=False,useSVG=True)
    # img=Draw.MolsToGridImage(mols_relevant,molsPerRow=3,subImgSize=(200,200),legends=[x.GetProp("_Name") for x in mols_relevant],
                            #  ax=ax)
    # return ax
    # print(type(img))
    with open(f'{path}/{name}_relevant_states.svg','w') as f:
        # f.write(img.data)
        f.write(img)
    
    # cairosvg.svg2pdf(url=f'{path}/{name}_relevant_states.svg',
                    #  write_to=f'{path}/{name}_relevant_states.pdf')
    # os.system(f'rm {path}/{name}_relevant_states.svg')
    # return img

def plot_optimal_state(name, mol, path='figures'):
    # print('plotting svg')
    tmp=AllChem.Compute2DCoords(mol)
    MolToFile(mol, f'{path}/{name}_opti.svg', size=(800,630), imageType='svg')
    # print('converting to pdf')
    cairosvg.svg2pdf(url=f'{path}/{name}_opti.svg',write_to=f'{path}/{name}_opti.pdf')
    # print('removing svg')
    os.system(f'rm {path}/{name}_opti.svg')

def compose_image(name,N_relevant_states,path='figures'):
    if N_relevant_states % 4 == 0:
        y = 350 + (N_relevant_states//4) * 150
    else:
        y = 350 + (N_relevant_states//4 + 1) * 150
    # print(path)
    # print(name)
    # print(f"{y}px")
    Figure(
        "600px", f"{y}px",
        SVG(f'{path}/{name}_ph_scan.svg').move(30, 0),
        SVG(f'{path}/{name}_relevant_states.svg').move(0, 350)
    ).save(f'{path}/{name}_combined.svg')

    cairosvg.svg2pdf(url=f'{path}/{name}_combined.svg',write_to=f'{path}/{name}_combined.pdf')
    os.system(f'rm {path}/{name}_ph_scan.svg')
    os.system(f'rm {path}/{name}_relevant_states.svg')
    os.system(f'rm {path}/{name}_combined.svg')

#############################################################################################

def calc_state_strs(state_vecs):
    state_strs = []
    for state_vec in state_vecs:
        state_str = pack_vec(state_vec)
        state_strs.append(state_str)
    return state_strs

def calc_qs_all(state_vecs):
    qs_all = []
    for state_vec in state_vecs:
        qs = state_vec - 1
        qs_all.append(qs)
    return qs_all

# def get_state_vecs_subset(rel_idx, filter, state_vecs):
# def get_filtered_state_vecs(state_vecs, filter_close):
#     state_vecs_filtered = []
#     for state_vec in state_vecs:
#         accept = True # accept as state relating to at_idx
#         for rel_idx, v in enumerate(state_vec):
#             if not filter_close[rel_idx]:
#                 if v != 1:
#                     accept = False
#         if accept:
#             # print(f'Accepted for filter {filter_close}: {state_vec}')
#             state_vecs_filtered.append(state_vec)
#         # else:
#             # print(f'Not accepted for filter {filter_close}: {state_vec}')
#     return state_vecs_filtered

# def curate_state_vecs(state_vecs,indices,mol0,cutoff_distance=7):
#     dmap = Chem.rdmolops.GetDistanceMatrix(mol0)
#     # print(indices)
#     # print(dmap)
#     state_vecs_fragmented = []
#     for rel_idx, at_idx in enumerate(indices):
#         # print(rel_idx, at_idx)
#         distances = dmap[at_idx,indices]
#         # print(distances)
#         filter_close = np.where(distances <= cutoff_distance, True, False)
#         # print(filter_close)
#         state_vecs_filtered = get_filtered_state_vecs(state_vecs, filter_close)
#         state_vecs_fragmented.append(state_vecs_filtered) # list of list of svecs to considered around at_idx
#     return state_vecs_fragmented

def calc_state_freqs(tmatrix):
    w, v = np.linalg.eig(tmatrix.T)
    idx = np.argmin(np.abs(w - 1))
    pi = np.real(v[:, idx])
    pi = pi / pi.sum()
    return pi

def calc_state_freqs_sparse(tmatrix):

    P = csr_matrix(tmatrix)

    pi = np.ones(P.shape[0]) / P.shape[0]
    for idx in range(1000):
        pi = pi @ P
    return pi

def calc_state_freqs_power_iter(tmatrix):
    pi = np.ones(tmatrix.shape[0]) / tmatrix.shape[0]

    for _ in range(10_000):
        pi = pi @ tmatrix

    pi /= pi.sum()
    return pi

def construct_state_vectors_single(indices, q_options, verbose=False):
    # tfs = np.array(list(itertools.product([0,1,2], repeat=len(indices))))
    # state_vecs = []
    state_vecs = []
    state_vec = np.ones((len(indices)),dtype=int) #[1 for _ in indices]
    state_vecs.append(state_vec)

    for rel_idx, at_idx in enumerate(indices):
        for q in [0,2]:
            if q_options[q][rel_idx] == 1:
                state_vec = np.ones((len(indices)),dtype=int) #[1 for _ in indices]
                state_vec[rel_idx] = q
                state_vecs.append(state_vec)
    # print(state_vecs)
    return state_vecs

def compare_pkas(indices, q_options, state_str0, state_str1, base_lib, acid_lib):
    base_pka_diff = np.zeros((len(indices)))
    acid_pka_diff = np.zeros((len(indices)))
    for rel_idx, at_idx in enumerate(indices):
        if (q_options[2][rel_idx] == 1): # allowed for base
            if (at_idx in base_lib[state_str0]) and at_idx in (base_lib[state_str1]):
                base_pka_diff[rel_idx] = abs(base_lib[state_str1][at_idx] - base_lib[state_str0][at_idx])
            else:
                base_pka_diff[rel_idx] = 10. # one disappeared
        if (q_options[0][rel_idx] == 1): # allowed for acid
            if (at_idx in acid_lib[state_str0]) and at_idx in (acid_lib[state_str1]):
                acid_pka_diff[rel_idx] = abs(acid_lib[state_str1][at_idx] - acid_lib[state_str0][at_idx])
            else:
                acid_pka_diff[rel_idx] = 10. # one disappeared
    return base_pka_diff, acid_pka_diff

def construct_coupling_matrix(indices, state_strs, state_vecs, base_pka_diffs, acid_pka_diffs, coupling_cutoff=1.):
    coupling_matrix = np.zeros((len(indices),len(indices)))
    # print(indices)
    for state_str, state_vec in zip(state_strs[1:], state_vecs[1:]):
        # print(state_str)
        changed_rel_idx = np.where(state_vec != 1)[0][0]
        # print(changed_rel_idx)
        # print(base_pka_diffs[state_str])
        # print(acid_pka_diffs[state_str])
        coupling_matrix[changed_rel_idx] += np.where(base_pka_diffs[state_str] > coupling_cutoff, 1, 0)
        coupling_matrix[changed_rel_idx] += np.where(acid_pka_diffs[state_str] > coupling_cutoff, 1, 0)
    return coupling_matrix

def cluster_coupling_matrix(M):

    n = M.shape[0]
    visited = set()
    clusters = []

    def dfs(i, cluster):
        for j in range(n):
            if j not in visited and (M[i, j] != 0 or M[j, i] != 0):
                visited.add(j)
                cluster.add(j)
                dfs(j, cluster)

    for i in range(n):
        if i not in visited:
            visited.add(i)
            cluster = {i}
            dfs(i, cluster)
            clusters.append(cluster)
    clusters = [list(c) for c in clusters]
    return clusters

def coupling_assay(indices,q_options, mol0, mols_lib, smiles_lib, model_base, model_acid, base_lib, acid_lib, coupling_cutoff=1.0, device='cpu'):
    state_vecs = construct_state_vectors_single(indices, q_options)
    # print(state_vecs)
    state_strs = calc_state_strs(state_vecs)
    # print(state_strs)
    N_states = len(state_vecs)
    # print(f'N coupling test states: {N_states}')
    qs_all = calc_qs_all(state_vecs)
    mols_lib, smiles_lib = construct_mols(mol0, state_strs, state_vecs, indices, mols_lib, smiles_lib) # pH independent
    base_lib, acid_lib = run_acid_base_calc(state_strs,mols_lib,model_base,model_acid,base_lib,acid_lib,device=device,verbose=False) # pH independent
    # ps_all = calc_state_pkas(state_strs, state_vecs, base_lib, acid_lib, indices, pH=pH,
                                                        # verbose=False)
    # print(ps_all)

    state_str0 = state_strs[0]
    base_pka_diffs = {}
    acid_pka_diffs = {}
    for state_str1 in state_strs[1:]:
        base_pka_diffs[state_str1], acid_pka_diffs[state_str1] = compare_pkas(indices, q_options, state_str0, state_str1, base_lib, acid_lib)
        # print(base_pka_diffs)
        # print(acid_pka_diffs)
    coupling_matrix = construct_coupling_matrix(indices, state_strs, state_vecs, base_pka_diffs, acid_pka_diffs, coupling_cutoff=coupling_cutoff)
    clusters = cluster_coupling_matrix(coupling_matrix)
    return clusters, mols_lib

def run_pipeline(name,smiles_raw,pH_output=7,cutoff_states=1000,coupling_cutoff=0.,device='cpu',
                 pH_band=6.):
    print(name)
    print(smiles_raw, flush=True)

    # molgpka ML models
    path = '/home/sobuelow/miniconda3/envs/prop_profiler/lib/python3.14/site-packages/prop_profiler/model_weights'

    model_file_base = f'{path}/weight_base.pth'
    model_file_acid = f'{path}/weight_acid.pth'
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

    pHs = np.arange(0,14.1,0.5)#base_cutoff,acid_cutoff+0.0001,0.5)
    # pHs = np.arange(7,7.1,0.5)#base_cutoff,acid_cutoff+0.0001,0.5)
    # pHs = np.arange(12,12.6,0.5)#base_cutoff,acid_cutoff+0.0001,0.5)

    mol0, q0s, exclude_indices = preprocess(smiles_raw)
    mol0_h = Chem.rdmolops.AddHs(mol0)

    base0, acid0 = predict_acid_base(mol0_h,model_base,model_acid,device=device,verbose=False)

    # indices, q_options = find_candidate_sites(base0, acid0, exclude_indices,verbose=True)
    # print(q_options)
    # state_vecs = construct_state_vectors(indices, q_options)
    # N_states = len(state_vecs)
    # print(f'N state vectors: {N_states}',flush=True)

    N_states = 0 # placeholder
        
    for pH_idx, pH in enumerate(pHs): #,total=len(pHs)):#,total=len(pHs)):
        # print(f'pH: {pH}')
        indices0, q_options0 = find_candidate_sites(base0, acid0, exclude_indices,pH,pH_band=pH_band,verbose=False)
        # print(q_options0)
        accept_clusters = False
        coupling_cutoff = 0.0
        while not accept_clusters:
            # print(f'coupling cutoff: {coupling_cutoff}')
            accept_clusters = True
            clusters, mols_lib = coupling_assay(indices0, q_options0, mol0, mols_lib, smiles_lib, model_base, model_acid, base_lib, acid_lib, coupling_cutoff=coupling_cutoff, device=device)
            # print(clusters)
            for c_idx, cluster in enumerate(clusters):
                indices = [indices0[c] for c in cluster] # indices0[cluster]
                q_options = q_options0[:,cluster]
                state_vecs = construct_state_vectors(indices, q_options)
                N_states = len(state_vecs)
                # print(f'N_states: {N_states}')
                if N_states > cutoff_states:
                    accept_clusters = False
                    coupling_cutoff += 0.5
        if coupling_cutoff > 1.5:
            print(f'Coupling cutoff high: {coupling_cutoff}')
        # print(f'Accepted clusters {clusters}')
        state_freqs_clusters = []
        # net_charges_clusters = []
        state_strs_clusters = []
        indices_clusters = []
        
        for c_idx, cluster in enumerate(clusters):
            # print(f'cluster {cluster}')

            smiles_frag_lib = {}

            indices = [indices0[c] for c in cluster] # indices0[cluster]
            # print(indices)
            indices_str = ''
            for id in indices:
                indices_str += f'{id},'
            indices_str = indices_str[:-1]
            # indices_str = ".".join(f'{indices}')
            # print(f'Indices_str: {indices_str}')

            if indices_str not in mols_frag_lib:
                mols_frag_lib[indices_str] = {}
            if indices_str not in base_frag_lib:
                base_frag_lib[indices_str] = {}
            if indices_str not in acid_frag_lib:
                acid_frag_lib[indices_str] = {}
            
            q_options = q_options0[:,cluster]
            # print(indices)
            # print(q_options)
            state_vecs = construct_state_vectors(indices, q_options)
            N_states = len(state_vecs)

            # print(f'N state vectors for pH {pH}: {N_states}',flush=True)
            # if N_states > 5500:
                # raise
            state_strs = calc_state_strs(state_vecs)
            # qs_all = calc_qs_all(state_vecs)

            mols_frag_lib[indices_str], smiles_frag_lib = construct_mols(mol0, state_strs, state_vecs, indices, mols_frag_lib[indices_str], smiles_frag_lib) # pH independent
            base_frag_lib[indices_str], acid_frag_lib[indices_str] = run_acid_base_calc(state_strs,mols_frag_lib[indices_str],model_base,model_acid,
                                                                                        base_frag_lib[indices_str],acid_frag_lib[indices_str],device=device) # pH independent

            ps_all = calc_state_pkas(state_strs, state_vecs, base_frag_lib[indices_str], acid_frag_lib[indices_str], indices, pH=pH,
                                                        verbose=False)
            N_states = len(state_vecs)

            # print('Calculating tmatrix')
            tmatrix = calc_tmatrix(state_vecs,state_strs,ps_all,N_states)

            # print(f'Total entries in matrix: {N_states**2}')
            # print(f'N zeros in tmatrix: {np.sum(np.where(tmatrix==0.,1,0))}')

            state_freqs = calc_state_freqs_sparse(tmatrix)
            
            state_strs_clusters.append(state_strs)
            # net_charges_clusters.append(net_charges)
            state_freqs_clusters.append(state_freqs)
            indices_clusters.append(indices)

        # combine clusters
        # if len(clusters) > 2:
            # raise

        cluster_state_idxs = [list(range(len(state_strs))) for state_strs in state_strs_clusters]
        # print(cluster_state_idxs)

        combinations = list(itertools.product(*cluster_state_idxs))
        # print(combinations)

        state_strs = []
        indices = []
        for indices_cluster in indices_clusters:
            indices.extend(indices_cluster) # NEEDS SORTING SOMEHOW

        # print(indices)
        ps = np.argsort(indices)
        indices = [indices[p] for p in ps]
        # indices = indices[ps]
        # print(indices)

        # indices = indices_clusters[0] + indices_clusters[1]
        # print(indices)

        # for s_idx0, s_idx1 in combinations:
        
        for s_idxs in combinations:
            state_str = ''
            state_freq = 1.
            for c_idx, s_idx in enumerate(s_idxs):
                # print(c_idx, s_idx)
                state_str += state_strs_clusters[c_idx][s_idx]
                state_freq *= state_freqs_clusters[c_idx][s_idx]                   

            # print(state_str)
            state_str = list(state_str)
            # print(state_str)
            state_str = [state_str[p] for p in ps]
            # print(state_str)
            state_str = "".join(state_str)
            # state_str0 = state_strs_clusters[0][s_idx0]
            # state_str1 = state_strs_clusters[1][s_idx1]
            # state_str = state_str0 + state_str1
            state_strs.append(state_str)
            # state_freq = state_freqs_clusters[0][s_idx0] * state_freqs_clusters[1][s_idx1]
            if state_str not in state_freqs_all:
                state_freqs_all[state_str] = np.zeros(len(pHs))
            state_freqs_all[state_str][pH_idx] = state_freq

        # print(state_strs)
        state_vecs = [unpack_vec(state_str) for state_str in state_strs]
        mols_lib, smiles_lib = construct_mols(mol0, state_strs, state_vecs, indices, mols_lib, smiles_lib) # pH independent
        state_freq_max = 0.
        net_charge = 0.
        for state_str, state_freq in state_freqs_all.items():
            # print(state_freq)
            if state_freq[pH_idx] > state_freq_max:
                state_str_opti = state_str
            state_vec = unpack_vec(state_str)
            net_charge = Chem.GetFormalCharge(mols_lib[state_str]) * state_freq[pH_idx]

        net_charges.append(net_charge)

        if pH == pH_output:
            mol = mols_lib[state_str_opti]
            export_sdf(name,mol)
            export_smi(name,smiles_lib[state_str_opti])
            plot_optimal_state(name,mol)
            print(f'Optimal smiles for pH {pH}: {smiles_lib[state_str_opti]}')

    net_charges = np.array(net_charges)

    with open(f'output/{name}_net_charges.txt','w') as f:
        for pH, net_charge in zip(pHs, net_charges):
            f.write(f'{pH:.2f} {net_charge:.3f}\n')

    cutoff = 0.0
    tries = 0
    if len(state_freqs_all.keys()) > 1:
        state_strs_relevant = []
        sfreqs_relevant = []
        mols_relevant = []
        # state_strs_relevant, sfreqs_relevant, mols_relevant = get_relevant_states(state_freqs_all, state_strs, mols, cutoff=cutoff)
        for state_str, sfreqs in state_freqs_all.items():
            if np.max(sfreqs) > cutoff:
                state_strs_relevant.append(state_str)
                sfreqs_relevant.append(sfreqs)
                # mols_lib[state_str].SetProp("_Name",state_str)
                mols_relevant.append(mols_lib[state_str])
        N_relevant_states = len(state_strs_relevant)
        print(f'Initial N relevant states: {N_relevant_states} with cutoff {cutoff}')
        while N_relevant_states > 18:
            state_strs_relevant = []
            sfreqs_relevant = []
            mols_relevant = []
            tries += 1
            cutoff += 0.02
            # state_strs_relevant, sfreqs_relevant, mols_relevant = get_relevant_states(state_freqs_all, state_strs, mols, cutoff=cutoff)
            for state_str, sfreqs in state_freqs_all.items():
                if np.max(sfreqs) > cutoff:
                    state_strs_relevant.append(state_str)
                    sfreqs_relevant.append(sfreqs)
                    # mols_lib[state_str].SetProp("_Name",state_str)
                    mols_relevant.append(mols_lib[state_str])
            N_relevant_states = len(state_strs_relevant)
            # print(f'Current N relevant states: {N_relevant_states}')
        print(f'Final N relevant states: {N_relevant_states} with cutoff {cutoff}')
        plot_pH_scan(name, indices, state_strs_relevant, sfreqs_relevant, pHs, net_charges, cmap=plt.cm.Spectral,
                fsave=f'figures/{name}_ph_scan.svg')
        
        plot_relevant_states(name, mols_relevant)
        compose_image(name,N_relevant_states)

    # cutoff = 0.0
    # tries = 0
    # if len(state_strs) > 1:
    #     state_strs_relevant, sfreqs_relevant, mols_relevant = get_relevant_states(state_strs, state_freqs_all, mols_lib, cutoff=cutoff)
    #     N_relevant_states = len(state_strs_relevant)
    #     print(f'Initial N relevant states: {N_relevant_states} with cutoff {cutoff}')
    #     while N_relevant_states > 18:
    #         tries += 1
    #         # print(f'Tries: {tries} | Cutoff: {cutoff}')
    #         cutoff += 0.05
    #         state_strs_relevant, sfreqs_relevant, mols_relevant = get_relevant_states(state_strs, state_freqs_all, mols_lib, cutoff=cutoff)
    #         N_relevant_states = len(state_strs_relevant)
    #         # print(f'Current N relevant states: {N_relevant_states}')
    #     print(f'Final N relevant states: {N_relevant_states} with cutoff {cutoff}')
    #     plot_pH_scan(name, indices, state_strs_relevant, sfreqs_relevant, pHs, net_charges, cmap=plt.cm.Spectral,
    #             fsave=f'figures/{name}_ph_scan.svg')
        
    #     plot_relevant_states(name, mols_relevant)
    #     compose_image(name,N_relevant_states)