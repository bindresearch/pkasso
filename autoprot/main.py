from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

import numpy as np
import os

import copy
import yaml
import matplotlib.pyplot as plt

from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io

import cairosvg

class AutoProt:
    def __init__(self,name,f_input='smiles.smi',path_inp_qupkake='qupkake_input',path_out_qupkake='qupkake_output',
                 path_out_autoprot = 'path_out_autoprot',
                 pH = 7.,
                 cutoff=0.05,ncycles=10,
                 verbose=False,
                 mp = 2):
        self.name = name
        self.f_input = f_input
        self.path_inp_qupkake = path_inp_qupkake
        self.path_out_qupkake = path_out_qupkake
        self.path_out_autoprot = path_out_autoprot
        self.state_vecs = []
        self.state_strs = []

        self.all_ps = {}
        self.all_smiles = {}
        self.flagged_state_strs = []
        self.pH = pH
        self.cutoff = cutoff # frequency cutoff to consider for protonation/deprotonation
        self.ncycles = ncycles
        self.verbose = verbose
        self.mp = mp

        os.makedirs(path_out_autoprot,exist_ok=True)
        os.makedirs(path_out_qupkake,exist_ok=True)

        self.logfile = f'{name}.log'

        if os.path.exists(f'{self.path_out_autoprot}/{self.logfile}'):
            os.remove(f'{self.path_out_autoprot}/{self.logfile}')

    ########################################################

    def log(self,string):
        with open(f'{self.path_out_autoprot}/{self.logfile}','a') as f:
            f.write(f'{string}\n')

    def read_input(self):
        """ Find input smiles string from file """

        with open(self.f_input,'r') as f:
            for line in f.readlines():
                spl = line.split()
                if spl[1] == self.name:
                    self.smiles_raw = spl[0]
                    break

        # self.log(f'Raw input smiles: {self.smiles_raw}')
        self.log(f'Raw input smiles: {self.smiles_raw}')
        self.mol0_raw = Chem.MolFromSmiles(self.smiles_raw)
        
        symbols = [at.GetSymbol() for at in self.mol0_raw.GetAtoms()]
        symbols = "".join(symbols)
        self.log(f'Atom order raw: {symbols}')

        self.N = self.mol0_raw.GetNumAtoms()

        state_vec = np.ones((self.N),dtype=int) # 0=-, 1=neu, 2=+
        state_str = self.pack_vec(state_vec)

        self.mol0 = Chem.MolFromSmiles(self.smiles_raw) # only for first qupkake run

        # Fix ordering of atoms by running qupkake once
        fname = f'{self.name}_{state_str}'
        self.write_smiles(self.mol0,self.path_inp_qupkake,fname)
        self.clear_qupkake_output(fname)
        self.prep_qupkake_input(state_str,fname)
        ret = self.run_qupkake(fname)
        if os.path.isfile(f'{self.path_out_qupkake}/{fname}.sdf'):
            ms = self.load_molecules(f'{self.path_out_qupkake}/{fname}.sdf')
            self.mol0 = ms[0]
            return_code = 0
        else: # Write sdf from input smi. Prevents error exit.
            print(f'Writing sdf from input smi to {self.path_out_qupkake}/{fname}.')
            self.write_molecule(self.mol0,self.path_out_qupkake,fname)
            return_code = 1
        self.smiles_input = Chem.MolToSmiles(self.mol0,canonical=False)

        self.state_vecs.append(state_vec)
        self.state_strs.append(state_str)

        if self.verbose:
            self.log(self.state_strs)
        return return_code

    @staticmethod
    def load_molecules(fname,removeHs=True):
        with Chem.SDMolSupplier(fname, removeHs=removeHs) as suppl:
            ms = [x for x in suppl if x is not None]
        # self.log(f'number of pka predictions: {len(ms)}')
        return ms

    @staticmethod
    def pack_vec(state_vec):
        state_str = "".join([str(x) for x in state_vec])
        return state_str

    @staticmethod
    def unpack_vec(state_str):
        state_vec = np.array([int(s) for s in state_str],dtype=int)
        return state_vec
        
    #########################################################

    def prep_and_run(self,new_states):

    # def prep_and_run(self,discard_cutoff=0.3):
        # Prep, run, and analyse states
        # last_states_curated = []

        ##### FIX THIS ######

        for st_idx, state_str in enumerate(new_states):
            self.log('-'*30)
            self.log(f'New state {st_idx+1}/{len(new_states)}')
            # Prep and run qupkake
            fname = f'{self.name}_{state_str}'
            if self.verbose:
                self.log(f'Preparing qupkake input for {fname}.')
            
            self.prep_qupkake_input(state_str,fname)
            ret = self.run_qupkake(fname)

            if ret != 0: # qupkake run failed
                self.log(f'Qupkake run failed.')
                self.log(f'Flagging state {state_str} for deletion.')       
                self.flagged_state_strs.append(state_str)
                ps = np.zeros((2,self.N)) - 1 # blank state
                # continue
            # analyze qupkake runs
            else:
                if self.verbose:
                    self.log(f'Analysing results for {fname}.')
                ms = self.load_molecules(f'{self.path_out_qupkake}/{fname}.sdf')

                # Addresses re-ordering of atoms by qupkake
                # if state_str == '1'*self.N:
                #     self.mol0 = ms[0]
                #     symbols = [at.GetSymbol() for at in self.mol0.GetAtoms()]
                #     symbols = "".join(symbols)
                #     self.smiles_input = Chem.MolToSmiles(self.mol0,canonical=False)
                #     if self.verbose:
                #         self.log('Updating mol0 order and input smiles.')
                #         self.log(f'Atom order processed: {symbols}')

                ps = self.calc_ps(ms)

                smiles = Chem.MolToSmiles(ms[0],canonical=False)

                # if self.verbose:
                self.log(f'Qupkake output smiles (tautomer search): {smiles}')

            self.all_ps[state_str] = list(ps)
            # last_states_curated.append(state_str)
        # if self.verbose:
            # self.log(f'Old last states: {self.state_strs[-1]}')
            # self.log(f'Curated last states: {last_states_curated}')
        # self.state_strs[-1] = last_states_curated

    def select_new_states(self):
        """ Only spawn new test states from states with sufficient frequency > cutoff """
        
        origin_states = []
        for state_str, state_freq in zip(self.state_strs, self.state_freqs):
            if state_freq > self.cutoff:
                origin_states.append(state_str)
            else:
                self.log(f'Discarding {state_str} as origin; too low population.')

        self.log(f'Origin states:')
        self.log(origin_states)
        new_states = self.add_states(origin_states)
        
        for state_str in new_states:
            self.state_strs.append(state_str)
            self.state_vecs.append(self.unpack_vec(state_str))

        # new_states = self.add_states(self.state_strs[-1])
        self.log('Selecting new states.')
        self.log(f'{len(new_states)} new states: {new_states}')
        return new_states

        # new_states_curated = []

        # for state_str in new_states:
        #     found = self.check_found(state_str)
        #     # discarded = self.check_discarded(state_str)
        #     if (not found):# and (not discarded):
        #         new_states_curated.append(state_str)
        # if self.verbose:
        #     self.log(f'New states curated: {new_states_curated}')
        
    
        # if len(new_states) == 0:
        #     if self.verbose:
        #         self.log('No new states found. Exiting.')
        #     return -1
        # else:
        #     self.state_strs.append(new_states)
        # return 0

    def run_predictions(self):
        self.log(f'='*50)
        self.log(self.name)
        ret = self.read_input()
        new_states = [self.state_strs[0]] # starting 111111...
        for cycle in range(self.ncycles):
            self.log('-'*50)
            self.log(f'CYCLE {cycle}')
            self.prep_and_run(new_states)

            self.run_state_analysis()
            # self.log('states freq:')
            # for st_str, st_freq in zip(self.state_strs, self.state_freqs):
            #     self.log(f'{st_str} {st_freq:.2f}')
            # self.log(self.state_freqs)
            if cycle < self.ncycles - 1: # not last cycle
                new_states = self.select_new_states()
            if len(new_states) == 0:
                break
        self.dump_results()

    #########################################################

    def protonate_molecules(self,mol,state_vec,addHs=False):
        mol_prot = copy.deepcopy(mol)#.copy()
        if self.verbose:
            self.log(self.pack_vec(state_vec))
            self.log(f'Before (de)protonation: {Chem.MolToSmiles(mol_prot,canonical=False)}')
        for idx, s in enumerate(state_vec):
            if s > 1:
                mol_prot = self.set_protonation(mol_prot,idx,1)
            elif s < 1:
                mol_prot = self.set_protonation(mol_prot,idx,-1)
        if addHs:
            mol_prot = Chem.AddHs(mol_prot,addCoords=True)
        if self.verbose:
            self.log(f'After (de)protonation: {Chem.MolToSmiles(mol_prot,canonical=False)}')
        return mol_prot

    @staticmethod
    def set_protonation(molecule,at_idx,dq):
        atom = molecule.GetAtomWithIdx(at_idx)
        q0 = atom.GetFormalCharge()
        atom.SetFormalCharge(q0+dq)
        return molecule

    @staticmethod
    def write_molecule(mol,path_inp,fname):
        os.makedirs(path_inp, exist_ok=True)
        mol.SetProp('_Name', fname)
        with Chem.SDWriter(f'{path_inp}/{fname}.sdf') as f:
            f.write(mol)

    def clear_qupkake_output(self,fname):
        if os.path.isfile(f'{self.path_out_qupkake}/{fname}.sdf'):
            os.system(f'rm {self.path_out_qupkake}/{fname}.sdf')

    @staticmethod
    def write_smiles(mol,path,fname):
        os.makedirs(path, exist_ok=True)
        mol.SetProp('_Name', fname)
        smiles = Chem.MolToSmiles(mol,canonical=False)
        with open(f'{path}/{fname}.smi','w') as f:
            f.write(f'{smiles} {fname}')

    def prep_qupkake_input(self,state_str,fname,output='smiles'):
        """ Protonate and save molecules"""

        state_vec = self.unpack_vec(state_str)

        mol_new = self.protonate_molecules(self.mol0,state_vec)
        
        mol_new.SetProp('_Name', fname)
        smiles = Chem.MolToSmiles(mol_new,canonical=False)
        self.all_smiles[state_str] = smiles

        if self.verbose:
            self.log(f'q:{Chem.GetFormalCharge(mol_new)}, n_atoms:{mol_new.GetNumAtoms()}')
        if output == 'sdf':
            self.write_molecule(mol_new,self.path_inp_qupkake,fname)
        elif output == 'smiles':
            self.write_smiles(mol_new,self.path_inp_qupkake,fname)
        # self.clear_qupkake_output(fname)

    def run_qupkake(self,fname,input='smiles'):
        if input == 'sdf':
            pass
            # os.system(f'qupkake file -r data -o {fname}.sdf {self.path_inp_qupkake}/{fname}.sdf')
        elif input == 'smiles':
            with open(f'{self.path_inp_qupkake}/{fname}.smi') as f:
                line = f.readline()
                spl = line.split()
                smiles, fname = spl[0], spl[1]
            self.log(f'{fname} {smiles}')
            # if os.path.isfile(f'{self.path_out_qupkake}/{fname}.sdf'):
            #     os.remove(f'{self.path_out_qupkake}/{fname}.sdf')
            os.system(f'qupkake smiles -r data/{self.name} -o {fname}.sdf -mp {self.mp} "{smiles}"') # -t
        if not os.path.isfile(f'data/{self.name}/output/{fname}.sdf'):
            return -1
        else:
            os.system(f'cp data/{self.name}/output/{fname}.sdf {self.path_out_qupkake}/')
            os.system(f'rm -r data/{self.name}')
            return 0

    #########################################################

    def calc_ps(self,ms):
        # N = ms[0].GetNumAtoms()
        ps = np.zeros((2,self.N)) - 1 # (up/down, at_idx) (up=0,down=1)

        for mol in ms:
            props = mol.GetPropsAsDict()

            at_idx, pka_type, pka = self.get_at_props(mol)
            p_up = self.calc_charge(pka,pH=self.pH) # probability for higher + state
            p_down = 1 - p_up # probability for lower + state
            self.log(f'{at_idx} {pka_type} {pka} up:{p_up:.2f} down:{p_down:.2f}')
            if pka_type == 'basic':
                # self.log('basic',q_up)
                ps[0,at_idx] = p_up
                # self.log(qs)
            elif pka_type == 'acidic':
                ps[1,at_idx] = p_down
        ps_fixed = self.fix_ps(ps,self.cutoff)
        if self.verbose:
            self.log(ps)
            self.log('ps_fixed:')
            self.log(ps_fixed)
        return ps_fixed
    
    def fix_ps(self,ps,cutoff): # Hopefully fixes weird qupkake behaviour
        self.log(f'cutoff: {cutoff}')
        ps_T = ps.T
        ps_T_fixed = copy.deepcopy(ps_T)
        for at_idx, (p_up, p_down) in enumerate(ps_T):
            self.log(f'{at_idx}, {p_up}, {p_down}')
            if (p_up > cutoff) and (p_down > -1):
                self.log('fixing p_down')
                ps_T_fixed[at_idx,1] = -1 # reset p_down to -1
        ps_fixed = ps_T_fixed.T
        return ps_fixed

    @staticmethod
    def get_at_props(mol):
        at_idx = mol.GetPropsAsDict()['idx']
        pka = mol.GetPropsAsDict()['pka']

        pka_type = mol.GetPropsAsDict()['pka_type']
        if isinstance(pka,str):
            if pka[:7] == 'tensor(':
                pka = float(pka[7:-1])
            else:
                raise
        return at_idx, pka_type, pka
    
    @staticmethod
    def calc_charge(pka,pH=7.):
        ppos = 1. / ( 1 + 10**(pH-pka) ) # fraction of more positively charged res
        dq = 0
        return ppos

    #########################################################

    def add_states(self,origin_states):
        new_states = []
        # if self.verbose:
            # self.log(f'Last states: {last_states}')
        for state_str in origin_states:
            ps = self.all_ps[state_str]
            state_vec = self.unpack_vec(state_str)
            for idx, p_up in enumerate(ps[0]): # up because basic
                if (p_up > self.cutoff) and (state_vec[idx] < 2):
                    state_new_str = self.add_new_state(state_vec,idx,1)
                    
                    if state_new_str in new_states:
                        if self.verbose:
                            self.log(f'{state_new_str} already in new_states {new_states}.')
                    elif state_new_str in self.state_strs:
                        self.log(f'{state_new_str} values already predicted.')
                    elif state_new_str in self.flagged_state_strs:
                        self.log(f'{state_new_str} values already discarded.')
                    else:
                        new_states.append(state_new_str)
            for idx, p_down in enumerate(ps[1]): # down because acidic
                if (p_down > self.cutoff) and (state_vec[idx] > 0):
                    state_new_str = self.add_new_state(state_vec,idx,-1)
                    if state_new_str in new_states:
                        if self.verbose:
                            self.log(f'{state_new_str} already in new_states {new_states}.')
                    elif state_new_str in self.state_strs:
                        self.log(f'{state_new_str} values already predicted.')
                    elif state_new_str in self.flagged_state_strs:
                        self.log(f'{state_new_str} values already discarded.')
                    else:
                        new_states.append(state_new_str)
        return new_states

    def add_new_state(self,state_vec,idx,delta):
        state_new = state_vec.copy()
        state_new[idx] += delta
        state_new_str = self.pack_vec(state_new)
        return state_new_str
    
    def check_found(self,state_str):
        found = False
        # for states in self.state_strs:
        if state_str in self.state_strs:
            found = True
            if self.verbose:
                self.log(f'State {state_str} already found.')
        return found
    
    def check_discarded(self,state_str):
        # self.log(f'Checking {state_str} for discarded.')
        discarded = False
        if state_str in self.flagged_state_strs:
            discarded = True
            if self.verbose:
                self.log(f'State {state_str} already discarded.')
        return discarded

    def dump_results(self,path='autoprot_cache'):
        os.makedirs(path,exist_ok=True)
        with open(f'{path}/{self.name}_states.yaml','w') as stream:
            yaml.dump(self.state_strs,stream)
        np.savez(f'{path}/{self.name}_ps.npz', **self.all_ps)

    def load_results(self,path='autoprot_cache'):
        all_ps = np.load(f'{path}/{self.name}_ps.npz')
        return all_ps

    ####################################################################
    ####################################################################
    ####################################################################

    def run_state_analysis(self):
        self.prep_state_selection()
        self.calc_tmatrix()
        self.traj_states, self.traj = self.simulate_state_traj(self.state_vecs_short,self.tmatrix)
        self.calc_optimal_state()
        # self.plot_state_freqs()
        self.export_optimal_state()

    def export_results(self):
        self.plot_state_freqs()
        self.export_optimal_state()
        self.draw_all_states()
        self.draw_labeled_state()

    def prep_state_selection(self,load_results=False):

        if load_results:
            self.all_ps = self.load_results()

        # self.state_strs_list = np.array([self.unpack_vec(state) for state in list(self.all_ps.keys())])
        # self.state_strs_list_str = list(self.all_ps.keys())
        # self.log(f'self.state_strs_list_str: {self.state_strs_list_str}')
        self.N_states = len(self.state_strs)
        self.relevant_indices = np.unique(np.where(np.array(self.state_vecs) != 1)[1]) # find indices that change between states
        
        if self.verbose:
            self.log(f'N_states: {self.N_states}')
            self.log(f'relevant indices: {self.relevant_indices}')
        
        self.log(f'{np.array(self.state_vecs).shape}')
        self.state_vecs_short = np.array(self.state_vecs)[:,self.relevant_indices] # (state,index)
        self.state_strs_short = [self.pack_vec(state) for state in self.state_vecs_short]

        # ps list of relevant indices
        self.ps_relevant = np.array([self.all_ps[st] for st in self.state_strs]) 
        self.ps_relevant = self.ps_relevant[:,:,self.relevant_indices] # states, up/down, indices
        if self.verbose:
            self.log(f'{self.ps_relevant}')

    def get_s_idx(self,state,states_str):
        return states_str.index(self.pack_vec(state))

    def calc_tmatrix(self):
        """ Transition matrix between molecule protonation states"""

        tmatrix_raw = [[[] for _ in range(self.N_states)] for _ in range(self.N_states)] # N_states x N_states (x duplicate predictions)

        for s_idx, state_vec_short in enumerate(self.state_vecs_short):
            if self.verbose:
                self.log('------')
                self.log(f'{s_idx} {state_vec_short}')
            ps_up = self.ps_relevant[s_idx,0]
            ps_down = self.ps_relevant[s_idx,1]

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
                        state_target_vec = state_vec_short.copy()
                        state_target_vec[at_idx] += dq
                        state_target_str = self.pack_vec(state_target_vec)
                        if self.verbose:
                            self.log(f'target: {state_target_vec}')
                            self.log(f'target str: {state_target_str}')
                        if state_target_str in self.state_strs_short:
                            c_target_idx = self.state_strs_short.index(state_target_str)
                            # c_target_idx = self.get_s_idx(state_target,self.state_strs_short)
                            tmatrix_raw[c_target_idx][s_idx].append(p) # to, from
                            tmatrix_raw[s_idx][c_target_idx].append(1-p)

        tmatrix_mean = np.zeros((self.N_states,self.N_states)) # N_states x N_states, average over predictions per ij

        for idx, row in enumerate(tmatrix_raw):
            for jdx, col in enumerate(row):
                if len(col) > 0:
                    tmatrix_mean[idx,jdx] = np.mean(col)

        self.tmatrix = tmatrix_mean.copy()

        for idx, col in enumerate(tmatrix_mean.T):
            self.tmatrix[idx,idx] = np.prod(-1.*col+1) # probability not to transition

        # Normalized tmatrix (probabilities from one state to all other states sums to 1)
        self.tmatrix /= np.sum(self.tmatrix,axis=0) 

        if self.verbose:
            self.log(self.state_strs_short)
            self.log(self.tmatrix)

    #########################################################

    @staticmethod
    def simulate_state_traj(state_vecs,tmatrix,nsteps=1000000):
        """ Simulate evolution according to transition matrix """

        choice0 = np.random.randint(len(state_vecs))

        traj = [choice0]
        traj_states = [state_vecs[choice0]]

        for t in range(1,nsteps):
            trans = tmatrix.T[traj[t-1]]
            choice = np.random.choice(np.arange(len(state_vecs)),p=trans)
            # self.log(traj[t-1], trans, choice)
            traj.append(choice)
            traj_states.append(state_vecs[choice])

        traj_states = np.array(traj_states)
        traj = np.array(traj)

        return traj_states, traj

    #########################################################

    def calc_optimal_state(self):
        self.state_vecs_mean = np.mean(self.traj_states,axis=0)

        self.state_freqs = np.zeros((self.N_states))

        for idx in range(self.N_states):
            self.state_freqs[idx] = len(np.where(self.traj == idx)[0])
        self.state_freqs /= np.sum(self.state_freqs)
        # self.state_freqs *= 100
        # self.log(self.state_freqs)

        self.log('State | State (relevant) | Occupancy')
        for state_str, state_str_short, freq in zip(self.state_strs, self.state_strs_short, self.state_freqs):
            self.log(f'{state_str} | {state_str_short} | {freq*100:.2f}%')

        idx_max = np.argmax(self.state_freqs)

        self.state_vec_opti = self.state_vecs[idx_max]
        self.state_str_opti = self.state_strs[idx_max]

        # self.state_vec_opti = self.reconstruct_full_state(self.unpack_vec(self.states_str[idx_max]))
        # self.state_str_opti = self.pack_vec(self.state_vec_opti)

        self.log(f'Most likely state: {self.state_str_opti} at {self.state_freqs[idx_max]*100:.2f}%')

    # def reconstruct_full_state(self,state):
    #     """ Reconstruct full state vector from state vector of relevant indices"""
    #     state_vec = np.ones((self.N),dtype=int)
    #     for idx, st in zip(self.relevant_indices, state):
    #         state_vec[idx] = st # np.round(st) # Should be int now
    #     return state_vec

    #########################################################

    def plot_state_freqs(self):
        self.log(f'N_states: {self.N_states}')
        fig, ax = plt.subplots(figsize=(self.N_states/3.,3.5))
        xs = np.arange(self.N_states)
        ax.bar(xs, self.state_freqs*100)
        ax.set_xticks(xs)
        ax.set_xticklabels(self.state_strs_short,rotation=-45)
        ax.set(xlabel='States',ylabel='Occupancy [%]')

        for idx, st in zip(self.relevant_indices, self.state_vecs_mean):
            self.log(f'Atom {idx}: Charge: {st-1:.2f}')
        fig.savefig(f'{self.path_out_autoprot}/{self.name}_state_freqs.pdf')
        return fig

    @staticmethod
    def draw_molecule(mol,at_idx=None,
            rgba_color = (0.0, 1.0, 1.0, 0.4)): # transparent blue):#,f_out=None):
        mol.Compute2DCoords()

        if isinstance(at_idx,int):
            highlightAtoms = [at_idx]
        else:
            highlightAtoms=at_idx

        drawer = rdMolDraw2D.MolDraw2DCairo(350,300)
        drawer.drawOptions().fillHighlights=True
        drawer.drawOptions().setHighlightColour((rgba_color))
        drawer.drawOptions().highlightBondWidthMultiplier=20
        drawer.drawOptions().clearBackground = True
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=highlightAtoms)
        bio = io.BytesIO(drawer.GetDrawingText())
        img = Image.open(bio)
        return img
    
    def draw_state(self,state_str):
        fname = f'{self.name}_{state_str}'

        ms = self.load_molecules(f'{self.path_out_qupkake}/{fname}.sdf')
        mol = ms[0]
        img = self.draw_molecule(mol,at_idx=[int(idx) for idx in self.relevant_indices])
        img.save(f'{self.path_out_autoprot}/{fname}.pdf')
        return img
    
    def draw_all_states(self):
        ms = []
        for state_str in self.state_strs:
            # state_str = self.pack_vec(state_vec)
            sdf = f'{self.path_out_qupkake}/{self.name}_{state_str}.sdf'
            with Chem.SDMolSupplier(sdf) as suppl:
                tmp = [x for x in suppl if x is not None]
            for m in tmp: _=AllChem.Compute2DCoords(m)
            mol = tmp[0]
            ms.append(mol)  
        img=Draw.MolsToGridImage(ms,molsPerRow=5,subImgSize=(200,200),
                                 legends=[f'{self.state_freqs[idx]*100:.1f}%' for idx in range(len(ms))],
                                 returnPNG=False,useSVG=True)
        # print(img)
        with open(f'{self.path_out_autoprot}/{self.name}_all_states.svg','w') as f:
            f.write(img)
        cairosvg.svg2pdf(url=f'{self.path_out_autoprot}/{self.name}_all_states.svg',
                 write_to=f'{self.path_out_autoprot}/{self.name}_all_states.pdf')
        os.remove(f'{self.path_out_autoprot}/{self.name}_all_states.svg')

    def draw_labeled_state(self):
        sdf = f'{self.path_out_autoprot}/{self.name}.sdf'
        with Chem.SDMolSupplier(sdf) as suppl:
            ms = [x for x in suppl if x is not None]
        for m in ms: tmp=AllChem.Compute2DCoords(m)
        mol = ms[0]
        for atom in mol.GetAtoms():
            atom.SetProp('atomLabel', str(atom.GetIdx()))
        # img = Draw.MolToImage(mol)
        img_size = (300,300)
        drawer = rdMolDraw2D.MolDraw2DSVG(img_size[0],img_size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        with open(f'{self.path_out_autoprot}/{self.name}_labeled.svg','w') as f:
            f.write(svg)
        cairosvg.svg2pdf(url=f'{self.path_out_autoprot}/{self.name}_labeled.svg',
            write_to=f'{self.path_out_autoprot}/{self.name}_labeled.pdf')
        os.remove(f'{self.path_out_autoprot}/{self.name}_labeled.svg')

    def export_optimal_state(self,nconfs=200):

        fname = f'{self.name}_{self.state_str_opti}.sdf'

        # Structure (with H); use qupkake structure if possible, otherwise construct from input smiles
        if os.path.isfile(f'{self.path_out_qupkake}/{fname}'):
            ms = self.load_molecules(f'{self.path_out_qupkake}/{fname}',removeHs=False)
            mol = ms[0]
        else:
            self.smiles_optimal = self.all_smiles[self.state_str_opti]
            mol = Chem.MolFromSmiles(self.smiles_optimal)
            mol.SetProp('_Name', fname)
            mol = Chem.AddHs(mol,addCoords=True)
            self.log('Optimizing geometry of optimal state.')
            AllChem.EmbedMultipleConfs(mol,numConfs=nconfs,randomSeed=np.random.randint(1,1000),useRandomCoords=True)
            AllChem.UFFOptimizeMoleculeConfs(mol)
        
        self.write_molecule(mol,self.path_out_autoprot,self.name)
        
        # Smiles (no H); use smiles from qupkake output if possible, otherwise input smiles for prot state
        if os.path.isfile(f'{self.path_out_qupkake}/{fname}'):
            ms_noH = self.load_molecules(f'{self.path_out_qupkake}/{fname}',removeHs=True)
            mol_noH = ms_noH[0]
            self.smiles_optimal = Chem.MolToSmiles(mol_noH,canonical=False)
        else:
            mol_noH = Chem.MolFromSmiles(self.smiles_optimal)

        self.write_smiles(mol_noH,self.path_out_autoprot,self.name)
        self.write_results()

    def write_results(self):
        with open(f'{self.path_out_autoprot}/{self.name}.out','w') as f:
            f.write(f'Name: {self.name}\n')
            f.write(f'pH: {self.pH:.2f}\n')
            f.write(f'Input smiles: {self.smiles_input}\n')
            f.write('States:\n')
            for key in self.all_ps.keys():
                f.write(f'{key}\n')
            f.write('("1" = original protonation state, "0" = deprotonated, "2" = protonated)\n')
            f.write('\n')
                     
            f.write(f'Relevant atoms: {self.relevant_indices}\n')
            f.write(f'Average charge per relevant atoms:\n')
            for idx, st in zip(self.relevant_indices, self.state_vecs_mean):
                f.write(f'Atom {idx}: Charge: {st-1:.2f}\n')
            f.write('\n')

            f.write(f'Protonation state occupancies:\n')
            f.write('State | Occupancy\n')
            for state_str, freq in zip(self.state_strs, self.state_freqs):
                f.write(f'{state_str} | {freq*100:.2f}%\n')
            f.write('\n')

            f.write(f'Smiles of optimal protonation state: {self.smiles_optimal}\n')

#         if output == 'sdf':
#             self.write_molecule(mol_new,self.path_inp,fname)
#         elif output == 'smiles':
#             self.write_smiles(mol_new,self.path_inp,fname)
