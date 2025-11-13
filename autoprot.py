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
                 verbose=False):
        self.name = name
        self.f_input = f_input
        self.path_inp_qupkake = path_inp_qupkake
        self.path_out_qupkake = path_out_qupkake
        self.path_out_autoprot = path_out_autoprot
        self.all_states = []
        self.all_ps = {}
        self.all_smiles = {}
        self.flagged_states = []
        self.pH = pH
        self.cutoff = cutoff # frequency cutoff to consider for protonation/deprotonation
        self.ncycles = ncycles
        self.verbose = verbose
        os.makedirs(path_out_autoprot,exist_ok=True)
        os.makedirs(path_out_qupkake,exist_ok=True)

    ########################################################

    def read_input(self):
        """ Find input smiles string from file """

        with open(self.f_input,'r') as f:
            for line in f.readlines():
                spl = line.split()
                if spl[1] == self.name:
                    self.smiles_raw = spl[0]
                    break

        print(f'Raw input smiles: {self.smiles_raw}')
        self.mol0_raw = Chem.MolFromSmiles(self.smiles_raw)
        
        symbols = [at.GetSymbol() for at in self.mol0_raw.GetAtoms()]
        symbols = "".join(symbols)
        print(f'Atom order raw: {symbols}')

        self.N = self.mol0_raw.GetNumAtoms()

        state_vec = np.ones((self.N),dtype=int) # 0=-, 1=neu, 2=+
        state_str = self.pack_vec(state_vec)

        self.mol0 = Chem.MolFromSmiles(self.smiles_raw) # only for first qupkake run

        self.all_states.append([state_str])
        if self.verbose:
            print(self.all_states)

    @staticmethod
    def load_molecules(fname,removeHs=True):
        with Chem.SDMolSupplier(fname, removeHs=removeHs) as suppl:
            ms = [x for x in suppl if x is not None]
        # print(f'number of pka predictions: {len(ms)}')
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

    def prep_and_run(self,discard_cutoff=0.3):
        # Prep, run, and analyse states
        last_states_curated = []
        for state_str in self.all_states[-1]:
            # Prep and run qupkake
            fname = f'{self.name}_{state_str}'
            if self.verbose:
                print(f'Preparing qupkake input for {fname}.')
            
            self.prep_qupkake_input(state_str,fname)
            ret = self.run_qupkake(fname)

            if ret != 0: # qupkake run failed
                print(f'Qupkake run failed.')
                print(f'Flagging state {state_str} for deletion.')       
                self.flagged_states.append(state_str)
                ps = np.zeros((2,self.N)) - 1 # blank state
                # continue
            # analyze qupkake runs
            else:
                if self.verbose:
                    print(f'Analysing results for {fname}.')
                ms = self.load_molecules(f'{self.path_out_qupkake}/{fname}.sdf')

                # Addresses re-ordering of atoms by qupkake
                if state_str == '1'*self.N:
                    self.mol0 = ms[0]
                    symbols = [at.GetSymbol() for at in self.mol0.GetAtoms()]
                    symbols = "".join(symbols)
                    self.smiles_input = Chem.MolToSmiles(self.mol0,canonical=False)
                    if self.verbose:
                        print('Updating mol0 order and input smiles.')
                        print(f'Atom order processed: {symbols}')

                ps = self.calc_ps(ms)

                smiles = Chem.MolToSmiles(ms[0],canonical=False)

                # if self.verbose:
                print(f'Qupkake output smiles (tautomer search): {smiles}')

            self.all_ps[state_str] = list(ps)
            last_states_curated.append(state_str)
        if self.verbose:
            print(f'Old last states: {self.all_states[-1]}')
            print(f'Curated last states: {last_states_curated}')
        self.all_states[-1] = last_states_curated

    def select_new_states(self):
        # Select new states based on last runs
        
        new_states = self.add_states(self.all_states[-1])
        if self.verbose:
            print('Selecting new states.')
            print(f'New states: {new_states}')

        new_states_curated = []

        for state_str in new_states:
            found = self.check_found(state_str)
            # discarded = self.check_discarded(state_str)
            if (not found):# and (not discarded):
                new_states_curated.append(state_str)
        if self.verbose:
            print(f'New states curated: {new_states_curated}')
        if len(new_states_curated) == 0:
            if self.verbose:
                print('No new states found. Exiting.')
            return -1
        else:
            self.all_states.append(new_states_curated)
        return 0

    def run_predictions(self):
        print(f'='*50)
        print(self.name)
        self.read_input()
        for cycle in range(self.ncycles):
            print('-'*50)
            print(f'CYCLE {cycle}')
            self.prep_and_run()
            ret = self.select_new_states()
            if ret != 0: # return code
                break
        self.dump_results()

    #########################################################

    def protonate_molecules(self,mol,state_vec,addHs=False):
        mol_prot = copy.deepcopy(mol)#.copy()
        if self.verbose:
            print(self.pack_vec(state_vec))
            print(f'Before (de)protonation: {Chem.MolToSmiles(mol_prot,canonical=False)}')
        for idx, s in enumerate(state_vec):
            if s > 1:
                mol_prot = self.set_protonation(mol_prot,idx,1)
            elif s < 1:
                mol_prot = self.set_protonation(mol_prot,idx,-1)
        if addHs:
            mol_prot = Chem.AddHs(mol_prot,addCoords=True)
        if self.verbose:
            print(f'After (de)protonation: {Chem.MolToSmiles(mol_prot,canonical=False)}')
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
            print(f'q:{Chem.GetFormalCharge(mol_new)}, n_atoms:{mol_new.GetNumAtoms()}')
        if output == 'sdf':
            self.write_molecule(mol_new,self.path_inp_qupkake,fname)
        elif output == 'smiles':
            self.write_smiles(mol_new,self.path_inp_qupkake,fname)
        self.clear_qupkake_output(fname)

    def run_qupkake(self,fname,input='smiles'):
        if input == 'sdf':
            os.system(f'qupkake file -r data -o {fname}.sdf {self.path_inp_qupkake}/{fname}.sdf')
        elif input == 'smiles':
            with open(f'{self.path_inp_qupkake}/{fname}.smi') as f:
                line = f.readline()
                spl = line.split()
                smiles, fname = spl[0], spl[1]
            print(fname, smiles)
            # if os.path.isfile(f'{self.path_out_qupkake}/{fname}.sdf'):
            #     os.remove(f'{self.path_out_qupkake}/{fname}.sdf')
            os.system(f'qupkake smiles -r data -o {fname}.sdf "{smiles}"') # -t
        if not os.path.isfile(f'data/output/{fname}.sdf'):
            return -1
        else:
            os.system(f'cp data/output/{fname}.sdf {self.path_out_qupkake}/')
            os.system(f'rm -r data')
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
            print(at_idx, pka_type, pka, f'up:{p_up:.2f}', f'down:{p_down:.2f}')
            if pka_type == 'basic':
                # print('basic',q_up)
                ps[0,at_idx] = p_up
                # print(qs)
            elif pka_type == 'acidic':
                ps[1,at_idx] = p_down
        if self.verbose:
            print(ps)
        return ps

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

    def add_states(self,last_states):
        new_states = []
        if self.verbose:
            print(f'Last states: {last_states}')
        for state_str in last_states:
            ps = self.all_ps[state_str]
            state_vec = self.unpack_vec(state_str)
            for idx, p_up in enumerate(ps[0]): # up because basic
                if (p_up > self.cutoff) and (state_vec[idx] < 2):
                    state_new_str = self.add_new_state(state_vec,idx,1)
                    if state_new_str in new_states:
                        if self.verbose:
                            print(f'{state_new_str} already in new_states {new_states}.')
                    else:
                        new_states.append(state_new_str)
            for idx, p_down in enumerate(ps[1]): # down because acidic
                if (p_down > self.cutoff) and (state_vec[idx] > 0):
                    state_new_str = self.add_new_state(state_vec,idx,-1)
                    if state_new_str in new_states:
                        if self.verbose:
                            print(f'{state_new_str} already in new_states {new_states}.')
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
        for states in self.all_states:
            if state_str in states:
                found = True
                if self.verbose:
                    print(f'State {state_str} already found.')
        return found
    
    def check_discarded(self,state_str):
        # print(f'Checking {state_str} for discarded.')
        discarded = False
        if state_str in self.flagged_states:
            discarded = True
            if self.verbose:
                print(f'State {state_str} already discarded.')
        return discarded

    def dump_results(self,path='autoprot_cache'):
        os.makedirs(path,exist_ok=True)
        with open(f'{path}/{self.name}_states.yaml','w') as stream:
            yaml.dump(self.all_states,stream)
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
        self.simulate_state_traj()
        self.calc_optimal_state()
        self.plot_states_freq()
        self.export_optimal_state()
        self.draw_all_states()
        self.draw_labeled_state()

    def prep_state_selection(self):

        self.all_ps = self.load_results()

        self.all_states_list = np.array([self.unpack_vec(state) for state in list(self.all_ps.keys())])
        self.N_states = len(self.all_states_list)
        self.relevant_indices = np.unique(np.where(self.all_states_list != 1)[1]) # find indices that change between states
        
        if self.verbose:
            print(self.all_states_list)
            print(f'N_states: {self.N_states}')
            print(f'relevant indices: {self.relevant_indices}')

        self.states = self.all_states_list[:,self.relevant_indices] # (state,index)
        self.states_str = [self.pack_vec(state) for state in self.states]

        self.ps_list = np.array([self.all_ps[st] for st in self.all_ps.keys()])
        self.ps_list = self.ps_list[:,:,self.relevant_indices] # states, up/down, indices
        if self.verbose:
            print(self.ps_list)

    def get_s_idx(self,state,states_str):
        return states_str.index(self.pack_vec(state))

    def calc_tmatrix(self):
        """ Transition matrix between molecule protonation states"""

        tmatrix_raw = [[[] for _ in range(len(self.states))] for _ in range(len(self.states))] # N_states x N_states (x duplicate predictions)

        for s_idx, state in enumerate(self.states):
            if self.verbose:
                print('------')
                print(s_idx, state)
            ps_up = self.ps_list[s_idx,0]
            ps_down = self.ps_list[s_idx,1]

            recipes = [
                [ps_up, 1],
                [ps_down, -1]
            ]

            for rec in recipes:
                ps = rec[0]
                dq = rec[1]

                for at_idx, p in enumerate(ps):
                    # print(at_idx, p)
                    if p > -1.:
                        state_target = state.copy()
                        state_target[at_idx] += dq
                        if self.verbose:
                            print(f'target: {state_target}')
                        if self.pack_vec(state_target) in self.states_str:
                            c_target_idx = self.get_s_idx(state_target,self.states_str)
                            tmatrix_raw[c_target_idx][s_idx].append(p) # to, from
                            tmatrix_raw[s_idx][c_target_idx].append(1-p)

        # print('='*50)

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
            print(self.states_str)
            print(self.tmatrix)

    #########################################################

    def simulate_state_traj(self,nsteps=1000000):
        """ Simulate evolution according to transition matrix """

        choice0 = np.random.randint(len(self.states))

        self.traj = [choice0]
        self.traj_states = [self.states[choice0]]

        for t in range(1,nsteps):
            trans = self.tmatrix.T[self.traj[t-1]]
            choice = np.random.choice(np.arange(self.N_states),p=trans)
            # print(traj[t-1], trans, choice)
            self.traj.append(choice)
            self.traj_states.append(self.states[choice])

        self.traj_states = np.array(self.traj_states)
        self.traj = np.array(self.traj)

        # eigenvalues, eigenvectors = np.linalg.eig(tmatrix)
        # print(eigenvalues)

    #########################################################

    def calc_optimal_state(self):
        self.states_mean = np.mean(self.traj_states,axis=0)

        self.states_freq = np.zeros((self.N_states))

        for idx in range(self.N_states):
            self.states_freq[idx] = len(np.where(self.traj == idx)[0])
        self.states_freq /= np.sum(self.states_freq)
        self.states_freq *= 100
        # print(self.states_freq)

        print('State | Occupancy')
        for state_str, freq in zip(self.states_str, self.states_freq):
            print(f'{state_str} | {freq:.2f}%')

        idx_max = np.argmax(self.states_freq)

        self.state_vec_opti = self.reconstruct_full_state(self.unpack_vec(self.states_str[idx_max]))
        self.state_str_opti = self.pack_vec(self.state_vec_opti)

        print(f'Most likely state: {self.state_vec_opti} at {self.states_freq[idx_max]:.2f}%')

    def reconstruct_full_state(self,state):
        """ Reconstruct full state vector from state vector of relevant indices"""
        state_vec = np.ones((self.N),dtype=int)
        for idx, st in zip(self.relevant_indices, state):
            state_vec[idx] = st # np.round(st) # Should be int now
        return state_vec

    #########################################################

    def plot_states_freq(self):
        print(f'N_states: {self.N_states}')
        fig, ax = plt.subplots(figsize=(self.N_states/3.,3.5))
        xs = np.arange(self.N_states)
        ax.bar(xs, self.states_freq)
        ax.set_xticks(xs)
        ax.set_xticklabels(self.states_str,rotation=-45)
        ax.set(xlabel='States',ylabel='Occupancy [%]')

        for idx, st in zip(self.relevant_indices, self.states_mean):
            print(f'Atom {idx}: Charge: {st-1:.2f}')
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
        for state_vec in self.all_states_list:
            state_str = self.pack_vec(state_vec)
            sdf = f'qupkake_output/{self.name}_{state_str}.sdf'
            with Chem.SDMolSupplier(sdf) as suppl:
                tmp = [x for x in suppl if x is not None]
            for m in tmp: _=AllChem.Compute2DCoords(m)
            mol = tmp[0]
            ms.append(mol)
        img=Draw.MolsToGridImage(ms,molsPerRow=5,subImgSize=(200,200),
                                 legends=[f'{self.states_freq[idx]:.1f}%' for idx in range(len(ms))],
                                 returnPNG=False,useSVG=True)

        with open(f'{self.path_out_autoprot}/{self.name}_all_states.svg','w') as f:
            f.write(img.data)
        cairosvg.svg2pdf(url=f'{self.path_out_autoprot}/{self.name}_all_states.svg',
                 write_to=f'{self.path_out_autoprot}/{self.name}_all_states.pdf')

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
            print('Optimizing geometry of optimal state.')
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
        self.write_log()

    def write_log(self):
        with open(f'{self.path_out_autoprot}/{self.name}.log','w') as f:
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
            for idx, st in zip(self.relevant_indices, self.states_mean):
                f.write(f'Atom {idx}: Charge: {st-1:.2f}\n')
            f.write('\n')

            f.write(f'Protonation state occupancies:\n')
            f.write('State | Occupancy\n')
            for state_str, freq in zip(self.states_str, self.states_freq):
                f.write(f'{state_str} | {freq:.2f}%\n')
            f.write('\n')

            f.write(f'Smiles of optimal protonation state: {self.smiles_optimal}\n')

#         if output == 'sdf':
#             self.write_molecule(mol_new,self.path_inp,fname)
#         elif output == 'smiles':
#             self.write_smiles(mol_new,self.path_inp,fname)
