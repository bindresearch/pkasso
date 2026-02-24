import torch

from .ionization_group import get_ionization_aid
from .descriptor import mol2vec
from .net import GCNNet

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

def predict_acid_base(mol_h,model_base,model_acid,device='cpu',verbose=False,
                      pred_acid=True, pred_base=True):

    if pred_base:
        base = predict_base(mol_h,model_base,device=device)

        base_curated = {} # atom mapping

        for at_idx, pka in base.items():
            atom = mol_h.GetAtomWithIdx(at_idx) 
            map_idx = atom.GetAtomMapNum()
            print(at_idx, map_idx)
            base_curated[map_idx] = pka

        print('base')
        print(base)
        print('base curated')
        print(base_curated)
        base = base_curated
    else:
        base = {}

    if pred_acid:
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

        acid_curated = {} # atom mapping

        for at_idx, pka in acid.items():
            atom = mol_h.GetAtomWithIdx(at_idx) 
            map_idx = atom.GetAtomMapNum()
            print(at_idx, map_idx)
            acid_curated[map_idx] = pka

        print('acid')
        print(acid)
        print('acid curated')
        print(acid_curated)
        acid = acid_curated
    else:
        acid = {}
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