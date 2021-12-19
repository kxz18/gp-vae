from collections import defaultdict
from queue import Queue
import heapq
from copy import copy

import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import AllChem
import torch

MAX_VALENCE = {'B': 3, 'Br':1, 'C':4, 'Cl':1, 'F':1, 'I':1, 'N':5, 'O':2, 'P':5, 'S':6} #, 'Se':4, 'Si':4}
Bond_List = [None, BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE]  # aromatic bonds are shits


class AtomVocab:
    def __init__(self, atom_special=None, bond_special=None):
        self.max_Hs = max(MAX_VALENCE.values())
        self.min_formal_charge = -2
        self.max_formal_charge = 2
        # atom
        self.idx2atom = list(MAX_VALENCE.keys())
        if atom_special is None:
            atom_special = []
        self._atom_pad = '<apad>'
        atom_special.append(self._atom_pad)
        self.idx2atom += atom_special
        self.atom2idx = { atom: i for i, atom in enumerate(self.idx2atom) }
        # bond
        self.idx2bond = copy(Bond_List)
        self._bond_pad = '<bpad>'
        if bond_special is None:
            bond_special = []
        bond_special.append(self._bond_pad)
        self.idx2bond += bond_special
        self.bond2idx = { bond: i for i, bond in enumerate(self.idx2bond) }
        
        self.atom_special = atom_special
        self.bond_special = bond_special
    
    def get_atom_vector(self, atom, explictHs, formal_charge):
        atomic = np.zeros((len(self.idx2atom)))
        atomic[self.atom_to_idx(atom)] = 1
        explict = np.zeros((self.max_Hs + 1))
        explict[explictHs] = 1
        charge = np.zeros((self.max_formal_charge - self.min_formal_charge + 1))
        charge[formal_charge - self.min_formal_charge] = 1
        return np.concatenate([atomic, explict, charge])
    
    def dim_atom_vector(self):
        return len(self.idx2atom) + self.max_Hs + 1 + \
               self.max_formal_charge - self.min_formal_charge + 1
    
    def atom_vector_to_atom(self, vector):
        if isinstance(vector, torch.Tensor):
            vector = vector.cpu().numpy()
        elif isinstance(vector, list):
            vector = np.array(vector)
        split_idx = [len(self.idx2atom), self.max_Hs + 1]
        split_idx[1] += split_idx[0]
        atom_vec = vector[:split_idx[0]]
        explictHs_vec = vector[split_idx[0]:split_idx[1]]
        charge_vec = vector[split_idx[1]:]
        symbol = self.idx_to_atom(atom_vec.argmax())
        explictHs = int(explictHs_vec.argmax())
        charge = int(charge_vec.argmax()) + self.min_formal_charge
        atom = Chem.Atom(symbol)
        # atom.SetNumExplicitHs(explictHs)
        atom.SetFormalCharge(charge)
        return atom

    def idx_to_atom(self, idx):
        return self.idx2atom[idx]
    
    def atom_to_idx(self, atom):
        return self.atom2idx[atom]

    def atom_pad(self):
        return self._atom_pad
    
    def atom_pad_idx(self):
        return self.atom_to_idx(self.atom_pad())

    def idx_to_bond(self, idx):
        return self.idx2bond[idx]
    
    def bond_to_idx(self, bond):
        return self.bond2idx[bond]

    def bond_pad(self):
        return self._bond_pad

    def bond_pad_idx(self):
        return self.bond_to_idx(self.bond_pad())
    
    def bond_idx_to_valence(self, idx):
        bond_enum = self.idx2bond[idx]
        if bond_enum == BondType.SINGLE:
            return 1
        elif bond_enum == BondType.DOUBLE:
            return 2
        elif bond_enum == BondType.TRIPLE:
            return 3
        else:   # invalid bond
            return -1
    
    def num_atom_type(self):
        return len(self.idx2atom)
    
    def num_bond_type(self):
        return len(self.idx2bond)


def smi2mol(smiles: str, kekulize=True):
    '''turn smiles to molecule'''
    mol = Chem.MolFromSmiles(smiles)
    if kekulize:
        Chem.Kekulize(mol, True)
    return mol


def mol2smi(mol):
    return Chem.MolToSmiles(mol)


def similarity(mol1, mol2):
    fps1 = AllChem.GetMorganFingerprint(mol1, 2)
    fps2 = AllChem.GetMorganFingerprint(mol2, 2)
    return DataStructs.TanimotoSimilarity(fps1, fps2)


def get_submol(mol, atom_indices):
    if len(atom_indices) == 1:
        return smi2mol(mol.GetAtomWithIdx(atom_indices[0]).GetSymbol())
    aid_dict = { i: True for i in atom_indices }
    edge_indices = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        begin_aid = bond.GetBeginAtomIdx()
        end_aid = bond.GetEndAtomIdx()
        if begin_aid in aid_dict and end_aid in aid_dict:
            edge_indices.append(i)
    mol = Chem.PathToSubmol(mol, edge_indices)
    return mol


def get_submol_atom_map(mol, submol, group):
    # print(mol2smi(mol))
    # print(mol2smi(submol))
    # print(group)
    # turn to smiles order
    smi = mol2smi(submol)
    submol = smi2mol(smi)
    # special with N+ and N- (fuck this shit)
    for atom in submol.GetAtoms():
        if atom.GetSymbol() == 'N' and (atom.GetExplicitValence() == 3 and atom.GetFormalCharge() == 1) or atom.GetExplicitValence() < 3:
            atom.SetNumRadicalElectrons(0)
            atom.SetNumExplicitHs(2)
    
    matches = mol.GetSubstructMatches(submol)
    old2new = { i: 0 for i in group }  # old atom idx to new atom idx
    found = False
    for m in matches:
        hit = True
        for i, atom_idx in enumerate(m):
            if atom_idx not in old2new:
                hit = False
                break
            old2new[atom_idx] = i
        if hit:
            found = True
            break
    assert found
    return old2new


def data2molecule(vocab, data, sanitize=True):
    '''turn PyG data to molecule'''
    mol = Chem.RWMol()
    idx2atom = []
    for atom_idx in data.x:     # add atoms
        if not isinstance(atom_idx, int):  # one-hot form
            atom_idx = torch.argmax(atom_idx[:len(vocab)])
        atom = Chem.Atom(vocab.idx_to_atom(int(atom_idx)))
        idx2atom.append(mol.AddAtom(atom))
    edge_list = data.edge_index.t()  # [num, 2]
    edge_dict = {}
    for edge, attr in zip(edge_list, data.edge_attr):
        i1, i2 = edge
        i1, i2 = int(i1), int(i2)
        if i1 > i2:
            i1, i2 = i2, i1
        key = f'{i1},{i2}'
        if key in edge_dict:
            continue
        edge_dict[key] = True
        a1, a2 = idx2atom[i1], idx2atom[i2]
        if len(attr) > 1:
            attr = torch.argmax(attr)
        bond_type = vocab.get_bond_enum(int(attr))
        # if bond_type == BondType.AROMATIC:
        #     bond_type = BondType.DOUBLE
        mol.AddBond(a1, a2, bond_type)
    mol = mol.GetMol()
    if sanitize:
        Chem.SanitizeMol(mol)
    return mol


def mol2file(mol, file_name, grid=False, molsPerRow=6, legends=None):
    if isinstance(mol, list):
        if not grid:
            for i, m in enumerate(mol):
                Draw.MolToFile(m, f'{i}_{file_name}')
        else:
            if legends is not None:
                assert len(legends) == len(mol)
                img = Draw.MolsToGridImage(mol, molsPerRow=molsPerRow, subImgSize=(400, 400), legends=legends)
            else:
                img = Draw.MolsToGridImage(mol, molsPerRow=molsPerRow, subImgSize=(400, 400))
            with open(file_name, 'wb') as fig:
                img.save(fig)
            
    else:
        Draw.MolToFile(mol, file_name)


def cnt_atom(smi, return_dict=False):
    atom_dict = { atom: 0 for atom in MAX_VALENCE }
    for i in range(len(smi)):
        symbol = smi[i].upper()
        next_char = smi[i+1] if i+1 < len(smi) else None
        if symbol == 'B' and next_char == 'r':
            symbol += next_char
        elif symbol == 'C' and next_char == 'l':
            symbol += next_char
        if symbol in atom_dict:
            atom_dict[symbol] += 1
    if return_dict:
        return atom_dict
    else:
        return sum(atom_dict.values())


def get_base64(mol):
    return Chem.RDKFingerprint(mol).ToBase64()


def dfs_order(mol, root):
    '''return list of atoms in dfs order and idx2order mapping dict'''
    stack = [root]
    visited = {}
    order_list = []
    idx2order = {}
    visited[root.GetIdx()] = True
    while stack:
        atom = stack.pop()
        idx2order[atom.GetIdx()] = len(order_list)
        order_list.append(atom)
        for nei in atom.GetNeighbors():
            idx = nei.GetIdx()
            if idx not in visited:
                stack.append(nei)
                visited[idx] = True
    return order_list, idx2order


def bfs_order(mol, root):
    queue = Queue()
    queue.put(root)
    visited = {}
    order_list = []
    idx2order = {}
    visited[root.GetIdx()] = True
    while not queue.empty():
        atom = queue.get()
        idx2order[atom.GetIdx()] = len(order_list)
        order_list.append(atom)
        for nei in atom.GetNeighbors():
            idx = nei.GetIdx()
            if idx not in visited:
                queue.put(nei)
                visited[idx] = True
    return order_list, idx2order


def bfs_order_by_admat(admat):
    root_idx = 0
    queue = Queue()
    queue.put(root_idx)
    visited = {}
    order_list = []
    idx2order = {}
    visited[root_idx] = True
    while not queue.empty():
        next_id = queue.get()
        idx2order[next_id] = len(order_list)
        order_list.append(next_id)
        neis = []
        for nei_id, has_edge in enumerate(admat[next_id]):
            if not has_edge:
                continue
            if nei_id not in visited:
                visited[nei_id] = True
                queue.put(nei_id)
    return order_list, idx2order


def shortest_path_len(i, j, mol):
    queue = Queue()
    queue.put((mol.GetAtomWithIdx(i), 1))
    visited = {}
    visited[i] = True
    while not queue.empty():
        atom, dist = queue.get()
        neis = []
        for nei in atom.GetNeighbors():
            idx = nei.GetIdx()
            if idx == j:
                return dist + 1
            if idx not in visited:
                visited[idx] = True
                neis.append(idx)
                queue.put((mol.GetAtomWithIdx(idx), dist + 1))
    return None


def cycle_check(i, j, mol):
    cycle_len = shortest_path_len(i, j, mol)
    return cycle_len is None or (cycle_len > 4 and cycle_len < 7)


def valence_check(aid1, aid2, edges1, edges2, new_edge, vocab, c1=0, c2=0):
    new_valence = vocab.bond_idx_to_valence(new_edge)
    if new_valence == -1:
        return False
    atom1 = vocab.idx_to_atom(aid1)
    atom2 = vocab.idx_to_atom(aid2)
    a1_val = sum(list(map(vocab.bond_idx_to_valence, edges1)))
    a2_val = sum(list(map(vocab.bond_idx_to_valence, edges2)))
    # special for S as S is likely to have either 2 or 6 valence
    if (atom1 == 'S' and a1_val == 2) or (atom2 == 'S' and a2_val == 2):
        return False
    return a1_val + new_valence + abs(c1) <= MAX_VALENCE[atom1] and \
           a2_val + new_valence + abs(c2) <= MAX_VALENCE[atom2]


def get_random_submol(mol):  # use bfs order and randomly drop 1-5 atoms
    root_idx = np.random.randint(0, mol.GetNumAtoms())
    root_atom = mol.GetAtomWithIdx(root_idx)
    order_list, idx2order = bfs_order(mol, root_atom)
    drop_num = np.random.randint(0, 5)
    rw_mol = Chem.RWMol()
    for atom in mol.GetAtoms():
        atom_sym = atom.GetSymbol()
        new_atom = Chem.Atom(atom_sym)
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        rw_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rw_mol.AddBond(begin, end, bond.GetBondType())
    if drop_num == 0:
        return rw_mol.GetMol()
    removed = [atom.GetIdx() for atom in order_list[-drop_num:]]
    removed = sorted(removed, reverse=True)
    for idx in removed:
        rw_mol.RemoveAtom(idx)
    return rw_mol.GetMol()
