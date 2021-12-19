#!/usr/bin/python
# -*- coding:utf-8 -*-
from copy import copy, deepcopy
import networkx as nx
from rdkit import Chem

from utils.chem_utils import smi2mol, mol2smi
from utils.chem_utils import get_submol, get_submol_atom_map


class PieceNode:
    def __init__(self, smiles: str, pos: int, atom_mapping: dict):
        self.smiles = smiles
        self.pos = pos
        self.mol = smi2mol(smiles)
        self.atom_mapping = copy(atom_mapping)  # map atom idx in the whole mol to those in the piece
    
    def get_mol(self):
        '''return molecule in rdkit form'''
        return self.mol

    def get_atom_mapping(self):
        return copy(self.atom_mapping)

    def __str__(self):
        return f'''
                    smiles: {self.smiles},
                    position: {self.pos}
                '''


class PieceEdge:
    def __init__(self, src: int, dst: int, edges: list):
        self.edges = copy(edges)  # list of tuple (a, b, type) where the canonical order is used
        self.src = src
        self.dst = dst
        self.dummy = False
        if len(self.edges) == 0:
            self.dummy = True
    
    def get_edges(self):
        return copy(self.edges)
    
    def get_num_edges(self):
        return len(self.edges)

    def __str__(self):
        return f'''
                    src piece: {self.src}, dst piece: {self.dst},
                    atom bonds: {self.edges}
                '''


class Molecule(nx.Graph):
    '''molecule represented in piece-level'''

    def __init__(self, smiles: str=None, groups: list=None):
        super().__init__()
        if smiles is None:
            return
        self.graph['smiles'] = smiles
        rdkit_mol = smi2mol(smiles)
        # processing atoms
        aid2pos = {}
        for pos, group in enumerate(groups):
            for aid in group:
                aid2pos[aid] = pos
            piece_mol = get_submol(rdkit_mol, group)
            piece_smi = mol2smi(piece_mol)
            atom_mapping = get_submol_atom_map(rdkit_mol, piece_mol, group)
            node = PieceNode(piece_smi, pos, atom_mapping)
            self.add_node(node)
        # process edges
        edges_arr = [[[] for _ in groups] for _ in groups]  # adjacent
        for edge_idx in range(rdkit_mol.GetNumBonds()):
            bond = rdkit_mol.GetBondWithIdx(edge_idx)
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            begin_piece_pos = aid2pos[begin]
            end_piece_pos = aid2pos[end]
            begin_mapped = self.nodes[begin_piece_pos]['piece'].atom_mapping[begin]
            end_mapped = self.nodes[end_piece_pos]['piece'].atom_mapping[end]
            bond_type = bond.GetBondType()
            edges_arr[begin_piece_pos][end_piece_pos].append((begin_mapped, end_mapped, bond_type))
            edges_arr[end_piece_pos][begin_piece_pos].append((end_mapped, begin_mapped, bond_type))
        # add egdes into the graph
        for i in range(len(groups)):
            for j in range(len(groups)):
                if not i < j or len(edges_arr[i][j]) == 0:
                    continue
                edge = PieceEdge(i, j, edges_arr[i][j])
                self.add_edge(edge)
    
    @classmethod
    def from_nx_graph(cls, graph: nx.Graph, deepcopy=True):
        if deepcopy:
            graph = deepcopy(graph)
        graph.__class__ = Molecule
        return graph

    @classmethod
    def merge(cls, mol0, mol1, edge=None):
        # reorder
        node_mappings = [{}, {}]
        mols = [mol0, mol1]
        mol = Molecule.from_nx_graph(nx.Graph())
        for i in range(2):
            for n in mols[i].nodes:
                node_mappings[i][n] = len(node_mappings[i])
                node = deepcopy(mols[i].get_node(n))
                node.pos = node_mappings[i][n]
                mol.add_node(node)
            for src, dst in mols[i].edges:
                edge = deepcopy(mols[i].get_edge(src, dst))
                edge.src = node_mappings[i][src]
                edge.dst = node_mappings[i][dst]
                mol.add_edge(src, dst, connects=edge)
        # add new edge
        edge = deepcopy(edge)
        edge.src = node_mappings[0][edge.src]
        edge.dst = node_mappings[1][edge.dst]
        mol.add_edge(edge)
        return mol

    def get_edge(self, i, j) -> PieceEdge:
        return self[i][j]['connects']
    
    def get_node(self, i) -> PieceNode:
        return self.nodes[i]['piece']

    def add_edge(self, edge: PieceEdge):
        src, dst = edge.src, edge.dst
        super().add_edge(src, dst, connects=edge)
    
    def add_node(self, node: PieceNode):
        n = node.pos
        super().add_node(n, piece=node)

    def subgraph(self, nodes: list, deepcopy=True):
        graph = super().subgraph(nodes)
        assert isinstance(graph, Molecule)
        return graph

    def to_rdkit_mol(self):
        mol = Chem.RWMol()
        aid_mapping = {}
        # add all the pieces to rwmol
        for n in self.nodes:
            piece = self.get_node(n)
            submol = piece.get_mol()
            for atom in submol.GetAtoms():
                new_atom = Chem.Atom(atom.GetSymbol())
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                mol.AddAtom(atom)
                aid_mapping[(n, atom.GetIdx())] = len(aid_mapping)
            for bond in submol.GetBonds():
                begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                begin, end = aid_mapping[(n, begin)], aid_mapping[(n, end)]
                mol.AddBond(begin, end, bond.GetBondType())
        for src, dst in self.edges:
            piece_edge = self.get_edge(src, dst)
            pid_src, pid_dst = piece_edge.src, piece_edge.dst
            for begin, end, bond_type in piece_edge.edges:
                begin, end = aid_mapping[(pid_src, begin)], aid_mapping[(pid_dst, end)]
                mol.AddBond(begin, end, bond_type)
        mol = mol.GetMol()
        # sanitize, firstly handle mal-formed N+
        mol.UpdatePropertyCache(strict=False)
        ps = Chem.DetectChemistryProblems(mol)
        if not ps:
            Chem.SanitizeMol(mol)
            return mol
        for p in ps:
            if p.GetType()=='AtomValenceException':  # N
                at = mol.GetAtomWithIdx(p.GetAtomIdx())
                if at.GetAtomicNum()==7 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
                    at.SetFormalCharge(1)
        Chem.SanitizeMol(mol)
        return mol

    def to_smiles(self):
        rdkit_mol = self.to_rdkit_mol()
        return mol2smi(rdkit_mol)

    def __str__(self):
        desc = 'nodes: \n'
        for ni, node in enumerate(self.nodes):
            desc += f'{ni}:{self.get_node(node)}\n'
        desc += 'edges: \n'
        for src, dst in self.edges:
            desc += f'{src}-{dst}:{self.get_edge(src, dst)}\n'
        return desc



if __name__ == '__main__':
    import sys
