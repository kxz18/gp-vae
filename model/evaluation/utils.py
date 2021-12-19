#!/usr/bin/python
# -*- coding:utf-8 -*-
import pickle
import argparse
import multiprocessing as mp
# import numpy as np
import networkx as nx
from rdkit import Chem, DataStructs
from rdkit.Chem.QED import qed
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import RDConfig
# import sascorer
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
sys.path.remove(os.path.join(RDConfig.RDContribDir, 'SA_Score'))

from evaluation.rf import smiles2molecule, ecfpn
from utils.chem_utils import rec


def similarity(mol1, mol2):
    fps1 = AllChem.GetMorganFingerprint(mol1, 2)
    fps2 = AllChem.GetMorganFingerprint(mol2, 2)
    return DataStructs.TanimotoSimilarity(fps1, fps2)


def num_long_cycles(mol):
    """Calculate the number of long cycles.
    Args:
      mol: Molecule. A molecule.
    Returns:
      negative cycle length.
    """
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if not cycle_list:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return -cycle_length


def get_sa(molecule):
    '''return synthesis accessibility of given molecule.
       The value ranges from 1 to 10, the lower, the better (the easier to make)'''
    return sascorer.calculateScore(molecule)


# def get_penalized_logp(molecule):
#     '''Penalized logP of given molecule. 
#        Penalized logP is octanol-water partition coefficients (logP) penalized by the synthetic accessibility (SA)
# score and number of long cycles, the higher, the better'''
#     log_p = Descriptors.MolLogP(molecule)
#     sas_score = get_sa(molecule)
#     cycle_score = num_long_cycles(molecule)
#     return log_p - sas_score + cycle_score
def get_penalized_logp(mol):
    """
    Reward that consists of log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset
    :param mol: rdkit mol object
    :return: float
    """
    # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = -sascorer.calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(
        Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length
    
    # return log_p + SA + cycle_score
    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    return normalized_log_p + normalized_SA + normalized_cycle


def get_qed(molecule):
    '''QED of given molecule. The value ranges from 0 to 1, the higher, the better'''
    return qed(molecule)


# following property need to explicitly call init
root = os.path.split(os.path.abspath(__file__))[0]
gsk3b = None
jnk3 = None
def init(root_path=root):
    global root, gsk3b, jnk3
    root = root_path
    with open(os.path.join(root, 'gsk3b.pkl'), 'rb') as fin:
        gsk3b = pickle.load(fin)
    with open(os.path.join(root, 'jnk3.pkl'), 'rb') as fin:
        jnk3 = pickle.load(fin)


def get_gsk3b_active(molecule):
    '''return 1 if the molecule has gsk3b activity, else 0'''
    global gsk3b
    if gsk3b is None:
        init()
    ecfp4 = ecfpn(molecule)
    pred = gsk3b.predict_proba([ecfp4])[0]  # (prob of 0, prob of 1)
    return pred[1]  # only report probability of being active


def get_jnk3_active(molecule):
    '''return 1 if the molecule has jnk3 activity, else 0'''
    global jnk3
    if jnk3 is None:
        init()
    ecfp4 = ecfpn(molecule)
    pred = jnk3.predict_proba([ecfp4])[0]
    return pred[1]


def eval_funcs_dict():
    eval_funcs = {
        'qed': get_qed,
        'sa': get_sa,
        'logp': get_penalized_logp,
        'gsk3b': get_gsk3b_active,
        'jnk3': get_jnk3_active
    }
    return eval_funcs


# use gaussian distribution to normalize all scores
# STATS_FILE = os.path.join(os.path.split(__file__)[0], 'stats.pkl')
# STATS_DICT = None
# 
# 
# def init_stats(fname, cpus):
#     global STATS_FILE, PROPS
#     with open(fname, 'r') as fin:
#         lines = fin.read().strip().split('\n')
#     with mp.Pool(cpus) as pool:
#         mols = pool.map(smiles2molecule, lines)
#         eval_funcs = eval_funcs_dict()
#         prop_vals = {}
#         for pname in PROPS:
#             prop_vals[pname] = pool.map(eval_funcs[pname], mols)
#     # calculate mean and var
#     for pname in prop_vals:
#         np_array = np.array(prop_vals[pname])
#         mean = np.mean(np_array, axis=0)
#         var = np.std(np_array, axis=0)
#         prop_vals[pname] = (mean, var)
#     with open(STATS_FILE, 'wb') as fout:
#         pickle.dump(prop_vals, fout)


def get_normalized_property_scores(mol):
    # make every property approximately in range (0, 1), and all are the higher, the better
    # note: shitty normalization might be divide by mean value
    # global STATS_DICT, STATS_FILE, PROPS
    # if STATS_DICT is None:
    #     with open(STATS_FILE, 'rb') as fin:
    #         STATS_DICT = pickle.load(fin)
    # if isinstance(mol, str):
    #     mol = smiles2molecule(mol)
    # eval_funcs = eval_funcs_dict()
    # res = []
    # for pname in PROPS:
    #     val = eval_funcs[pname](mol)
    #     mean, var = STATS_DICT[pname]
    #     val = (val - mean) / var
    #     if pname == 'sa':
    #         val = -val # because only sa is the lower, the better
    #     res.append(val)
    # return res
    qed = get_qed(mol)
    sa = get_sa(mol)
    logp = get_penalized_logp(mol)
    gsk3b = get_gsk3b_active(mol)
    jnk3 = get_jnk3_active(mol)
    return [qed, 1 - sa / 10, (logp + 10) / 13, gsk3b, jnk3]  # all are the higher, the better


def restore_property_scores(normed_props):
    # global STATS_DICT, STATS_FILE, PROPS
    # if STATS_DICT is None:
    #     with open(STATS_FILE, 'rb') as fin:
    #         STATS_DICT = pickle.load(STATS_FILE)
    # res = []
    # for pname, val in zip(PROPS, normed_props):
    #     mean, var = STATS_DICT[pname]
    #     if pname == 'sa':
    #         val = -val
    #     val = val * var + mean
    #     res.append(val)
    # return res
    return [normed_props[0], 10 * (1 - normed_props[1]),
            13 * normed_props[2] - 10, normed_props[3], normed_props[4]]


PROP_TH = [0.6, 4.0, 0, 0.5, 0.5]
NORMALIZED_TH = None
PROPS = ['qed', 'sa', 'logp', 'gsk3b', 'jnk3']
def map_prop_to_idx(props):
    global PROPS
    idxs = []
    p2i = {}
    for i, p in enumerate(PROPS):
        p2i[p] = i
    for p in props:
        if p in p2i:
            idxs.append(p2i[p])
        else:
            raise ValueError('Invalid property')
    return sorted(list(set(idxs)))


def overpass_th(prop_vals, prop_idx):
    ori_prop_vals = [0 for _ in PROPS]
    for i, val in zip(prop_idx, prop_vals):
        ori_prop_vals[i] = val
    ori_prop_vals = restore_property_scores(ori_prop_vals)
    for i in prop_idx:
        if ori_prop_vals[i] < PROP_TH[i]:
            return False
    return True


class TopStack:
    '''Only save the top-k results and the corresponding '''
    def __init__(self, k, cmp):
        # k: capacity, cmp: binary comparator indicating if x is prior to y
        self.k = k
        self.stack = []
        self.cmp = cmp

    def push(self, val, data=None):
        i = len(self.stack) - 1
        while i >= 0:
            if self.cmp(self.stack[i][0], val):
                break
            else:
                i -= 1
        i += 1
        self.stack.insert(i, (val, data))
        if len(self.stack) > self.k:
            self.stack.pop()
    
    def get_iter(self):
        return iter(self.stack)


# def parse():
#     parser = argparse.ArgumentParser(description='init mean and var for props')
#     parser.add_argument('--data', type=str, required=True, help='Path to dataset')
#     parser.add_argument('--cpus', type=int, default=8, help='How many cpus to use')
#     return parser.parse_args()


if __name__ == '__main__':
    # args = parse()
    # init_stats(args.data, args.cpus)
    eg = 'CN(C)CC[C@@H](c1ccc(Br)cc1)c1ccccn1'
    m = smiles2molecule(eg)
    eval_funcs = eval_funcs_dict()
    for key in eval_funcs:
        f = eval_funcs[key]
        print(f'{key}: {f(m)}')
    print(f'normalized: {get_normalized_property_scores(m)}')