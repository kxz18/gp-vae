#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Load data from kinase.tsv and train/test random forest classifier for gsk3b and jnk3"""
import sys
import random
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import rdBase
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score


rdBase.DisableLog('rdApp.error')


def smiles2molecule(smiles: str):
    '''turn smiles to molecule'''
    return Chem.MolFromSmiles(smiles)


def ecfpn(molecule, n=4):
    '''return ecfp6 fingerprint of given molecule, default is ecfp6.
       n should be even number'''
    features_vec = AllChem.GetMorganFingerprintAsBitVect(molecule, n // 2, nBits=2048)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)
    return features.reshape(1, -1)[0]


def load_data(path):
    '''load gsk3b and jnk3 data from the file'''
    data = pd.read_csv(path, delimiter='\t')
    gsk3b = data.loc[data['target'] == 'gsk3b']
    jnk3 = data.loc[data['target'] == 'jnk3']
    return {
        'gsk3b': gsk3b,
        'jnk3': jnk3
    }


def split_train_test(raw_data):
    '''split train/test set by given label'''
    train = raw_data.loc[raw_data['is_train'] == 1]
    test = raw_data.loc[raw_data['is_train'] == 0]
    return train, test


def get_xy_from_raw_data(raw_data, shuffle=True):
    if shuffle:
        raw_data = raw_data.sample(frac=1)
    x = []
    y = []
    for i in range(len(raw_data)):
        molecule = smiles2molecule(raw_data.iloc[i]['smiles'])
        if molecule is None:
            continue
        x.append(ecfpn(molecule, 4))  # ECFP4
        y.append(raw_data.iloc[i]['is_active'])
    return x, y


def train_rf(rf, train_set):
    '''train random forest classifier'''
    x, y = get_xy_from_raw_data(train_set)
    rf.fit(x, y)


def test_rf(rf, test_set):
    '''test random forest, return accuracy and f1'''
    x, y = get_xy_from_raw_data(test_set)
    pred = rf.predict_proba(x)
    y = np.array(y)
    pred_continuous = np.array([p[1] for p in pred])
    pred = np.argmax(pred, 1)
    accu = (pred == y).sum() / len(y)
    f1 = f1_score(y, pred, zero_division=1)
    ra = roc_auc_score(y, pred_continuous)
    return accu, f1, ra


def print_dataset_details(dataset, name):
    '''print length, positive / negative numbers of datasets'''
    l = len(dataset)
    pos = 0
    neg = 0
    for i in range(l):
        if dataset.iloc[i]['is_active'] == 1:
            pos += 1
        else:
            neg += 1
    print(f'details of {name}:')
    print(f'\t{l} entries, {pos} positive and {neg} negative entries.')
    print(f'\tpos ratio: {round(pos / l, 3)}, neg ratio: {round(neg / l, 3)}')


def train_classifier(data_path, show_only=False):
    '''train classifier for gsk3b and jnk3'''
    print('training classifier for gsk3b and jnk3')
    print(f'loading data from {data_path}')
    data = load_data(data_path)
    print(f'data loaded')
    for key in data:
        print(key)
        rf = RandomForestClassifier(n_estimators=100)
        train, test = split_train_test(data[key])
        print_dataset_details(train, 'training set')
        print_dataset_details(test, 'test set')
        if show_only:
            continue
        print('start training')
        train_rf(rf, train)
        print('finished training')
        print('start testing')
        accu, f1, ra = test_rf(rf, test)
        print(f'accu: {accu}, f1: {f1}, ra: {ra}')
        with open(f'{key}.pkl', 'wb') as fout:
            pickle.dump(rf, fout)


if __name__ == '__main__':
    data_path = sys.argv[1]
    train_classifier(data_path, 'show' in sys.argv)
