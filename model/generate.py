#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import argparse
import random
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import multiprocessing as mp
from functools import partial
import torch
from torch_geometric.data import Batch, DataLoader

import pl_models
from utils.nn_utils import to_one_hot
from utils.chem_utils import data2molecule, mol2file, molecule2smiles, smiles2molecule
from utils.chem_utils import get_base64, get_random_submol
from utils.logger import print_log
from evaluation.utils import get_normalized_property_scores, overpass_th, similarity
from evaluation.utils import map_prop_to_idx, TopStack, PROPS
from evaluation.utils import get_penalized_logp


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def get_loss(model, props, target):
    '''
    args:
        model: model object
        props: raw property tensor in shape [num_props]
        target: target values in shape [num_props]. value may be None, indicating skipping the property
    return:
        loss of props and target
    '''
    tar_vals = [val for val in target if val is not None]
    tar_vals = torch.tensor(tar_vals, dtype=torch.float, device=props.device)
    select_idxs = [i for i, val in enumerate(target) if val is not None]
    select_idxs = torch.tensor(select_idxs, dtype=torch.long, device=props.device)
    pred_vals = torch.index_select(props, dim=0, index=select_idxs)
    return model.pred_loss(pred_vals, tar_vals)


def gen(model, z, max_atom_num, add_edge_th, temperature, constraint_mol=None):
    if constraint_mol is None:
        return model.inference_single_z(z, max_atom_num, add_edge_th, temperature)
    constraint_mol = get_random_submol(constraint_mol)
    return model.inference_single_z_constraint(z, max_atom_num, add_edge_th, temperature, constraint_mol)


def beam_gen(model, z, beam, target, max_atom_num, add_edge_th, temperature, constraint_mol=None):
    gens = [gen(model, z, max_atom_num, add_edge_th, temperature, constraint_mol) for _ in range(beam)]
    mols = [model.return_data_to_mol(g) for g in gens]
    props = [get_normalized_property_scores(m) for m in mols]
    with torch.no_grad():
        losses = [get_loss(model, torch.tensor(p), target) for p in props]
    sorted_idx = sorted([i for i in range(len(losses))], key=lambda x: losses[x])
    best_idx = sorted_idx[0]
    if constraint_mol is not None:
        return mols
    return mols[best_idx]


def direct_optimize(model, z, target, lr, max_iter, patience,
                    max_atom_num, add_edge_th, temperature):
    '''direct gradient optimization, return optimized z'''
    optimized = z.clone()
    optimized.requires_grad = True
    optimizer = torch.optim.Adam([optimized], lr=lr)
    best_loss, not_inc, best_z = 100, 0, optimized.clone()
    for i in range(max_iter):
        props = model.predict_props(optimized)
        loss = get_loss(model, props, target)
        optimizer.zero_grad()
        model.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()
        print_log(f'step {i}, loss: {loss}, props: {[round(x, 2) for x in props.tolist()]}', level='DEBUG')
        if loss < best_loss:
            best_loss, not_inc, best_z = loss, 0, optimized.clone()
        else:
            not_inc += 1
            if not_inc > patience:
                break
    return best_z


def iter_optimize(model, z, beam, target, lr, max_iter, patience, max_atom_num, add_edge_th, temperature):
    optimized = z.clone()
    best_loss, not_inc, best_z, best_mol = 100, 0, optimized.clone(), None
    for i in range(max_iter):
        # decode then encode
        with torch.no_grad():
            graph = beam_gen(model, optimized, beam, target, max_atom_num, add_edge_th, temperature)
            optimized = model.get_z_from_return_data(graph)
        optimized.requires_grad = True
        model.zero_grad()
        props = model.predict_props(optimized)
        loss = get_loss(model, props, target)
        loss.backward()
        loss = loss.item()
        print_log(f'step {i}, loss: {loss}, props: {[round(x, 2) for x in props.tolist()]}', level='DEBUG')
        if loss < best_loss:
            best_loss, not_inc, best_z, best_mol = loss, 0, optimized.clone(), graph
        else:
            not_inc += 1
            if not_inc > patience:
                break
        optimized = optimized - lr * optimized.grad
    return best_z, best_mol


def str2model(s):
    if s == 'vae_dgmg':
        return pl_models.vae_dgmg_model.VAEDGMGModel
    elif s == 'dgmg':
        return pl_models.dgmg_model.DGMGModel
    elif s == 'vae_piece':
        return pl_models.vae_piece_model.VAEPieceModel
    elif s == 'vae_piece_dgmg':
        return pl_models.vae_piece_dgmg_model.VAEPieceDGMGModel
    else:
        raise NotImplementedError(f'{s} model not implemented!')


def load_model(model_name, ckpt, gpus):
    if gpus == -1:
        loc = torch.device('cpu')
    else:
        loc = torch.device(f'cuda:{gpus}')
    model = str2model(model_name).load_from_checkpoint(ckpt)
    model.to(loc)
    return model


def config(args):
    return {
        'lr': args.lr,
        'max_iter': args.max_iter,
        'patience': args.patience,
        'max_atom_num': args.max_atom_num,
        'add_edge_th': args.add_edge_th,
        'temperature': args.temperature
    }


def parallel(func, data, cpu, arg_pos=-1, args=None):
    if cpu < 1:
        cpu = mp.cpu_count()
    pool = mp.Pool(cpu)
    if args is not None:
        new_data = []
        for d in data:
            arg = list(args)
            arg.insert(arg_pos, d)
            new_data.append(arg)
        data = new_data
        res = pool.starmap(func, data)
    else:
        res = pool.map(func, data)
    pool.close()
    return res


def exp_prop(args):
    '''First generate n_sample molecules constrainted by selected properties.
       Then save the generated smiles to output_path.
       Output success rate to log.
       props: list of str which includes the names of selected properties
       n_sample: number of molecules to generate
       output_path: path to save the output molecules'''
    model = str2model(args.model).load_from_checkpoint(args.ckpt)
    model.eval()
    # generate target list
    tar_props = args.props.split(',')
    top_stacks_pred = [TopStack(3, lambda x, y: x > y) for _ in tar_props]
    top_stacks_real = [TopStack(3, lambda x, y: x > y) for _ in tar_props]
    prop_idx = map_prop_to_idx(tar_props)
    target = [None for _ in PROPS]
    for i in prop_idx:
        target[i] = args.target
    # generate molecule (serial)
    print_log('Optimizing')
    latents = model.sample_z(args.n_samples)  # tensor or ndarray, [n, latent_size]
    if args.optimize_method == 'direct':
        optimized_latents = [direct_optimize(model, z, target, **config(args)) for z in tqdm(latents)]
        print_log('decoding')
        optimized_graphs = [beam_gen(model, z, args.beam, target, args.max_atom_num,
                                     args.add_edge_th, args.temperature) for z in tqdm(optimized_latents)]
    elif args.optimize_method == 'iterative':
        optimized_latents, optimized_graphs = [], []
        for z in tqdm(latents):
            op_z, op_mol = iter_optimize(model, z, args.beam, target, **config(args))
            optimized_latents.append(op_z)
            optimized_graphs.append(op_mol)
    else:
        raise NotImplementedError(f'optimize method {args.optimize_method} not implemented')
    
    print_log('Generating molecules')
    smis = parallel(molecule2smiles, optimized_graphs, args.cpus)
    mols = parallel(partial(smiles2molecule, kekulize=False), smis, args.cpus)

    # pred properties and success rate based on pred properties
    print_log('Evaluating on predicted properties')
    cnt = 0
    batch_size = 128
    for i in tqdm(range(0, len(optimized_latents), batch_size)):
        end = min(len(optimized_latents), i + batch_size)
        with torch.no_grad():
            props = model.predict_props(torch.stack(optimized_latents[i:end]))
            for p, s in zip(props, smis[i:end]):
                cnt += int(overpass_th(p, prop_idx))
                for pid, stack in enumerate(top_stacks_pred):
                    stack.push(p[prop_idx[pid]], s)
    suc_pred = cnt / len(optimized_latents)

    # real properties and success rate
    print_log('Evaluating on real properties')
    fout = open(args.output_path, 'w')
    cnt = 0
    for i, mol in tqdm(enumerate(mols)):
        props = get_normalized_property_scores(mol)
        cnt += int(overpass_th(props, prop_idx))
        for pid, stack in enumerate(top_stacks_real):
            stack.push(props[prop_idx[pid]], smis[i])
        fout.write(f'{smis[i]}\n')
    suc_real = cnt / len(mols)
    fout.close()
    print(f'IMPORTANT::success rate on prediction: {suc_pred}, '
          f'success rate on reality: {suc_real}')
    print(f'IMPORTANT::Top 3 scores for {tar_props}')
    for i, p in enumerate(tar_props):
        print(f'{p}:')
        print('\tOn prediction:')
        for place, item in enumerate(top_stacks_pred[i].get_iter()):
            val, smi = item
            print(f'\t\t{place}: {val} {smi}')
        print('\tOn reality:')
        for place, item in enumerate(top_stacks_real[i].get_iter()):
            val, smi = item
            print(f'\t\t{place}: {val} {smi}')


def rec(args, enc_num=1, dec_num=1, sample_num=100):
    '''Reconstruct molecules from given test set. TODO: not yet finished'''
    if args.gpus == -1:
        loc = 'cpu'
    else:
        loc = f'cuda:{args.gpus}'
    model = str2model(args.model).load_from_checkpoint(args.ckpt, map_location=loc)
    model.eval()
    # load dataset
    data_loader = model.get_dataloader(args.train_set, batch_size=100, shuffle=True, num_workers=4)
    with open(args.test_set, 'r') as fin:
        lines = fin.read().strip().split('\n')
        ori_mols = list(map(smiles2molecule, lines))
    sims, succ = [], 0
    with torch.no_grad():
        for _ in range(enc_num):
            latents = []
            cnt = 0
            for batch in data_loader:
                latents.append(model.get_z(batch))
                cnt += model.get_from_batch(batch, 'batch_size')
                if cnt >= sample_num:
                    break
            latents = torch.cat(latents, dim=0).squeeze()
            for _ in range(dec_num):
                res_data = [gen(model, z, args.max_atom_num, args.add_edge_th, args.temperature)\
                            for z in latents]
                mols = []
                for d in res_data:
                    mols.append(model.return_data_to_mol(d))
                for m1, m2 in zip(ori_mols, mols):
                    sims.append(similarity(m1, m2))
                    if sims[-1] > 0.99:
                        succ += 1
    rate = succ / (len(ori_mols) * enc_num * dec_num)
    print(f'mean sim: {np.mean(sims)}, max sim: {max(sims)}, min sim: {min(sims)}, success rate: {rate}, total sample num: {cnt}')
    # good_mol_idxs = sorted([i for i in range(len(sims))], key=lambda x: sims[x])[-20:]
    # good_mols = [mols[i] for i in good_mol_idxs]
    # mol2file(good_mols, 'rec.png', True, 4)


def pred_eval(args):
    model = str2model(args.model).load_from_checkpoint(args.ckpt)
    model.eval()
    # load dataset
    # dataset = get_dataset(args.test_set, model.vocab)
    # data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    data_loader = model.get_dataloader(args.test_set, batch_size=128, shuffle=False, num_workers=4)
    with torch.no_grad():
        pred_y, true_y = [], []
        for batch in data_loader:
            batch_size = model.get_from_batch(batch, 'batch_size')
            z = model.get_z(batch)  # [batch_size, latent_dim]
            pred_prop = model.predict_props(z)  # [batch_size, num_properties]
            golden = model.get_from_batch(batch, 'props').reshape(batch_size, -1)[:,model.config['selected_properties']]
            pred_y.extend(pred_prop.numpy())
            true_y.extend(golden.numpy())
    pred_y, true_y = np.array(pred_y), np.array(true_y)
    judge_regression(pred_y, true_y, er_th=0.2)


def pred_dec_eval(args):
    model = str2model(args.model).load_from_checkpoint(args.ckpt)
    model.eval()
    print_log(f'Sampling {args.n_samples} for testing')
    latents = model.sample_z(args.n_samples)  # tensor or ndarray, [n, latent_size]
    with torch.no_grad():
        pred_y = model.predict_props(latents)  # [batch_size, num_properties]
        res_data = [gen(model, z, args.max_atom_num, args.add_edge_th, args.temperature)\
                    for z in latents]
    mols = [model.return_data_to_mol(d) for d in res_data]
    true_y = [get_normalized_property_scores(mol) for mol in mols]
    pred_y, true_y = pred_y.numpy(), np.array(true_y)
    judge_regression(pred_y, true_y, er_th=0.2)


def judge_regression(pred_y, true_y, er_th):  # er_th: relative error rate
    mse = mean_squared_error(pred_y, true_y)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_y, pred_y)
    print_log(f'overall mse: {mse}, rmse: {rmse}, r2: {r2}')
    for i in range(pred_y.shape[-1]):
        p, t = pred_y[:, i], true_y[:, i]
        delta = np.abs(p - t)
        suc = delta < np.abs(t) * er_th
        suc = suc.flatten()
        print_log(f'prop {i}, er_th {er_th}, success rate {suc.sum() / len(suc)}')


def common_metrics(args):
    '''calculate uniqueness and novelty'''
    model = str2model(args.model).load_from_checkpoint(args.ckpt)
    model.eval()
    print_log(f'Sampling {args.n_samples} for common metrics testing')
    latents = model.sample_z(args.n_samples)  # tensor or ndarray, [n, latent_size]
    with torch.no_grad():
        res_data = [gen(model, z, args.max_atom_num, args.add_edge_th, args.temperature)\
                    for z in latents]
    mols = [model.return_data_to_mol(d) for d in res_data]
    with open(args.train_set, 'r') as fin:
        train_smi = fin.read().strip().split('\n')
    train_mols = parallel(smiles2molecule, train_smi, args.cpus)
    gen_base64 = parallel(get_base64, mols, args.cpus)
    train_base64 = parallel(get_base64, train_mols, args.cpus)
    del train_mols
    del mols
    # uniqueness
    same = 0
    tmp_dict = {}
    for b in gen_base64:
        if b not in tmp_dict:
            tmp_dict[b] = True
        else:
            same += 1
    print_log(f'Uniqueness: {1 - same / len(gen_base64)}')
    # novelty
    train_base64_dict = { b: True for b in train_base64 }
    new_mol_cnt = 0
    for b in gen_base64:
        if b not in train_base64_dict:
            new_mol_cnt += 1
    print_log(f'Novelty: {new_mol_cnt / len(gen_base64)}')


def constraint_direct_optimize(ori_mol, model, z, target, args):
    '''direct gradient optimization, return optimized z'''
    optimized = z.clone()
    optimized.requires_grad = True
    optimizer = torch.optim.Adam([optimized], lr=args.lr)
    best_loss, not_inc, best_z = 100, 0, optimized.clone()
    latents = []
    for i in range(args.max_iter):
        props = model.predict_props(optimized)
        loss = get_loss(model, props, target)
        optimizer.zero_grad()
        model.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()
        print_log(f'step {i}, loss: {loss}, props: {[round(x, 2) for x in props.tolist()]}', level='DEBUG')
        if loss < best_loss:
            best_loss, not_inc, best_z = loss, 0, optimized.clone()
            latents.append(best_z)
        else:
            not_inc += 1
            if not_inc > args.patience:
                break
    # all mols
    mols = []
    with torch.no_grad():
        for z in latents:
            new_mols = beam_gen(model, z, args.beam, target, args.max_atom_num,
                                args.add_edge_th, args.temperature, ori_mol)
            mols.extend(new_mols)
    return mols


def constraint_optim(args):
    '''only for experiments of penalized logp constraints optimization'''
    # find n molecules with minimum logp from train set
    model = load_model(args.model, args.ckpt, args.gpus)
    model.eval()
    print_log(f'device: {model.device}')
    print_log(f'Sampling {args.n_samples} for constraints optimization')
    if not os.path.exists(args.zinc800_logp):
        print_log(f'Finding {args.n_samples} molecules with the lowest logp in train set')
        with open(args.train_set, 'r') as fin:
            train_smi = fin.read().strip().split('\n')
        train_mols = parallel(smiles2molecule, train_smi, args.cpus)
        train_logps = parallel(get_penalized_logp, train_mols, args.cpus)
        selected_ids = sorted([i for i in range(len(train_mols))], key=lambda i: train_logps[i])[:args.n_samples]
        sample_mols = [train_mols[idx] for idx in selected_ids]
        sample_logps = [train_logps[idx] for idx in selected_ids]
        with open(args.zinc800_logp, 'w') as fout:
            for idx in selected_ids:
                fout.write(f'{train_smi[idx]} {train_logps[idx]}\n')
        del train_mols
        del train_logps
    else:
        print_log('Loading from cached file')
        with open(args.zinc800_logp, 'r') as fin:
            lines = fin.read().strip().split('\n')
        sample_mols, sample_logps, sample_smis = [], [], []
        for line in lines:
            smi, logp = line.split()
            sample_smis.append(smi)
            sample_logps.append(float(logp))
        sample_mols = parallel(smiles2molecule, sample_smis, args.cpus)
    prop_idx = map_prop_to_idx(['logp'])
    target = [None for _ in PROPS]
    for i in prop_idx:
        target[i] = args.target
    with torch.no_grad():
        print_log('Getting latent representation')
        latents = [model.get_z_from_mol(mol) for mol in tqdm(sample_mols)]
    sim_cutoffs = [0, 0.2, 0.4, 0.6]
    fail_cnts = [0 for _ in sim_cutoffs]
    opt_improves = [[] for _ in sim_cutoffs]
    opt_sims = [[] for _ in sim_cutoffs]
    fnames = [os.path.join(args.output_path, f'{sim_cutoff}.smi') for sim_cutoff in sim_cutoffs]
    pool = mp.Pool(args.cpus)
    print_log('Optimizing...')
    for i, ori_mol in enumerate(tqdm(sample_mols)):
        ori_logp, ori_smi, z = sample_logps[i], molecule2smiles(ori_mol), latents[i]
        mols = constraint_direct_optimize(ori_mol, model, z, target, args)
        smis = pool.map(molecule2smiles, mols)
        # use unkekulized form for evaluation, follow GCPN and GraphAF
        mols = pool.map(partial(smiles2molecule, kekulize=False), smis)
        ori_mol = smiles2molecule(sample_smis[i], kekulize=False)
        sims = pool.map(partial(similarity, mol2=ori_mol), mols)
        new_logps = pool.map(get_penalized_logp, mols)
        for sid, sim_cutoff in enumerate(sim_cutoffs):
            best_improve, best_sim, best_smi = 0, 0, None
            for j, new_mol in enumerate(mols):
                smi, sim, new_logp = smis[j], sims[j], new_logps[j]
                if sim < sim_cutoff or sim == 1.0:
                    continue
                else:
                    new_improve = new_logp - ori_logp
                    if new_improve > best_improve:
                        best_improve, best_sim, best_smi = new_improve, sim, smi
            if best_smi is not None:
                opt_improves[sid].append(best_improve)
                opt_sims[sid].append(best_sim)
                line = f'{ori_smi}||{best_smi}|{best_improve}|{best_sim}'
            else:
                line = f'{ori_smi}||NotFound|-100|-100'
                fail_cnts[sid] += 1
            with open(fnames[sid], 'a') as fout:
                fout.write(line + '\n')
    pool.close()
    for sid, sim_cutoff in enumerate(sim_cutoffs):
        fail_cnt, opt_improve, opt_sim = fail_cnts[sid], opt_improves[sid], opt_sims[sid]
        success_rate = 1 - fail_cnt / len(sample_mols)
        mean, stdvar = np.mean(opt_improve), np.std(opt_improve)
        print_log(f'success rate: {success_rate}, mean improvement: {mean}, stdvar: {stdvar}')
        print_log(f'similarity mean: {np.mean(opt_sim)}, stdvar: {np.std(opt_sim)}')


def parse():
    need_train_set = '--rec' in sys.argv or '--common_metrics' in sys.argv or '--constraint_optim' in sys.argv
    parser = argparse.ArgumentParser(description='generate molecule given property')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('--model', type=str, choices=['vae_dgmg', 'dgmg', 'vae_piece', 'vae_piece_dgmg'], required=True,
                        help='Type of model')

    parser.add_argument('--rec', action='store_true', help='Reconstruct')
    parser.add_argument('--pred_eval', action='store_true', help='Eval predictor')
    parser.add_argument('--pred_dec_eval', action='store_true', help='Eval pred and dec consistency')
    parser.add_argument('--common_metrics', action='store_true', help='Uniqueness and Novelty')
    parser.add_argument('--constraint_optim', action='store_true', help='Constraint optimization')
    parser.add_argument('--train_set', type=str, required=need_train_set,
                        help='Training set path (for reconstruction and common metrics)')
    parser.add_argument('--test_set', type=str, required='--pred_eval' in sys.argv,
                        help='Path to test set')

    parser.add_argument('--eval', action='store_true', help='Evaluation mode')
    parser.add_argument('--props', type=str, required='--eval' in sys.argv,
                        help='Selected properties')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of molecules to generate')
    parser.add_argument('--output_path', type=str, required='--eval' in sys.argv,
                        help='Path to store smiles of generated molecules')
    parser.add_argument('--sample_method', type=str, default='gaussian',
                        help='Sampling method. Choices are [gaussian, uniform].')
    parser.add_argument('--optimize_method', type=str, choices=['direct', 'iterative'],
                        default='iterative', help='Methods to do optimization')
    parser.add_argument('--cpus', type=int, default=-1,  # -1 for maxium of available cpu cores
                        help='Number of cpu cores to parallel generation')
    parser.add_argument('--gpus', type=int, default=-1,
                        help='Gpu number to use')
    # default parameters
    parser.add_argument('--max_atom_num', type=int, default=60,
                        help='Max number of atoms to generate')
    parser.add_argument('--add_edge_th', type=float, default=0.5,
                        help='Threshold of adding an edge')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Decoding temperature')
    parser.add_argument('--beam', type=int, default=5,
                        help='Number of mols to generate for one latent z')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Optimizing learning rate')
    parser.add_argument('--max_iter', type=int, default=20,
                        help='Max iteration steps')
    parser.add_argument('--patience', type=int, default=3,
                        help='Patience before quit')
    parser.add_argument('--target', type=float, default=2,
                        help='Optimization target')
    # parser.add_argument('--sim_cutoff', type=float, required='--constraint_optim' in sys.argv,
    #                     help='Threshold of constraint optimization')
    parser.add_argument('--zinc800_logp', type=str, required='constraint' in sys.argv,
                       help='Path to zinc 800')
    parser.add_argument('--seed', type=int, default=2021)
    return parser.parse_args()


def main(args):
    setup_seed(args.seed)
    if args.eval:
        exp_prop(args)
        return
    if args.rec:
        rec(args)
        return
    if args.pred_eval:
        pred_eval(args)
        return
    if args.pred_dec_eval:
        pred_dec_eval(args)
        return
    if args.common_metrics:
        common_metrics(args)
        return
    if args.constraint_optim:
        constraint_optim(args)
        return


    model = str2model(args.model).load_from_checkpoint(args.ckpt)
    model.eval()
    # generate target list
    tar_props = args.props.split(',')
    prop_idx = map_prop_to_idx(tar_props)
    target = [None for _ in PROPS]
    for i in prop_idx:
        target[i] = args.target
    # generate molecule (serial)
    print_log('Optimizing')
    import datetime
    setup_seed(int(datetime.datetime.utcnow().timestamp()))
    latents = model.sample_z(4)  # tensor or ndarray, [n, latent_size]
    if args.optimize_method == 'direct':
        optimized_latents = [direct_optimize(model, z, target, **config(args)) for z in tqdm(latents)]
        print_log('decoding')
        optimized_graphs = [beam_gen(model, z, args.beam, target, args.max_atom_num,
                                     args.add_edge_th, args.temperature) for z in tqdm(optimized_latents)]
    elif args.optimize_method == 'iterative':
        optimized_latents, optimized_graphs = [], []
        for z in tqdm(latents):
            op_z, op_mol = iter_optimize(model, z, args.beam, target, **config(args))
            optimized_latents.append(op_z)
            optimized_graphs.append(op_mol)
    mols = []
    for d in optimized_graphs:
        mols.append(model.return_data_to_mol(d))
    mol2file(mols, 'rec.png', grid=True, molsPerRow=4)
    with torch.no_grad():
        pred_props = [model.predict_props(z) for z in optimized_latents]
    for mol, prop in zip(mols, pred_props):
        smiles = molecule2smiles(mol)
        mol = smiles2molecule(smiles)
        try:
            pred = list(map(lambda x: round(x, 2), prop.numpy().tolist()))
            real = list(map(lambda x: round(x, 2), get_normalized_property_scores(mol)))
            print(f'\nPred Prop: {pred}')
            print(f'Real Prop: {real}')
        except ValueError:
            print(type(mol))

if __name__ == '__main__':
    main(parse())
