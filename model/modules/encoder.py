#!/usr/bin/python
# -*- coding:utf-8 -*-
from collections import defaultdict
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import GINEConv
from torch_geometric.data import Batch, Data

from modules.common_nn import MLP


class Embedding(nn.Module):
    """embed one-hot features (node type)"""
    def __init__(self, sizes, dim_embeddings):
        super(Embedding, self).__init__()
        self.embedding = []
        for size, dim in zip(sizes, dim_embeddings):
            if size > 10:
                self.embedding.append(nn.Embedding(size, dim))
            else:
                self.embedding.append(None)
    
    def forward(self, x):
        # x: [data_num, num_features]
        num_features = x.shape[-1]
        embed = []
        for i in range(num_features):
            embed_nn = self.embedding[i]
            if embed_nn is None:
                embed.append(x[:, i].unsqueeze(-1))
            else:
                embed.append(embed_nn(x[:, i]))
        return torch.cat(embed, dim=-1)


class Encoder(nn.Module):
    def __init__(self, dim_in, num_edge_type, dim_hidden, dim_out, t=4):
        super(Encoder, self).__init__()
        # self.embedding = Embedding(one_hot_sizes, dim_embeddings)
        self.num_edge_type = num_edge_type
        self.t = t  # number of iterations
        self.node_trans = nn.Linear(dim_in, dim_hidden)
        # self.edge_trans = nn.Embedding(num_edge_type, dim_hidden)
        self.edge_trans = nn.Linear(num_edge_type, dim_hidden)
        self.conv = GINEConv(MLP(dim_hidden, dim_hidden, dim_hidden, nn.ReLU, 2))
        # self.conv1 = GINEConv(MLP(dim_hidden, dim_hidden, dim_hidden, nn.ReLU, 2))
        # self.conv2 = GINEConv(MLP(dim_hidden, dim_hidden, dim_hidden, nn.ReLU, 2))
        self.linear = nn.Linear(dim_hidden * self.t, dim_out)

    def embed_node(self, x, edge_index, edge_attr):
        x = self.node_trans(x.float())                  # [total_num_nodes, dim_hidden]
        edge_attr = self.edge_trans(edge_attr.float()).squeeze(1)  # [total_num_edges, dim_hidden]
        all_x = []
        for _ in range(self.t):
            x = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
            all_x.append(x)
        all_x = torch.cat(all_x, dim=-1)  # [total_num_nodes, dim_hidden * t]
        return x, all_x

    def embed_graph(self, all_x, graph_ids, node_mask=None):
        res = torch.zeros((graph_ids[-1] + 1, all_x.shape[-1]), device=all_x.device)  # [num_graphs, dim_out]
        if node_mask is not None:
            graph_ids, all_x = graph_ids[~node_mask], all_x[~node_mask]
        res.index_add_(0, graph_ids, all_x)
        res = self.linear(res)  # to dim out
        return res

    def forward(self, batch, return_x=False):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        x, all_x = self.embed_node(x, edge_index, edge_attr)
        res = torch.zeros((batch.num_graphs, all_x.shape[-1]), device=all_x.device)  # [num_graphs, dim_out]
        res.index_add_(0, batch.batch, all_x)
        res = self.linear(res)  # to dim out
        
        # x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        # x = F.relu(x)
        # x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.linear(x)  # [total_num_nodes, dim_out]
        # res = torch.zeros((batch.num_graphs, x.shape[-1]), device=x.device)  # [num_graphs, dim_out]
        # res.index_add_(0, batch.batch, x)
        if return_x:
            return res, x
        return res


# class MolData:
#     def __init__(self, device):
#         self.device = device
#         self.x = torch.tensor([])
#         self.edge_index = torch.tensor([], dtype=torch.long)
#         self.edge_attr = torch.tensor([])
#         self.x_idxs = []
#     
#     def add_x(self, x):
#         x = x.unsqueeze(0)
#         if len(self.x) == 0:
#             self.x = x.clone()
#         else:
#             self.x = torch.cat([self.x, x], dim=0)
# 
#     def add_edge(self, ei, attr):
#         ei = torch.tensor(ei, device=self.device).unsqueeze(-1)  # [2, 1]
#         attr = torch.tensor([attr], device=self.device).unsqueeze(0) # [1, 1]
#         if len(self.edge_attr) == 0:
#             self.edge_index = ei.clone()
#             self.edge_attr = attr.clone()
#         else:
#             self.edge_index = torch.cat([self.edge_index, ei], dim=1)  # [2, num_edges]
#             self.edge_attr = torch.cat([self.edge_attr, attr], dim=0)  # [num_edges, 1]
#     
#     def add_x_one_hot(self, x):
#         self.x_idxs.append([x])
# 
#     def get_dataobj(self):
#         return Data(x=self.x, edge_index=self.edge_index, edge_attr=self.edge_attr)
# 
#     def get_dataobj_gen(self):
#         return Data(x=self.x_idxs, edge_index=self.edge_index, edge_attr=self.edge_attr)
# 
#     def get_num_nodes(self):
#         return len(self.x)
    

# class Decoder(nn.Module):
#     '''given an embedding, reconstruct the graph'''
#     def __init__(self, num_atom_type, num_edge_type, dim_hidden, dim_ctx):
#         super(Decoder, self).__init__()
#         self.stop_idx = num_atom_type
#         self.num_edge_type = num_edge_type
#         self.encoder = Encoder([dim_ctx], num_edge_type, dim_hidden, dim_ctx, skip_one_hot=True)
#         dim_in = dim_ctx * 2  # conditional and current graph ctx
#         self.node_embedding = nn.Embedding(num_atom_type + 1, dim_ctx)
#         self.node_trans = nn.Linear(dim_in + dim_ctx, dim_ctx)
#         self.add_node = MLP(dim_in=dim_in, dim_hidden=dim_ctx,
#                             dim_out=num_atom_type + 1, act_func=nn.ReLU,
#                             num_layers=2)  # predict a node type or a stop
#         self.add_edge = MLP(dim_in=dim_ctx + dim_in, dim_hidden=dim_ctx,
#                             dim_out=1, act_func=nn.ReLU,
#                             num_layers=2)  # predict if there will be any atoms connected to v
#         self.node_edge = MLP(dim_in=dim_ctx * 2, dim_hidden=dim_ctx,  # two node embedding
#                             dim_out=num_edge_type, act_func=nn.ReLU,
#                             num_layers=2)  # num_nodes * num_edge_type
#         self.add_node_loss = nn.CrossEntropyLoss()
#         self.add_edge_loss = nn.BCEWithLogitsLoss()
#         self.node_edge_loss = nn.CrossEntropyLoss(ignore_index=-1)  # stop edge
#     
#     def init_node(self, node_idxs, ctxs, conds):
#         if isinstance(node_idxs, torch.Tensor):
#             i_tensor = node_idxs
#         else:
#             i_tensor = torch.tensor(node_idxs, device=ctxs.device)
#         embeds = self.node_embedding(i_tensor)
#         embeds = self.node_trans(torch.cat([ctxs, conds, embeds], dim=-1))
#         return embeds
# 
#     def forward(self, batch, conds):
#         data_list = batch.to_data_list()
#         # add_node_preds, add_edge_preds, node_edge_preds = [], [], []
#         add_node_preds, add_edge_preds = [], []
#         ne_preds_dict = defaultdict(list)
# 
#         batch_size = len(data_list)
#         max_t = max([len(d.add_node) for d in data_list])
#         ctxs = torch.zeros_like(conds, device=conds.device, requires_grad=True)
#         gen_data = [MolData(conds.device) for _ in range(batch_size)]
#         # supervised data
#         # add_node_golden, add_edge_golden, node_edge_golden = [], [], []
#         add_node_golden, add_edge_golden = [], []
#         ne_golden_dict = defaultdict(list)
#         for t in range(max_t):
#             # find all graphs that still need to generate
#             batch_list = [i for i in range(batch_size) if t < len(data_list[i].add_node)]
#             new_ctxs = ctxs[batch_list]
#             new_conds = conds[batch_list]
# 
#             add_node = self.add_node(torch.cat([new_ctxs, new_conds], dim=-1))
#             add_node_preds.append(add_node)
#             atom_idxs = []
#             new_batch_list = []
#             for i in batch_list:
#                 node_idx = data_list[i].add_node[t]
#                 add_node_golden.append(node_idx)
#                 atom_idxs.append(node_idx)
#                 # check stop
#                 if node_idx != self.stop_idx:
#                     new_batch_list.append(i)
#             batch_list = new_batch_list
#             if len(batch_list) == 0:
#                 break
#             # add golden atom embedding
#             node_embeds = self.init_node(atom_idxs, new_ctxs, new_conds)
#             for idx, i in enumerate(batch_list):
#                 gen_data[i].add_x(node_embeds[idx])
# 
#             # pred edges
#             max_et = max([len(data_list[i].edge_sets[t]) for i in batch_list])
#             for et in range(max_et):
#                 batch_list = [i for i in batch_list if et < len(data_list[i].edge_sets[t])]
#                 new_conds = conds[batch_list]
#                 # preprocess
#                 tmp_data = []
#                 end_idxs = []
#                 for i in batch_list:
#                     tmp_data.append(gen_data[i].get_dataobj()) # generate batch data
#                     if len(end_idxs) > 0:  # find ending node index of each graph
#                         end_idxs.append(end_idxs[-1] + gen_data[i].get_num_nodes())
#                     else:
#                         end_idxs.append(gen_data[i].get_num_nodes() - 1)
#                     # add golden supervised edge data
#                     if data_list[i].edge_sets[t][et] is None:
#                         add_edge_golden.append(0)
#                         # node_edge_golden.append(-1)  # padding
#                         ne_golden_dict[t].append(-1) # padding
# 
#                     else:
#                         front, attr = data_list[i].edge_sets[t][et]
#                         pos_attr = front * self.num_edge_type + attr
#                         add_edge_golden.append(1)
#                         # node_edge_golden.append(pos_attr)
#                         ne_golden_dict[t].append(pos_attr)
#                 # do calculation
#                 new_batch = Batch.from_data_list(tmp_data)  # did not update gen_data???
#                 if len(new_batch.edge_attr):
#                     new_ctxs, new_x = self.encoder(new_batch, return_x=True)
#                     new_batch.x = new_x
#                     # new_data_list = new_batch.to_data_list()
#                     # for idx, i in enumerate(batch_list):
#                     #     gen_data[i].x = new_data_list[idx].x
#                 else:
#                     new_ctxs = ctxs[batch_list]
#                     new_x = new_batch.x
#                 new_nodes_embed = new_x[end_idxs]
#                 # pred if the node need new edges
#                 add_edge = self.add_edge(torch.cat(
#                                             [new_nodes_embed, new_ctxs, new_conds],
#                                             dim=-1))  # [1]
#                 add_edge_preds.append(add_edge)
#                 # dump data back to ctxs
#                 ctxs = ctxs.clone()  # prevent in-place operation on leaf node
#                 ctxs[batch_list] = new_ctxs
#                 # add golden to graph, this takes large amount of time
#                 num = t + 1
#                 for i in batch_list:
#                     if data_list[i].edge_sets[t][et] is None:
#                         continue
#                     front, attr = data_list[i].edge_sets[t][et]
#                     # cur_node_idx = num - 1
#                     cur_node_idx = t
#                     gen_data[i].add_edge([cur_node_idx, front], attr)
#                     gen_data[i].add_edge([front, cur_node_idx], attr)
#                 # predict node type
#                 num_nodes = torch.tensor([num for _ in range(len(new_nodes_embed))],
#                                          device=new_nodes_embed.device)
#                 new_nodes_embed = torch.repeat_interleave(new_nodes_embed, num_nodes, dim=0)
#                 node_edge = self.node_edge(torch.cat([new_x, new_nodes_embed], dim=1)) # [num_graph * (t+1), num_edge_type]
#                 node_edge = node_edge.view(len(batch_list), -1)  # [num_graph, (t+1) * num_edge_type]
#                 node_edge = node_edge[:, :-self.num_edge_type]  # get rid of newly added node
#                 # node_edge_preds.append(node_edge)
#                 ne_preds_dict[node_edge.shape[1]].append(node_edge)
# 
#         # add_node loss
#         add_node_preds = torch.cat(add_node_preds, dim=0)
#         add_node_golden = torch.tensor(add_node_golden,
#                                        device=add_node_preds.device,
#                                        dtype=torch.long)
#         an_loss = self.add_node_loss(add_node_preds, add_node_golden)
#         # add edge loss
#         add_edge_preds = torch.cat(add_edge_preds, dim=0).squeeze()
#         add_edge_golden = torch.tensor(add_edge_golden,
#                                        device=add_edge_preds.device,
#                                        dtype=torch.float)
#         ae_loss = self.add_edge_loss(add_edge_preds, add_edge_golden)
#         # node edge loss
#         ne_loss = torch.tensor(0, device=ae_loss.device, dtype=float)
#         for key in ne_preds_dict:
#             t = key // self.num_edge_type
#             preds = torch.cat(ne_preds_dict[key], dim=0)
#             gold = torch.tensor(ne_golden_dict[t], device=preds.device, dtype=torch.long)
#             ne_loss = ne_loss + self.node_edge_loss(preds, gold) / len(ne_preds_dict)
# 
#         # node_edge_golden = torch.tensor(node_edge_golden,
#         #                                 device=ae_loss.device,
#         #                                 dtype=torch.long)
#         # l, r = 0, 0
#         # ne_loss = torch.tensor(0, device=ae_loss.device, dtype=float)
#         # for p in node_edge_preds:
#         #     r += len(p)
#         #     ne_loss += self.node_edge_loss(p, node_edge_golden[l:r])
#         #     l = r
#         # ne_loss = ne_loss / len(node_edge_preds)
#         loss = an_loss + ae_loss + ne_loss
#         return loss
# 
#     def inference(self, conds, max_atom_num, max_edge_num=4, add_edge_th=0.5):
#         '''given conditions(graph embedding), reconstruct the molecule'''
#         add_node_preds, add_edge_preds, node_edge_preds = [], [], []
#         batch_size = len(conds)
#         stop = [False for _ in range(batch_size)]
#         ctxs = torch.zeros_like(conds, device=conds.device, requires_grad=True)
#         gen_data = [MolData(conds.device) for _ in range(batch_size)]
#         for t in range(max_atom_num):
#             # find all graphs that still need to generate
#             batch_list = [i for i in range(batch_size) if not stop[i]]
#             new_ctxs = ctxs[batch_list]
#             new_conds = conds[batch_list]
# 
#             add_node = self.add_node(torch.cat([new_ctxs, new_conds], dim=-1)) # [batch_size, num_atom_type]
#             atom_idxs = torch.argmax(add_node, dim=1)
#             # check stop
#             new_batch_list = []
#             for i, atom in enumerate(atom_idxs):
#                 if atom == self.stop_idx:
#                     stop[batch_list[i]] = True
#                 else:
#                     new_batch_list.append(batch_list[i])
#             batch_list = new_batch_list
#             if len(batch_list) == 0:
#                 break
#             # add generated atom idx
#             node_embeds = self.init_node(atom_idxs, new_ctxs, new_conds)        
#             for idx, i in enumerate(batch_list):
#                 gen_data[i].add_x(node_embeds[idx])
#                 gen_data[i].add_x_one_hot(atom_idxs[idx])
# 
#             # pred edges
#             edge_stop = torch.tensor([False for _ in range(batch_size)],
#                                       device=node_embeds.device)
#             for et in range(max_edge_num):
#                 batch_list = [i for i in batch_list if not edge_stop[i]]
#                 new_conds = conds[batch_list]
#                 # preprocess
#                 tmp_data = []
#                 end_idxs = []
#                 for i in batch_list:
#                     tmp_data.append(gen_data[i].get_dataobj()) # generate batch data
#                     if len(end_idxs) > 0:  # find ending node index of each graph
#                         end_idxs.append(end_idxs[-1] + gen_data[i].get_num_nodes())
#                     else:
#                         end_idxs.append(gen_data[i].get_num_nodes() - 1)
# 
#                 # do calculation
#                 new_batch = Batch.from_data_list(tmp_data)
#                 if len(new_batch.edge_attr):
#                     new_ctxs, new_x = self.encoder(new_batch, return_x=True)
#                     new_batch.x = new_x
#                 else:
#                     new_ctxs = ctxs[batch_list]
#                     new_x = new_batch.x
#                 new_nodes_embed = new_x[end_idxs]
#                 # pred if the node need new edges
#                 add_edge = self.add_edge(torch.cat(
#                                             [new_nodes_embed, new_ctxs, new_conds],
#                                             dim=-1))  # [1]
#                 # check stop
#                 pred_stop = add_edge.squeeze() < 0.5  # [batch_size, 1]
#                 stop_idxs = torch.nonzero(pred_stop).squeeze()
#                 edge_stop[stop_idxs] = True
#                 # dump data back to ctxs
#                 ctxs = ctxs.clone()  # prevent in-place operation on leaf node
#                 ctxs[batch_list] = new_ctxs
#                 # predict node type
#                 num = gen_data[0].get_num_nodes()
#                 num_nodes = torch.tensor([num for _ in range(len(new_nodes_embed))],
#                                          device=new_nodes_embed.device)
#                 new_nodes_embed = torch.repeat_interleave(new_nodes_embed, num_nodes, dim=0)
#                 node_edge = self.node_edge(torch.cat([new_x, new_nodes_embed], dim=1)) # [num_graph * t, num_edge_type]
#                 node_edge = node_edge.view(len(batch_list), -1)  # [num_graph, t * num_edge_type]
#                 node_edge = node_edge[:, :-1]  # get rid of newly added node
#                 pred_edge = torch.argmax(node_edge, dim=-1)  # [num_graph]
#                 # add egdes to graph, this takes  amount of time, remember to check stop
#                 fronts = pred_edge // self.num_edge_type 
#                 edge_types = pred_edge % self.num_edge_type
#                 idx = 0
#                 for front, attr in zip(fronts, edge_types):
#                     if not edge_stop[batch_list[idx]]:
#                         gen_data[idx].add_edge([t, front], attr)
#                         gen_data[idx].add_edge([front, t], attr)
#                     idx += 1
#         return [d.get_dataobj_gen() for d in gen_data]