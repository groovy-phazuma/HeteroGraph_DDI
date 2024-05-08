# -*- coding: utf-8 -*-
"""
Created on 2024-01-26 (Fri) 17:24:18

@author: I.Azuma
"""
# %%
import numpy as np
import scipy.sparse as sp
import torch
from typing import Tuple
from argparse import Namespace
import logging

# %%
def sparse_mx_to_torch_sparse_tensor(sparse_mx) \
        -> torch.sparse.FloatTensor:
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_adj(adj: sp.csr_matrix) -> sp.coo_matrix:
    adj = sp.coo_matrix(adj)
    # eliminate self-loop
    adj_ = adj
    rowsum = np.array(adj_.sum(0))
    rowsum_power = []
    for i in rowsum:
        for j in i:
            if j !=0 :
                j_power = np.power(j, -0.5)
                rowsum_power.append(j_power)
            else:
                j_power = 0
                rowsum_power.append(j_power)
    rowsum_power = np.array(rowsum_power)
    degree_mat_inv_sqrt = sp.diags(rowsum_power)

    adj_norm = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_norm

def simmat2knngraph(sim_mat, topk=5, undirected=True):
    """Construct KNN graph from similarity matrix

    Args:
        sim_mat (np.array): Similarity (correlation) matrix.
            e.g. 
            [[1.0, 0.2, 0.3],
             [0.2, 1.0, 0.4],
             [0.3, 0.4, 1.0]]
        topk (int, optional): Top K. Defaults to 5.
        undirected (bool, optional): Defaults to True.

    """
    # Cutting
    update_sim = []
    for v in sim_mat:
        tmp = sorted(v,reverse=True)
        threshold = tmp[topk]
        v = np.where(v>threshold,1,0)
        update_sim.append(v)
    update_sim = np.array(update_sim)

    # Convert to undirected graph
    if undirected:
        merged_sim = update_sim + update_sim.T
        merged_sim = np.where(merged_sim>0,1,0)
        return merged_sim
    else:
        return update_sim