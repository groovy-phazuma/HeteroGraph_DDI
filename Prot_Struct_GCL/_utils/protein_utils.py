# -*- coding: utf-8 -*-
"""
Created on 2024-01-29 (Mon) 13:06:00

Protein utils

References
- https://github.com/NYXFLOWER/TIP/blob/cfe3df48cf5b50e922e22d980cbe87b476352576/data/utils.py


@author: I.Azuma
"""
# %%
import numpy as np

import torch

# %%
def process_prot_edge(pp_net):
    indices = torch.LongTensor(np.concatenate((pp_net.col.reshape(1, -1),
                                               pp_net.row.reshape(1, -1)),
                                              axis=0))
    indices = remove_bidirection(indices, None)
    n_edge = indices.shape[1]

    rd = np.random.binomial(1, 0.9, n_edge)
    train_mask = rd.nonzero()[0]
    test_mask = (1 - rd).nonzero()[0]

    train_indices = indices[:, train_mask]
    train_indices = to_bidirection(train_indices, None)

    test_indices = indices[:, test_mask]
    test_indices = to_bidirection(test_indices, None)

    return train_indices, test_indices

def remove_bidirection(edge_index, edge_type):
    mask = edge_index[0] > edge_index[1]
    keep_set = mask.nonzero().view(-1)

    if edge_type is None:
        return edge_index[:, keep_set]
    else:
        return edge_index[:, keep_set], edge_type[keep_set]


def to_bidirection(edge_index, edge_type=None):
    tmp = edge_index.clone()
    tmp[0, :], tmp[1, :] = edge_index[1, :], edge_index[0, :]
    if edge_type is None:
        return torch.cat([edge_index, tmp], dim=1)
    else:
        return torch.cat([edge_index, tmp], dim=1), torch.cat([edge_type, edge_type])
