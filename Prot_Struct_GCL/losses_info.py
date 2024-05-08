# -*- coding: utf-8 -*-
"""
Created on 2024-01-24 (Wed) 19:42:23

Losses info

Reference: https://github.com/ranzhran/HTCL-DDI
- losses_info.py
- gan_losses.py
- misc.py
- deepinfomax.py

@author: I.Azuma
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace

import math

class PriorDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = nn.Linear(input_dim, input_dim)
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))

class FF_local(nn.Module):
    def __init__(self, args:Namespace, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, args.FF_hidden1),
            nn.ReLU(),
            nn.Linear(args.FF_hidden1, args.FF_hidden2),
            nn.ReLU(),
            nn.Linear(args.FF_hidden2, args.FF_output),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, args.FF_output)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

class FF_global(nn.Module):
    def __init__(self, args:Namespace, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, args.FF_hidden1),
            nn.ReLU(),
            nn.Linear(args.FF_hidden1, args.FF_hidden2),
            nn.ReLU(),
            nn.Linear(args.FF_hidden2, args.FF_output),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, args.FF_output)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

class GcnInfomax(nn.Module):
  def __init__(self, args: Namespace, gamma=.1):
    super(GcnInfomax, self).__init__()
    self.args = args
    self.gamma = gamma
    self.prior = args.prior
    self.features_dim = args.hidden_size
    self.embedding_dim = args.gat_hidden*8 # FIXME: args.num_heads1 ??
    self.local_d = FF_local(args, self.features_dim)
    self.global_d = FF_global(args, self.embedding_dim)

    if self.prior:
        self.prior_d = PriorDiscriminator(self.embedding_dim)

  def forward(self, embeddings, features, adj_tensor, num_drugs):
    
    g_enc = self.global_d(embeddings)
    l_enc = self.local_d(features)
    measure='JSD' # ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
    local_global_loss = local_global_drug_loss_(self.args, l_enc, g_enc, adj_tensor, num_drugs, measure)
    eps = 1e-5
    if self.prior:
        prior = torch.rand_like(embeddings)
        term_a = torch.log(self.prior_d(prior) + eps).mean()
        term_b = torch.log(1.0 - self.prior_d(embeddings) + eps).mean()
        PRIOR = - (term_a + term_b) * self.gamma
    else:
        PRIOR = 0
    
    return local_global_loss + PRIOR

# %%
def log_sum_exp(x, axis=None):
    """Log sum exp function

    Args:
        x: Input.
        axis: Axis over which to perform sum.

    Returns:
        torch.Tensor: log sum exp

    """
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y

def random_permute(X):
    """Randomly permutes a tensor.

    Args:
        X: Input tensor.

    Returns:
        torch.Tensor

    """
    X = X.transpose(1, 2)
    b = torch.rand((X.size(0), X.size(1))).cuda()
    idx = b.sort(0)[1]
    adx = torch.range(0, X.size(1) - 1).long()
    X = X[idx, adx[None, :]].transpose(1, 2)
    return X


def raise_measure_error(measure):
    supported_measures = ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
    raise NotImplementedError(
        'Measure `{}` not supported. Supported: {}'.format(measure,
                                                           supported_measures))


def get_positive_expectation(p_samples, measure, average=True):
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise_measure_error(measure)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise_measure_error(measure)

    if average:
        return Eq.mean()
    else:
        return Eq

def local_global_drug_loss_(args: Namespace, l_enc, g_enc, adj_tensor, num_drugs, measure):
    if args.cuda:
        pos_mask = adj_tensor.cuda() + torch.eye(num_drugs).cuda()
        neg_mask = torch.ones((num_drugs, num_drugs)).cuda() - pos_mask
    else:
        pos_mask = adj_tensor + torch.eye(num_drugs)
        neg_mask = torch.ones((num_drugs, num_drugs)) - pos_mask

    res = torch.mm(l_enc, g_enc.t())
    num_edges = args.num_edges_w + num_drugs
    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_edges
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_drugs ** 2 - 2 * num_edges)

    return (E_neg - E_pos)