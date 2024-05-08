# -*- coding: utf-8 -*-
"""
Created on 2024-01-24 (Wed) 19:16:03
Encoder

Reference: https://github.com/ranzhran/HTCL-DDI
- encoder.py
- graph_conv.py

@author: I.Azuma
"""

# %%
import torch
import torch.nn as nn
from argparse import Namespace

import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# %%
class AttnGraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 dropout: float = 0.3, alpha: float = 0.2, act = F.elu):
        super(AttnGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.act = act

        self.W = nn.Parameter(torch.zeros(in_features, out_features))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.a = nn.Parameter(torch.zeros(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)


    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        :param input: (num_nodes, in_features)
        :param adj: (num_nodes, num_nodes)
        :return:
        """
        h = torch.mm(input, self.W)
        if self.bias is not None:
            h = h + self.bias
        
        e = self._prepare_attentional_mechanism_input(h)

        zero_vec = -9e15*torch.ones_like(e)

        adj_at = adj.to_dense()

        attention = torch.where(adj_at > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h)

        return self.act(h_prime)

    def _prepare_attentional_mechanism_input(self, h):
        Wh1 = torch.matmul(h, self.a[:self.out_features, :])
        Wh2 = torch.matmul(h, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class Attntopo(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 dropout: float = 0.3, alpha: float = 0.2, act = F.elu):
        super(Attntopo, self).__init__()
        self.in_features = in_features*2
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.act = act

        self.W = nn.Parameter(torch.zeros(in_features, out_features)).cuda()
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.a = nn.Parameter(torch.zeros(2*out_features, 1)).cuda()
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.W_si = nn.Parameter(torch.zeros(size=(1, 1))).cuda()
        nn.init.xavier_uniform_(self.W_si.data, gain=1.414)
        self.W_ei = nn.Parameter(torch.zeros(size=(1, 1))).cuda()
        nn.init.xavier_uniform_(self.W_ei.data, gain=1.414)

    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        :param input: (num_nodes, in_features)
        :param adj: (num_nodes, num_nodes)
        :return:
        """

        h = torch.mm(input, self.W).cuda()
        if self.bias is not None:
            h = h + self.bias

        e = self._prepare_attentional_mechanism_input(h)

        adj = adj.to_dense()
        e = e.cuda()
        adj = adj.cuda()
        
        # combine sij and eij
        adj2 = torch.matmul(adj, adj)
        adj3 = torch.matmul(adj2, adj)


        adj_w = adj + adj2 + adj3
        adj_w = adj_w.fill_diagonal_(0)


        e=abs(self.W_ei)*e+abs(self.W_si)*adj_w

        zero_vec = -9e15*torch.ones_like(e)
        adj=adj.cuda()

        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, h)

        return self.act(h_prime)
    
    def _prepare_attentional_mechanism_input(self, h):
        Wh1 = torch.matmul(h, self.a[:self.out_features, :])
        Wh2 = torch.matmul(h, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# %%
class GAT(nn.Module):
    def __init__(self, args: Namespace, num_features: int, features_nonzero: int,
                 dropout: float = 0.1, bias: bool = False, nheads=8, sparse: bool = True):
        super(GAT, self).__init__()
        self.input_dim = num_features
        self.dropout = nn.Dropout(dropout)
        self.bias = bias
        self.sparse = sparse

        GC = AttnGraphConvolution
        self.attentions = [GC(in_features=num_features, out_features=args.gat_hidden) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GC(args.gat_hidden * nheads, args.gat_hidden)

    def forward(self, features: torch.Tensor, adj: torch.FloatTensor) -> torch.Tensor:
        
        features = self.dropout(features)
        features = torch.cat([att(features, adj) for att in self.attentions], dim=1)
        features = self.dropout(features)

        return features


class TopoGAT(nn.Module):
    def __init__(self, args: Namespace, num_features: int, features_nonzero: int,
                 dropout: float = 0.1, bias: bool = False, nheads=8, sparse: bool = True):
        super(TopoGAT, self).__init__()
        self.input_dim = num_features
        self.dropout = nn.Dropout(dropout)
        self.bias = bias
        self.sparse = sparse

        GC = Attntopo
        self.attentions_1 = [GC(in_features=num_features, out_features=args.gat_hidden) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_1):
            self.add_module('attention_{}'.format(i), attention)

        self.attentions_2 = [GC(in_features=args.gat_hidden * nheads, out_features=args.gat_hidden) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_2):
            self.add_module('attention_{}'.format(i), attention)

        self.attentions_3 = [GC(in_features=args.gat_hidden * nheads, out_features=args.gat_hidden) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions_3):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, features: torch.Tensor, adj: torch.FloatTensor) -> torch.Tensor:
        
        features = self.dropout(features)
        features = torch.cat([att(features, adj) for att in self.attentions_1], dim=1)
        features = self.dropout(features)
        features = torch.cat([att(features, adj) for att in self.attentions_2], dim=1)
        features = self.dropout(features)
        features = torch.cat([att(features, adj) for att in self.attentions_3], dim=1)
        features = self.dropout(features)

        return features