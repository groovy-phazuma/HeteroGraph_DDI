# -*- coding: utf-8 -*-
"""
Created on 2024-01-31 (Wed) 14:43:01

Dual Gat Model for Inductive Scenario

@author: I.Azuma
"""
# %%
import torch
import torch.nn as nn

from ..my_encoder import GAT

from argparse import Namespace
from typing import Union, Tuple, List

# %%
def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')


class DualGat(nn.Module):
    def __init__(self, ddi_drug_nums:int, sim_drug_nums:int,num_features:int, dropout:float=0.3, gat_hidden=64):
        super(DualGat, self).__init__()
        self.ddi_drug_nums = ddi_drug_nums
        self.sim_drug_nums = sim_drug_nums
        self.num_features = num_features
        self.gat_hidden = gat_hidden
        
        self.nheads = 8
        self.dropout = dropout
        self.bias = False

        self.activation = 'ReLU'
        self.ffn_hidden_size = 264

        # architecture
        self.ddi_enc = self.gat_encoder(in_features=self.num_features, out_features=self.gat_hidden) # Knowledge-based
        self.sim_enc = self.gat_encoder(in_features=self.num_features, out_features=self.gat_hidden) # Similarity-based

        self.ddi_sigmoid = nn.Sigmoid()
        self.sim_sigmoid = nn.Sigmoid()

        # ddi_drug_nums = 1364, sim_drug_nums = 1364+342
        self.ddi_ffn_global, self.ddi_ffn, self.ddi_dropout = self.create_ffn(self.num_features)
        self.sim_ffn_global, self.sim_ffn, self.sim_dropout = self.create_ffn(sim_drug_nums)
    
    def gat_encoder(self, in_features, out_features):
        return GAT(in_features, out_features, dropout=self.dropout, nheads=self.nheads, bias=self.bias)
    
    def create_ffn(self, out_dim:int):
        dropout = nn.Dropout(self.dropout)
        activation = get_activation_function(self.activation)

        fusion_ffn_global = nn.Linear(self.gat_hidden*self.nheads, self.ffn_hidden_size)

        ffn = []
        ffn.extend([
            activation,
            dropout,
            nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size),
        ])
        ffn.extend([
            activation,
            dropout,
            nn.Linear(self.ffn_hidden_size, out_dim),
        ])
        # Create FFN model
        ffn = nn.Sequential(*ffn)
        dropout = dropout # Update

        return fusion_ffn_global, ffn, dropout
    
    def forward(self,
                batch: torch.Tensor,
                ddi_adj: torch.sparse.FloatTensor,
                sim_adj: torch.sparse.FloatTensor,
                return_embeddings: bool = False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        feat_orig = batch # (1706, 133)
        ddi_feat_orig = feat_orig[0:self.ddi_drug_nums,:] # (1364, 133)

        # Knowledge-based DDI
        ddi_emb = self.ddi_enc(ddi_feat_orig, ddi_adj)
        feat_g1 = self.ddi_dropout(ddi_emb) # --> Similarity-based graph
        # feat_g1: (ddi_drug_Num, self.gat_hidden*self.nheads)

        fused_feat_g1 = self.ddi_ffn_global(feat_g1)
        output_g1 = self.ddi_ffn(fused_feat_g1)
        outputs_1 = self.ddi_sigmoid(output_g1)
        #flatten_g1 = outputs_1.view(-1)

        # Structural similariy-based graph
        sim_feat_orig = torch.concat([outputs_1, feat_orig[self.ddi_drug_nums:,:] ]) # FIXME: Try other layer as well

        sim_emb = self.sim_enc(sim_feat_orig, sim_adj)
        feat_g2 = self.sim_dropout(sim_emb)

        fused_feat_g2 = self.sim_ffn_global(feat_g2)
        output_g2 = self.sim_ffn(fused_feat_g2)
        outputs_2 = self.sim_sigmoid(output_g2)
        #flatten_g2 = outputs_2.view(-1)

        if return_embeddings:
            return outputs_1, outputs_2, ddi_emb, sim_emb # (544, 544), (544, 4096)

        return outputs_1, outputs_2
