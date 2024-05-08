# -*- coding: utf-8 -*-
"""
Created on 2024-01-26 (Fri) 18:18:05

Multi-layer graph attention network (GAT)

@author: I.Azuma
"""
# %%
import torch
import torch.nn as nn

from ..encoder import GAT
from ..losses_info import GcnInfomax

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


class MultiLayerGAT(nn.Module):

    def __init__(self, args: Namespace, num_features: int, features_nonzero: int,
                 dropout: float = 0.3, bias: bool = False,
                 sparse: bool = True):
        super(MultiLayerGAT, self).__init__()
        self.num_features = num_features
        self.features_nonzero = features_nonzero
        self.dropout = dropout
        self.bias = bias
        self.sparse = sparse
        self.args = args
        #self.create_encoder(args)       

        self.struc_enc = self.select_encoder1(args)
        self.sigmoid = nn.Sigmoid()
        self.DGI_setup()
        self.create_ffn(args)

    def select_encoder1(self, args: Namespace):
        return GAT(args, self.num_features, self.features_nonzero,
        dropout=self.dropout, bias=self.bias,sparse=self.sparse)


    def create_ffn(self, args: Namespace):
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        self.fusion_ffn_global = nn.Linear(args.gat_hidden*8, args.ffn_hidden_size)

        ffn = []
        for _ in range(args.ffn_num_layers - 2):
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
            ])
        ffn.extend([
            activation,
            dropout,
            nn.Linear(args.ffn_hidden_size, args.drug_nums),
        ])
        # Create FFN model
        self.ffn = nn.Sequential(*ffn)
        self.dropout = dropout


    def DGI_setup(self):
        self.DGI_model1 = GcnInfomax(self.args)
        self.DGI_model2 = GcnInfomax(self.args)


    def forward(self,
                batch: torch.Tensor,
                adj: torch.sparse.FloatTensor,
                adj_tensor,
                drug_nums,
                return_embeddings: bool = False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        feat_orig = batch

        # structural view
        embeddings1 = self.struc_enc(feat_orig, adj)
        feat_g1 = self.dropout(embeddings1)
        fused_feat_g1 = self.fusion_ffn_global(feat_g1)
        output_g1 = self.ffn(fused_feat_g1)
        outputs_1 = self.sigmoid(output_g1)
        outputs_g1 = outputs_1.view(-1)

        local_embed = feat_orig
        DGI_loss1 = self.DGI_model1(embeddings1, local_embed, adj_tensor, drug_nums)
        
        if return_embeddings:
            return outputs_1, embeddings1 # (544, 544), (544, 4096)

        return outputs_g1, DGI_loss1