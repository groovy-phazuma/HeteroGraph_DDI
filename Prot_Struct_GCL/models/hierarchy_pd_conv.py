# -*- coding: utf-8 -*-
"""
Created on 2024-01-28 (Sun) 17:01:48

- Protein-Protein Interaction
- Drug-Protein Interaction

References
- TIP (https://github.com/NYXFLOWER/TIP)

@author: I.Azuma
"""
# %%
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import  GCNConv, MessagePassing

import torch.nn.functional as F
from torch_geometric.data import Data

# %%
class Setting(object):
    def __init__(self, sp_rate=0.9, lr=0.01, prot_drug_dim=16, n_embed=48, n_hid1=32, n_hid2=16, num_base=32) -> None:
        super().__init__()
        self.sp_rate = sp_rate
        self.lr = lr
        self.prot_drug_dim = prot_drug_dim
        self.n_embed = n_embed
        self.n_hid1 = n_hid1graph
        self.n_hid2 = n_hid2
        self.num_base = num_base

class MyPDConv(nn.Module):
    def __init__(self, settings:Setting, device, mod='cat', data_path='./data/data_dict.pkl', ) -> None:
        super().__init__()
        self.mod = mod
        assert mod in {'cat', 'add'}
        self.device = device
        self.settings = settings
        self.data = self.__prepare_data(data_path, settings.sp_rate).to(device)
        self.__prepare_model()
        self.sigmoid = nn.Sigmoid()
    
    def __get_ndata__(self):
        return self.data.n_drug_feat, self.data.n_dd_et, self.data.n_prot, self.data.n_prot, self.data.n_drug
    
    def __get_dimset__(self):
        return self.settings.prot_drug_dim, self.settings.num_base, self.settings.n_embed, self.settings.n_hid1, self.settings.n_hid2
    
    def __prepare_data(self, data_path, sp_rate):
        # Load data
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        data = Data.from_dict(data_dict)

        return data
    
    def __prepare_model():
        # encoder
        if self.mod == 'cat':
            self.encoder = FMEncoderCat(
                self.device,
                *self.__get_ndata__(), 
                *self.__get_dimset__()
            ).to(self.device)
        else:
            pass # FIXME
        
    def forward(self):
        self.embeddings = self.encoder(self.data.d_feat, self.data.dd_train_idx, self.data.dd_train_et, self.data.dd_train_range, self.data.d_norm, self.data.p_feat, self.data.pp_train_indices, self.data.dp_edge_index, self.data.dp_range_list)

        outputs_1 = self.sigmoid(self.embeddings)
        outputs_g1 = outputs_1.view(-1)


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

class PPEncoder(torch.nn.Module):

    def __init__(self, in_dim, hid1=32, hid2=16):
        super(PPEncoder, self).__init__()
        self.out_dim = hid2

        self.conv1 = GCNConv(in_dim, hid1, cached=True)
        self.conv2 = GCNConv(hid1, hid2, cached=True)

        # self.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x, inplace=True)
        x = self.conv2(x, edge_index)
        return x

    # def reset_parameters(self):
    #     self.embed.data.normal_()

class FMEncoderCat(torch.nn.Module):

    def __init__(self, device, in_dim_drug, in_dim_prot,
                 uni_num_prot, uni_num_drug, prot_drug_dim=16,
                 num_base=32, n_embed=48, n_hid1=32, n_hid2=16):

        super(FMEncoderCat, self).__init__()
        self.out_dim = n_hid2
        self.uni_num_drug = uni_num_drug
        self.uni_num_prot = uni_num_prot

        self.d_norm = torch.ones(in_dim_drug).to(device)

        # on pp-net
        self.pp_encoder = PPEncoder(in_dim_prot)

        # feat: drug index
        self.embed = Param(torch.Tensor(in_dim_drug, n_embed))

        # on pd-net
        self.hgcn = MyHierarchyConv(self.pp_encoder.out_dim, prot_drug_dim, uni_num_prot, uni_num_drug)
        self.hdrug = torch.zeros((self.uni_num_drug, self.pp_encoder.out_dim)).to(device)

        self.reset_parameters()

        dropout = nn.Dropout(0.4)
        activation = get_activation_function('ReLU')
        
        ffn = []
        ffn_num_layers=3
        ffn_hidden_size = n_hid2+n_embed
        for _ in range(ffn_num_layers - 2):
            ffn.extend([
                activation,
                dropout,
                nn.Linear(ffn_hidden_size, ffn_hidden_size),
            ])
        ffn.extend([
            activation,
            dropout,
            nn.Linear(ffn_hidden_size, uni_num_drug),
        ])
        # Create FFN model
        self.ffn = nn.Sequential(*ffn)
        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()


    def forward(self, x_prot, pp_edge_index, dp_edge_index, dp_range_list, x_drug, return_embeddings: bool = False):

        # pp-net
        x_prot = self.pp_encoder(x_prot, pp_edge_index) # (protein_num, n_hid2)

        # pd-net
        x_prot = torch.cat((x_prot, self.hdrug)) # (protein_num+drug_num, n_hid2)
        x_prot = self.hgcn(x_prot, dp_edge_index, dp_range_list) # (drug_num, n_hid2)

        # d-embed
        x_drug = torch.matmul(x_drug, self.embed)
        x_drug = x_drug / self.d_norm.view(-1, 1) # (drug_num, n_embed)

        # concat
        x_merge = torch.cat((x_drug, x_prot), dim=1) # (drug_num, n_hid2+n_embed)

        feat_g = self.dropout(x_merge)
        output_g = self.ffn(feat_g)
        outputs = self.sigmoid(output_g)
        outputs_g = outputs.view(-1)

        if return_embeddings:
            return outputs, x_merge

        return outputs_g

    def reset_parameters(self):
        self.embed.data.normal_()


class MyHierarchyConv(MessagePassing):
    """ directed gcn layer for pd-net """
    def __init__(self, in_dim, out_dim,
                 unique_source_num, unique_target_num,
                 is_after_relu=True, is_bias=False, **kwargs):

        super(MyHierarchyConv, self).__init__(aggr='mean', **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.unique_source_num = unique_source_num
        self.unique_target_num = unique_target_num
        self.is_after_relu = is_after_relu

        # parameter setting
        self.weight = Param(torch.Tensor(in_dim, out_dim))

        if is_bias:
            self.bias = Param(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.is_after_relu:
            self.weight.data.normal_(std=1/np.sqrt(self.in_dim))
        else:
            self.weight.data.normal_(std=2/np.sqrt(self.in_dim))

        if self.bias:
            self.bias.data.zero_()

    def forward(self, x, edge_index, range_list):
        return self.propagate(edge_index, x=x, range_list=range_list)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, range_list):
        if self.bias:
            aggr_out += self.bias

        out = torch.matmul(aggr_out[self.unique_source_num:, :], self.weight)
        assert out.shape[0] == self.unique_target_num

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                  self.in_dim,
                                  self.out_dim)