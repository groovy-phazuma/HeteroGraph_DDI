# -*- coding: utf-8 -*-
"""
Created on 2024-01-16 (Tue) 15:40:45

Weisfeiler-Lehman Graph Kernels with WWL

@author: I.Azuma
"""
import numpy as np
import pandas as pd
import torch
import igraph as ig
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem

def ig_from_edgelist(edge_list):
    #edge_list = MOL_EDGE_LIST_FEAT_MTX[2173][0].numpy()
    input_edge = [(edge_list[0][i],edge_list[1][i]) for i in range(len(edge_list[0]))]
    nxg = nx.from_edgelist(input_edge) # networkx
    g = ig.Graph.from_networkx(nxg) # igraph

    return g

def create_igaphlist_nodefeature(MOL_EDGE_LIST_FEAT_MTX, drug_to_mol_graph):
    igraph_list = []
    feature_list = []
    graph_k = []
    for i,k in enumerate(MOL_EDGE_LIST_FEAT_MTX):
        edge_list = MOL_EDGE_LIST_FEAT_MTX[k][0]
        unique_nodes = torch.unique(edge_list) # nodes with aby bond
        if len(unique_nodes)==0:
            print(k)
            display(drug_to_mol_graph[k])
        else:
            g = ig_from_edgelist(edge_list.numpy()) # construct igraph class
            f = MOL_EDGE_LIST_FEAT_MTX[k][1].numpy()
            f = f[unique_nodes]

            if g.vcount()==len(f):
                igraph_list.append(g)
                feature_list.append(f)
                graph_k.append(k)

    return igraph_list, feature_list, graph_k

def main():
    df_drugs_smiles = pd.read_csv('/workspace/mnt/data1/Azuma/DDI_HDD1/PubChem/TIP_645/pubchempy_645x5_info.csv') 
    drug_map = pd.read_pickle('/workspace/home/azuma/DDI/github/TIP/data/index_map/drug-map.pkl')
    drug_id_mol_graph_tup = [(id, Chem.MolFromSmiles(smiles.strip())) for id, smiles in zip(df_drugs_smiles['CID'], df_drugs_smiles['IsomericSMILES'])]
    drug_to_mol_graph = {id:Chem.MolFromSmiles(smiles.strip()) for id, smiles in zip(df_drugs_smiles['CID'], df_drugs_smiles['IsomericSMILES'])}

    MOL_EDGE_LIST_FEAT_MTX = {drug_id: dp.get_mol_edge_list_and_feat_mtx(mol) 
                                    for drug_id, mol in drug_id_mol_graph_tup}
    MOL_EDGE_LIST_FEAT_MTX = {drug_id: mol for drug_id, mol in MOL_EDGE_LIST_FEAT_MTX.items() if mol is not None}

    import sys
    sys.path.append('/workspace/home/azuma/DDI/github/WWL/src/')
    from wwl.propagation_scheme import WeisfeilerLehman, ContinuousWeisfeilerLehman

    es = WeisfeilerLehman()
    node_representations = es.fit_transform(igraph_list, num_iterations=4)

    es = ContinuousWeisfeilerLehman()
    node_representations = es.fit_transform(igraph_list, node_features=feature_list, num_iterations=4)