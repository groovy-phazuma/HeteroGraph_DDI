# -*- coding: utf-8 -*-
"""
Created on 2024-01-30 (Tue) 11:13:56

Data preprocessing for drugbank

@author: I.Azuma
"""
# %%
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedShuffleSplit

# %% Construct DDI graph
def csv2edgelist(ddi_df, drug_list):
    """Load csv file (Li's format) and create edgelist.

    Args:
        ddi_df (DataFrame): Undirected pairs
        	d1	d2	type	Neg samples
        0	DB04571	DB00460	0	DB01579$t
        1	DB00855	DB00460	0	DB01178$t
        2	DB09536	DB00460	0	DB06626$t
        ...	...	...	...	...
        191806	DB00415	DB00437	85	DB01058$h
        191807	DB00437	DB00691	85	DB09065$h

        drug_list (list): ['DB04571', 'DB00855', 'DB09536', ... ]s

    Returns:
        edges, edges_false: _description_
    """
    idx_d1 = [drug_list.index(t) for t in ddi_df['d1'].tolist()]
    idx_d2 = [drug_list.index(t) for t in ddi_df['d2'].tolist()]
    idx_neg = [drug_list.index(t.split('$')[0]) for t in ddi_df['Neg samples'].tolist()] # Remove $t,h

    edges = []
    edges_false = []
    for j in range(len(idx_d1)):
        edges.append((idx_d1[j], idx_d2[j]))
        edges_false.append((idx_d1[j], idx_neg[j]))
    
    return edges, edges_false

def load_data(separate_train_path, separate_test_path, drug_list, fold=0):
    trainval_ddi_df = pd.read_csv(separate_train_path)
    test_ddi_df = pd.read_csv(separate_test_path)

    # Valid preparation
    train_array = np.array(trainval_ddi_df)
    cv_split = StratifiedShuffleSplit(n_splits=2, test_size=len(test_ddi_df), random_state=fold)
    train_index, val_index = next(iter(cv_split.split(X=train_array, y=train_array[:, 2])))

    train_ddi_df = trainval_ddi_df.loc[train_index]
    val_ddi_df = trainval_ddi_df.loc[val_index]
    del trainval_ddi_df

    num_nodes = len(drug_list)
    train_edges, train_edges_false = csv2edgelist(train_ddi_df, drug_list) # 115084
    val_edges, val_edges_false = csv2edgelist(val_ddi_df, drug_list) # 38362
    test_edges, test_edges_false = csv2edgelist(test_ddi_df, drug_list) # 38362 (24.9%)

    # Train + Val + Test
    all_edges = np.concatenate([train_edges, val_edges, test_edges], axis=0) # NOTE: Undirected pairs
    data = np.ones(all_edges.shape[0])
    adj = sp.csr_matrix((data, (all_edges[:, 0], all_edges[:, 1])), shape=(num_nodes, num_nodes))

    # Train
    train_edges = np.array(train_edges)
    train_edges_false = np.array(train_edges_false)

    data_train = np.ones(train_edges.shape[0])
    data_train_false = np.ones(train_edges_false.shape[0])

    adj_train = sp.csr_matrix((data_train, (train_edges[:, 0], train_edges[:, 1])),shape=(num_nodes, num_nodes))
    adj_train_false = sp.csr_matrix((data_train_false, (train_edges_false[:, 0], train_edges_false[:, 1])),shape=(num_nodes, num_nodes))

    return adj, adj_train, adj_train_false, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def load_inductive_s1s2(file_path='/path/to/inductive_data/fold1', drug_list=[]):
    # Load each data
    df_ddi_train = pd.read_csv(file_path+'/train.csv')
    df_ddi_s1 = pd.read_csv(file_path+'/s1.csv')
    df_ddi_s2 = pd.read_csv(file_path+'/s2.csv')

    # Rename columns
    re_col = ['d1','d2','type','Neg samples','tmp']
    df_ddi_train.columns = re_col
    df_ddi_s1.columns = re_col
    df_ddi_s2.columns = re_col

    # Reorder known and unknown drugs
    train_neg = [t.split('$')[0] for t in df_ddi_train['Neg samples'].tolist()]
    known_drugs = (set(df_ddi_train['d1']) | set(df_ddi_train['d2'])) | set(train_neg)
    known_drugs = sorted(list(known_drugs))
    unknown_drugs = sorted(list(set(drug_list) - set(known_drugs)))
    all_drugs = known_drugs + unknown_drugs

    print('known drugs:',len(known_drugs))
    print('unknown drugs:',len(unknown_drugs))
    
    train_edges, train_edges_false = csv2edgelist(df_ddi_train, all_drugs)
    s1_edges, s1_edges_false = csv2edgelist(df_ddi_s1, all_drugs)
    s2_edges, s2_edges_false = csv2edgelist(df_ddi_s2, all_drugs)

    # Train + Val
    num_nodes = len(all_drugs)
    all_edges = np.concatenate([train_edges, s1_edges, s2_edges], axis=0) # NOTE: Undirected pairs
    data = np.ones(all_edges.shape[0])
    adj = sp.csr_matrix((data, (all_edges[:, 0], all_edges[:, 1])), shape=(num_nodes, num_nodes))

    # Train
    train_edges = np.array(train_edges)
    train_edges_false = np.array(train_edges_false)
    data_train = np.ones(train_edges.shape[0])
    data_train_false = np.ones(train_edges_false.shape[0])

    adj_train = sp.csr_matrix((data_train, (train_edges[:, 0], train_edges[:, 1])),shape=(num_nodes, num_nodes))
    adj_train_false = sp.csr_matrix((data_train_false, (train_edges_false[:, 0], train_edges_false[:, 1])),shape=(num_nodes, num_nodes))

    return adj, adj_train, adj_train_false, (train_edges, train_edges_false), (s1_edges, s1_edges_false), (s2_edges, s2_edges_false), known_drugs, unknown_drugs


def load_inductive_data(file_path='/path/to/inductive_data/fold1', drug_list=[], fold=1):
    # Load each data
    df_ddi_train = pd.read_csv(file_path+'/train.csv')
    df_ddi_s1 = pd.read_csv(file_path+'/s1.csv')
    df_ddi_s2 = pd.read_csv(file_path+'/s2.csv')

    # Rename columns
    re_col = ['d1','d2','type','Neg samples','tmp']
    df_ddi_train.columns = re_col
    df_ddi_s1.columns = re_col
    df_ddi_s2.columns = re_col

    # Reorder known and unknown drugs
    train_neg = [t.split('$')[0] for t in df_ddi_train['Neg samples'].tolist()]
    known_drugs = (set(df_ddi_train['d1']) | set(df_ddi_train['d2'])) | set(train_neg)
    known_drugs = sorted(list(known_drugs))
    unknown_drugs = sorted(list(set(drug_list) - set(known_drugs)))

    print('known drugs:',len(known_drugs))
    print('unknown drugs:',len(unknown_drugs))

    # Valid preparation
    train_array = np.array(df_ddi_train)
    cv_split = StratifiedShuffleSplit(n_splits=2, test_size=len(df_ddi_s1) + len(df_ddi_s2), random_state=fold)
    train_index, val_index = next(iter(cv_split.split(X=train_array, y=train_array[:, 2])))

    train_ddi_df = df_ddi_train.loc[train_index]
    val_ddi_df = df_ddi_train.loc[val_index]
    del df_ddi_train

    num_nodes = len(known_drugs)
    train_edges, train_edges_false = csv2edgelist(train_ddi_df, known_drugs) # NOTE: not drug_list
    val_edges, val_edges_false = csv2edgelist(val_ddi_df, known_drugs)

    # Train + Val
    all_edges = np.concatenate([train_edges, val_edges], axis=0) # NOTE: Undirected pairs
    data = np.ones(all_edges.shape[0])
    adj = sp.csr_matrix((data, (all_edges[:, 0], all_edges[:, 1])), shape=(num_nodes, num_nodes))

    # Train
    train_edges = np.array(train_edges)
    train_edges_false = np.array(train_edges_false)

    data_train = np.ones(train_edges.shape[0])
    data_train_false = np.ones(train_edges_false.shape[0])

    adj_train = sp.csr_matrix((data_train, (train_edges[:, 0], train_edges[:, 1])),shape=(num_nodes, num_nodes))
    adj_train_false = sp.csr_matrix((data_train_false, (train_edges_false[:, 0], train_edges_false[:, 1])),shape=(num_nodes, num_nodes))

    return adj, adj_train, adj_train_false, train_edges, train_edges_false, val_edges, val_edges_false, known_drugs, unknown_drugs


# %%
# Load Li's work (stored in zip format in SN-DDI repository)
dsn_ddi_path = '/workspace/home/azuma/DDI/github/Drug-Interaction-Research/DSN-DDI-dataset/dataset/drugbank/ddis.csv' # Li's work
ddi_df = pd.read_csv(dsn_ddi_path)

# Obtain smiles information
drug_smiles_df = pd.read_csv('/workspace/home/azuma/DDI/github/Drug-Interaction-Research/DSN-DDI-dataset/dataset/drugbank/drug_smiles.csv')
drug_list = drug_smiles_df['drug_id'].tolist()

# Example of fold0
separate_train_path = '/workspace/home/azuma/DDI/github/Drug-Interaction-Research/DSN-DDI-dataset/dataset/drugbank/fold0/train.csv'
separate_test_path = '/workspace/home/azuma/DDI/github/Drug-Interaction-Research/DSN-DDI-dataset/dataset/drugbank/fold0/test.csv'

adj, adj_train, adj_train_false, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = load_data(separate_train_path, separate_test_path, drug_list, fold=0)

