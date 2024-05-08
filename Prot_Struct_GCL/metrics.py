import torch
import numpy as np
import scipy.sparse as sp
from typing import List, Tuple, Union
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,  log_loss, roc_curve

def gen_preds(edges_pos, edges_neg, adj_rec):
    preds = []
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]])

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]])

    return preds, preds_neg

def eval_threshold(labels_all, preds_all, preds, edges_pos, edges_neg, adj_rec, test):
    for i in range(int(0.5 * len(labels_all))):
        if preds_all[2*i] > 0.95 and preds_all[2*i+1] > 0.95:
            preds_all[2*i] = max(preds_all[2*i], preds_all[2*i+1])
            preds_all[2*i+1] = preds_all[2*i]
        else:
            preds_all[2*i] = min(preds_all[2*i], preds_all[2*i+1])
            preds_all[2*i+1] = preds_all[2*i]
    fpr, tpr, thresholds = roc_curve(labels_all, preds_all)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    preds_all_ = []
    for p in preds_all:
        if p >=optimal_threshold:
            preds_all_.append(1)
        else:
            preds_all_.append(0)
    return preds_all, preds_all_

def get_roc_score(model,
                  features, 
                  adj: torch.sparse.FloatTensor,
                  adj_tensor, 
                  drug_nums,
                  edges_pos: np.ndarray, edges_neg: Union[np.ndarray, List[list]], test = None) -> Tuple[float, float]:
    model.eval()
    
    rec, emb = model(features, adj, adj_tensor, drug_nums, return_embeddings = True)
    emb = emb.detach().cpu().numpy()
    rec = rec.detach().cpu().numpy()
    adj_rec = rec

    preds, preds_neg = gen_preds(edges_pos, edges_neg, adj_rec)
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    preds_all, preds_all_ = eval_threshold(labels_all, preds_all, preds, edges_pos, edges_neg, adj_rec, test)

    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    f1_score_ = f1_score(labels_all, preds_all_)
    acc_score = accuracy_score(labels_all, preds_all_)
    return roc_score, ap_score, f1_score_, acc_score

def get_roc_score_flex(rec, edges_pos: np.ndarray, edges_neg: Union[np.ndarray, List[list]], test = None) -> Tuple[float, float]:
    #model.eval()
    #rec, emb = model(features, adj, adj_tensor, drug_nums, return_embeddings = True)
    rec = rec.detach().cpu().numpy()
    adj_rec = rec

    preds, preds_neg = gen_preds(edges_pos, edges_neg, adj_rec)
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    preds_all, preds_all_ = eval_threshold(labels_all, preds_all, preds, edges_pos, edges_neg, adj_rec, test)

    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    f1_score_ = f1_score(labels_all, preds_all_)
    acc_score = accuracy_score(labels_all, preds_all_)

    return roc_score, ap_score, f1_score_, acc_score

