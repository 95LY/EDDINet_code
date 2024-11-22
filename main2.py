from __future__ import division
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from utils import *
# from models import SFGCN
# from models import SFGCN2
# from models import Model
import copy
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.dia
from scipy.linalg import fractional_matrix_power, inv
import numpy as np
import networkx as nx
import numpy
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import average_precision_score
import tensorflow as tf
from initializations import *
import tensorflow as tf
import math

from models import DGI

import random

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

import os
import torch.nn as nn
import argparse
from config import Config
import process
from logreg import LogReg


# AMGCN
###################

def zero(matrix,batch):
    # 设置两个数组，一个记录0元素的行号，一个记录0元素的列号。再将该行该列全变为0
    # row = [True for i in range(0, len(matrix))]
    # col = [True for j in range(0, len(matrix[0]))]
    for r in range(0, len(matrix)):
        if r in batch:
            None
        else:
            for c in range(0, len(matrix[0])):
                matrix[r][c] == 0
    return matrix

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj) #取出稀疏矩阵上三角部分的元素，保存仍然是稀疏矩阵形式
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    # num_test = int(np.floor(edges.shape[0] / 10.))
    # num_val = int(np.floor(edges.shape[0] / 10.))
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 10.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)

    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]

    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]

    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    tmp = 1

    test_edges_false = []
    while len(test_edges_false) < tmp * len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0]) #函数的作用是，返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)。如果没有写参数high的值，则返回[0,low)的值。
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < tmp * len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):  # own add
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def get_roc_score(edges_pos, edges_neg, adj_rec, adj_orig):
    sigm = nn.Sigmoid()
    preds = []

    pos = []
    for e in edges_pos:
        preds.append(sigm(adj_rec[e[0], e[1]]).detach().numpy())  # **
        pos.append(adj_orig[e[0], e[1]]) # 全是 1

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigm(adj_rec[e[0], e[1]]).detach().numpy())
        neg.append(adj_orig[e[0], e[1]])

    # loc1
    preds_all = np.hstack([preds, preds_neg])  # 存在的边与不存在的边的预测值 [0, 1]
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])  # 存在的边与不存在的边的真实值 0 | 1


    roc_score = roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)

    ap_sc = average_precision_score(labels_all, preds_all)

    # ----------------------------------全部预测值调整---------------------------
    preds_all = np.round(preds_all)
    # yu = 0.7
    # for m in range(0, len(preds_all)):
    #     if preds_all[m] > yu:
    #         preds_all[m] = 1
    #     else:
    #         preds_all[m] = 0

    # print(preds_all)
    F1_score = f1_score(labels_all, preds_all, average='binary')

    return roc_score, aupr_sc, ap_sc, F1_score
# ---------------------------------------------------------------------------------------------------------
def train(adj, features_enzyme, features_indication, features_sideeffect, features_transporter):
    adj_orig = adj

    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    # print(adj)
    #---------------------------------------划分边----------------------------------------------------
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)

    adj = adj_train
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    Featcoo = adj.tocoo()
    features = torch.sparse.FloatTensor(torch.LongTensor([Featcoo.row.tolist(), Featcoo.col.tolist()]),
                                        torch.FloatTensor(Featcoo.data.astype(np.float)))
    # features = features.to_dense()


    #------------------------------------------处理sp------------------------------------------------
    sparse = True
    # features, _ = process.preprocess_features(features)
    features_enzyme = features_enzyme.todense()
    features_indication = features_indication.todense()
    features_sideeffect = features_sideeffect.todense()
    features_transporter = features_transporter.todense()

    nb_nodes = features_enzyme.shape[0]
    ft_size = features_enzyme.shape[1]

    # adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = adj + sp.eye(adj.shape[0])

    if sparse:
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = (adj + sp.eye(adj.shape[0])).todense()

    features_enzyme = torch.FloatTensor(features_enzyme[np.newaxis])
    features_indication = torch.FloatTensor(features_indication[np.newaxis])
    features_sideeffect = torch.FloatTensor(features_sideeffect[np.newaxis])
    features_transporter = torch.FloatTensor(features_transporter[np.newaxis])

    if not sparse:
        adj = torch.FloatTensor(adj[np.newaxis])

    #---------------------------------------开始训练--------------------------------------------------
    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0
    batch_size = 1
    # nb_epochs = 10000
    nb_epochs = 1000
    # patience = 50
    patience = 100
    # lr = 0.001
    lr = 0.001
    l2_coef = 0.0
    drop_prob = 0.0
    hid_units = 16
    # hid_units = 110
    sparse = True
    nonlinearity = 'prelu'  # special name to separate parameters
    # reg_coef = 0.001
    reg_coef = 0.001


    model = DGI(ft_size, hid_units, nonlinearity)
    # H = nn.Parameter(torch.FloatTensor(1, 548, 16))
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    for epoch in range(nb_epochs):
        model.train()
        optimiser.zero_grad()

        idx = np.random.permutation(nb_nodes)
        # shuf_fts = features[idx]
        shuf_fts_enzyme = features_enzyme[:, idx, :]
        shuf_fts_indication = features_indication[:, idx, :]
        shuf_fts_sideeffect = features_sideeffect[:, idx, :]
        shuf_fts_transporter = features_transporter[:, idx, :]

        lbl_1 = torch.ones(batch_size, nb_nodes)
        lbl_2 = torch.zeros(batch_size, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)


        # logits_enzyme, logits_indication, logits_sideeffect, logits_transporter, reg_loss = model(H, features_enzyme, features_indication, features_sideeffect, features_transporter, shuf_fts_enzyme, shuf_fts_indication, shuf_fts_sideeffect, shuf_fts_transporter, sp_adj if sparse else adj, sparse, None, None, None)
        logits_enzyme, logits_indication, logits_sideeffect, logits_transporter, reg_loss = model(features_enzyme,
                                                                                                  features_indication,
                                                                                                  features_sideeffect,
                                                                                                  features_transporter,
                                                                                                  shuf_fts_enzyme,
                                                                                                  shuf_fts_indication,
                                                                                                  shuf_fts_sideeffect,
                                                                                                  shuf_fts_transporter,
                                                                                                  sp_adj if sparse else adj,
                                                                                                  sparse, None, None,
                                                                                                  None)
        loss_enzyme = b_xent(logits_enzyme, lbl)
        loss_indication = b_xent(logits_indication, lbl)
        loss_sideeffect = b_xent(logits_sideeffect, lbl)
        loss_transporter = b_xent(logits_transporter, lbl)
        #------------------------对比损失-----------------------
        contr_loss = loss_enzyme + loss_indication + loss_sideeffect + loss_transporter
        # print(reg_loss)
        #------------------------总损失--------------------------
        # print(-reg_loss)
        loss = contr_loss + abs(reg_coef * reg_loss)
        # loss = contr_loss
        # ----------------------------------------优化模型----------------------------------------------
        # loss.backward()
        # optimiser.step()

        print('Loss:', loss)

        if loss < best:
            best = loss
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_dgi.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break

        loss.backward()
        optimiser.step()
        # -----------------------------计算成绩---------------------------------------------------------
    h_1_enzyme_1, h_1_indication_1, h_1_sideeffect_1, h_1_transporter_1 = model.embed(features_enzyme, features_indication, features_sideeffect, features_transporter, sp_adj if sparse else adj, sparse)
    h_1_enzyme = torch.squeeze(h_1_enzyme_1, 0)
    h_1_indication = torch.squeeze(h_1_indication_1, 0)
    h_1_sideeffect = torch.squeeze(h_1_sideeffect_1, 0)
    h_1_transporter = torch.squeeze(h_1_transporter_1, 0)
    # h_1 = torch.cat((h_1_enzyme, h_1_indication), 1)
    # h_1 = torch.cat((h_1, h_1_sideeffect), 1)
    # h_1 = torch.cat((h_1, h_1_transporter), 1)
    h_1 = torch.stack([h_1_enzyme, h_1_indication, h_1_sideeffect, h_1_transporter], dim=1)
    # print(h_1.size())
    # h_1, att1 = Attention(h_1)
    # h_1 = h_1.unsqueeze(0)
    nb_classes = 8
    roc = 0
    prc = 0
    ap = 0
    F1 = 0
    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        # log.cuda()

        pat_steps = 0
        best_acc = torch.zeros(1)
        # best_acc = best_acc.cuda()
        best_acc = best_acc
        for _ in range(100):
            log.train()
            opt.zero_grad()

            logits = log(h_1)
            adj_rebuilt = torch.mm(torch.squeeze(logits), torch.t(torch.squeeze(logits)))
            # print(adj_rebuilt.size())
            adj_rebuilt = torch.reshape(adj_rebuilt, (1, -1))
            # adj_rebuilt = torch.unsqueeze(adj_rebuilt, 0)
            # print(adj_rebuilt.size())
            # print(type(adj_rebuilt))

            # adjcoo = np.reshape(adj.todense(), [-1])

            adjcoo = adj_train.tocoo()
            values = adjcoo.data
            indices = np.vstack((adjcoo.row, adjcoo.col))
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = adjcoo.shape
            adjb = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
            # adj = torch.sparse.FloatTensor(i, v, torch.Size(shape))
            # adj = adj.to_dense()
            # adj = torch.FloatTensor(i, v, torch.Size(shape))
            adjb = torch.reshape(adjb, (1, -1))
            # adj = torch.unsqueeze(adj, 0)
            # print(adj.size())

            # print(type(adj))

            loss1 = b_xent(adj_rebuilt, adjb)
            # print(loss1)
            loss1.backward(retain_graph=True)
            opt.step()

        adj_rebuilt = adj_rebuilt.reshape(548, 548)
        val_roc_curr, val_aupr_sc, val_ap_sc, val_F1_score = get_roc_score(val_edges, val_edges_false, adj_rebuilt,
                                                                           adj_orig)
        test_roc_score, test_aupr_sc, test_ap_sc, test_F1_score = get_roc_score(test_edges, test_edges_false,
                                                                                adj_rebuilt,
                                                                                adj_orig)
        roc_score = (val_roc_curr + test_roc_score) / 2.0
        aupr_sc = (val_aupr_sc + test_aupr_sc) / 2.0
        ap_sc = (val_ap_sc + test_ap_sc) / 2.0
        # macro_f1_sc = (val_macro_f1 + test_macro_f1) / 2.0
        # micro_f1_sc = (val_micro_f1 + test_micro_f1) / 2.0
        F1_sc = (val_F1_score + test_F1_score) / 2.0

        print('auroc_score=', "{:.5f}".format(roc_score),
              'auprc_score=', "{:.5f}".format(aupr_sc),
              'ap_score=', "{:.5f}".format(ap_sc),
              'F1_score=', "{:.5f}".format(F1_sc),
              )

        roc += roc_score.item()
        prc += aupr_sc.item()
        ap += ap_sc.item()
        F1 += F1_sc.item()

    ave_roc = roc / 50
    ave_prc = prc / 50
    ave_ap = ap / 50
    ave_F1 = F1 / 50
    return ave_roc, ave_prc, ave_ap, ave_F1


if __name__ == "__main__":

    effect_name = "C0553662_95"
    # ---------读取enzyme数据---------------------------------------------------------------------------
    df = pd.read_csv("./enzyme/%s.csv" % (effect_name))
    row = []
    col = []
    data = []
    for i, j, z in zip(df['A'], df['B'], df['V']):
        if i == j:
            row.append(i)
            col.append(j)
            data.append(1)
        elif z == -1:
            row.append(i)
            col.append(j)
            data.append(1)
    adj = sp.csr_matrix((data, (row, col)))

    similarity = "0"
    # df = pd.read_csv("./enzyme/%s.csv" % (similarity))
    df = pd.read_csv("./enzyme/%s.csv" % (effect_name))
    row = []
    col = []
    data = []
    for i, j, z in zip(df['A'], df['B'], df['V']):
        row.append(i)
        col.append(j)
        data.append(z)
    nM_graph = sp.csr_matrix((data, (row, col)), shape=(548, 548))
    nM_graph = nM_graph.toarray() + np.identity(548)
    nM_graph = nM_graph + nM_graph.T
    nM_graph[nM_graph > 0] = 1.
    nM_graph[nM_graph < 0] = -1.
    # SPM_1 = copy.deepcopy(nM_graph)
    # SPM_m1 = copy.deepcopy(SPM_1)
    # SPM_m2 = np.dot(SPM_m1, SPM_1)
    # SPM_m3 = np.dot(SPM_m2, SPM_1)
    # SPM_m = SPM_m1 + SPM_m2 + SPM_m3
    #
    # SPM_m[SPM_m > 0] = 1.
    # SPM_m[SPM_m < 0] = -1.
    # features_enzyme = sp.csr_matrix(SPM_m)
    features_enzyme = sp.csr_matrix(nM_graph)
    # ---------读取indication数据---------------------------------------------------------------------------
    # df = pd.read_csv("./indication/%s.csv" % (similarity))
    df = pd.read_csv("./indication/%s.csv" % (effect_name))
    row = []
    col = []
    data = []
    for i, j, z in zip(df['A'], df['B'], df['V']):
        row.append(i)
        col.append(j)
        data.append(z)
    nM_graph = sp.csr_matrix((data, (row, col)), shape=(548, 548))
    nM_graph = nM_graph.toarray() + np.identity(548)
    nM_graph = nM_graph + nM_graph.T
    nM_graph[nM_graph > 0] = 1.
    nM_graph[nM_graph < 0] = -1.
    # SPM_1 = copy.deepcopy(nM_graph)
    # SPM_m1 = copy.deepcopy(SPM_1)
    # SPM_m2 = np.dot(SPM_m1, SPM_1)
    # SPM_m3 = np.dot(SPM_m2, SPM_1)
    # SPM_m = SPM_m1 + SPM_m2 + SPM_m3
    #
    # SPM_m[SPM_m > 0] = 1.
    # SPM_m[SPM_m < 0] = -1.
    # features_indication = sp.csr_matrix(SPM_m)
    features_indication = sp.csr_matrix(nM_graph)
    # ---------读取sideeffect数据---------------------------------------------------------------------------
    # df = pd.read_csv("./sideeffect/%s.csv" % (similarity))
    df = pd.read_csv("./sideeffect/%s.csv" % (effect_name))
    row = []
    col = []
    data = []
    for i, j, z in zip(df['A'], df['B'], df['V']):
        row.append(i)
        col.append(j)
        data.append(z)
    nM_graph = sp.csr_matrix((data, (row, col)), shape=(548, 548))
    nM_graph = nM_graph.toarray() + np.identity(548)
    nM_graph = nM_graph + nM_graph.T
    nM_graph[nM_graph > 0] = 1.
    nM_graph[nM_graph < 0] = -1.
    # SPM_1 = copy.deepcopy(nM_graph)
    # SPM_m1 = copy.deepcopy(SPM_1)
    # SPM_m2 = np.dot(SPM_m1, SPM_1)
    # SPM_m3 = np.dot(SPM_m2, SPM_1)
    # SPM_m = SPM_m1 + SPM_m2 + SPM_m3
    #
    # SPM_m[SPM_m > 0] = 1.
    # SPM_m[SPM_m < 0] = -1.
    # features_sideeffect = sp.csr_matrix(SPM_m)
    features_sideeffect = sp.csr_matrix(nM_graph)
    # ---------读取transporter数据---------------------------------------------------------------------------
    # df = pd.read_csv("./transporter/%s.csv" % (similarity))
    df = pd.read_csv("./transporter/%s.csv" % (effect_name))
    row = []
    col = []
    data = []
    for i, j, z in zip(df['A'], df['B'], df['V']):
        row.append(i)
        col.append(j)
        data.append(z)
    nM_graph = sp.csr_matrix((data, (row, col)), shape=(548, 548))
    nM_graph = nM_graph.toarray() + np.identity(548)
    nM_graph = nM_graph + nM_graph.T
    nM_graph[nM_graph > 0] = 1.
    nM_graph[nM_graph < 0] = -1.
    # SPM_1 = copy.deepcopy(nM_graph)
    # SPM_m1 = copy.deepcopy(SPM_1)
    # SPM_m2 = np.dot(SPM_m1, SPM_1)
    # SPM_m3 = np.dot(SPM_m2, SPM_1)
    # SPM_m = SPM_m1 + SPM_m2 + SPM_m3
    #
    # SPM_m[SPM_m > 0] = 1.
    # SPM_m[SPM_m < 0] = -1.
    # features_transporter = sp.csr_matrix(SPM_m)
    features_transporter = sp.csr_matrix(nM_graph)
#----------------------------------------------开始训练---------------------------------------------------------------
    roc_score, aupr_sc, ap_sc, F1_sc = train(adj, features_enzyme, features_indication, features_sideeffect, features_transporter)
    # print(roc_score, aupr_sc, ap_sc, F1_sc)
    print('ave_roc_score=', "{:.5f}".format(roc_score),
          'ave_prc_score=', "{:.5f}".format(aupr_sc),
          'ave_ap_score=', "{:.5f}".format(ap_sc),
          'ave_F1_score=', "{:.5f}".format(F1_sc),
          )









