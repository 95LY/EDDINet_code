import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch.nn.parameter import Parameter
import torch
import math
import numpy as np
import tensorflow as tf
from initializations import *
_LAYER_UIDS = {}
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)
from layers import AvgReadout, Discriminator


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        self.weight = Parameter(torch.FloatTensor(out_ft, out_ft))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            # support = torch.mm(torch.squeeze(seq_fts, 0), self.weight)
            # out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(support, 0)), 0)
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
            # out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(out, 0)), 0)
            # out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(out, 0)), 0)
        else:
            out = torch.mm(adj, seq_fts)
            out = torch.mm(adj, out)
        if self.bias is not None:
            out += self.bias

        return self.act(out)



def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs

class BilinearDecoder(Layer):
    """Bilinear Decoder model layer for link prediction."""

    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid):
        super(BilinearDecoder, self).__init__()
        self.dropout = dropout
        self.act = act
        # with tf.variable_scope(self.name + '_vars'):
        with tf.variable_scope(self.name +'_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, input_dim, name="weights")

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1 - self.dropout)
        intermediate_product = tf.matmul(inputs, self.vars["weights"])
        x = tf.matmul(intermediate_product, tf.transpose(inputs))
        # x = tf.reshape(x, [-1])**
        # outputs = self.act(x)**
        return x  # outputs**

class MLP(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(MLP, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU()
        )
        self.linear_shortcut = nn.Linear(in_ft, out_ft)

    def forward(self, x):
        return self.ffn(x) + self.linear_shortcut(x)

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()

        self.gcn_enzyme = GCN(n_in, n_h, activation)
        self.gcn_indication = GCN(n_in, n_h, activation)
        self.gcn_sideeffect = GCN(n_in, n_h, activation)
        self.gcn_transporter = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

        # self.H = nn.Parameter(torch.rand(548, 16))
        self.H = nn.Parameter(torch.rand(1, 548, 16))




    def forward(self, seq1_enzyme, seq1_indication, seq1_sideeffect, seq1_transporter, seq2_enzyme, seq2_indication, seq2_sideeffect, seq2_transporter,adj, sparse, msk, samp_bias1, samp_bias2):
        # h_1 = self.gcn(seq1, adj)
        h_1_enzyme_1 = self.gcn_enzyme(seq1_enzyme, adj, sparse)
        h_1_indication_1 = self.gcn_indication(seq1_indication, adj, sparse)
        h_1_sideeffect_1 = self.gcn_sideeffect(seq1_sideeffect, adj, sparse)
        h_1_transporter_1 = self.gcn_transporter(seq1_transporter, adj, sparse)


        c_enzyme = self.read(h_1_enzyme_1, msk)
        c_indication = self.read(h_1_indication_1, msk)
        c_sideeffect = self.read(h_1_sideeffect_1, msk)
        c_transporter = self.read(h_1_transporter_1, msk)

        c_enzyme = self.sigm(c_enzyme)
        c_indication = self.sigm(c_indication)
        c_sideeffect = self.sigm(c_sideeffect)
        c_transporter = self.sigm(c_transporter)
        # print(c.size())

        h_2_enzyme_2 = self.gcn_enzyme(seq2_enzyme, adj, sparse)
        h_2_indication_2 = self.gcn_indication(seq2_indication, adj, sparse)
        h_2_sideeffect_2 = self.gcn_sideeffect(seq2_sideeffect, adj, sparse)
        h_2_transporter_2 = self.gcn_transporter(seq2_transporter, adj, sparse)



        ret_enzyme = self.disc(c_enzyme, h_1_enzyme_1, h_2_enzyme_2, samp_bias1, samp_bias2)
        ret_indication = self.disc(c_indication, h_1_indication_1, h_2_indication_2, samp_bias1, samp_bias2)
        ret_sideeffect = self.disc(c_sideeffect, h_1_sideeffect_1, h_2_sideeffect_2, samp_bias1, samp_bias2)
        ret_transporter = self.disc(c_transporter, h_1_transporter_1, h_2_transporter_2, samp_bias1, samp_bias2)
        # print(h_1_enzyme_1.size())
        h_1_all = []
        h_2_all = []
        h_1_all.append(h_1_enzyme_1)
        h_1_all.append(h_1_indication_1)
        h_1_all.append(h_1_sideeffect_1)
        h_1_all.append(h_1_transporter_1)
        h_2_all.append(h_2_enzyme_2)
        h_2_all.append(h_2_indication_2)
        h_2_all.append(h_2_sideeffect_2)
        h_2_all.append(h_2_transporter_2)
        h_1_all = torch.mean(torch.cat(h_1_all), 0).unsqueeze(0)
        h_2_all = torch.mean(torch.cat(h_2_all), 0).unsqueeze(0)
        # h_1_all = torch.mean(torch.cat(h_1_all), 0)
        # h_2_all = torch.mean(torch.cat(h_2_all), 0)
        # print(h_1_all.size())
        # H = torch.sum(torch.cat([h_1_all]), 0).unsqueeze(0)
        # H = h_1_all
        # print(self.H)
        pos_reg_loss = ((self.H - h_1_all) ** 2).sum()
        # print(pos_reg_loss)
        neg_reg_loss = ((self.H - h_2_all) ** 2).sum()
        # print(neg_reg_loss)
        # reg_loss = neg_reg_loss - pos_reg_loss
        reg_loss2 = pos_reg_loss - neg_reg_loss
        reg_loss = reg_loss2
        # print(reg_loss)

        return ret_enzyme, ret_indication, ret_sideeffect, ret_transporter, reg_loss

    # Detach the return variables
    def embed(self, seq1_enzyme, seq1_indication, seq1_sideeffect, seq1_transporter, adj, sparse):
        h_1_enzyme_1 = self.gcn_enzyme(seq1_enzyme, adj, sparse)
        h_1_indication_1 = self.gcn_indication(seq1_indication, adj, sparse)
        h_1_sideeffect_1 = self.gcn_sideeffect(seq1_sideeffect, adj, sparse)
        h_1_transporter_1 = self.gcn_transporter(seq1_transporter, adj, sparse)

        return h_1_enzyme_1, h_1_indication_1, h_1_sideeffect_1, h_1_transporter_1

