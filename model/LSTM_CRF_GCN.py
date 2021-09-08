# -*- coding: utf-8 -*-
# @Time    : 2021/1/4 14:16
# @Author  : hw
# @FileName: LSTM_CRF_GCN.py
# @Software: PyCharm
import torch
from torch import nn
from torch.nn import functional as F
from torchcrf import CRF
from .Attention import MultiHeadSelfAttention

def sparse_dropout(x, rate, noise_shape):
    """

    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte()
    i = x._indices() # [2, 49216]
    v = x._values() # [49216]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1./ (1-rate))

    return out


def dot(x, y, sparse=False):
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.mm(x, y)

    return res

class GraphConvolution(nn.Module):


    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation=F.relu,
                 featureless=False):
        super(GraphConvolution, self).__init__()


        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.num_features_nonzero = num_features_nonzero

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))


    def forward(self, inputs):
        # print('inputs:', inputs)
        x, support = inputs[0], inputs[1]
        if self.training and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        elif self.training:
            x = F.dropout(x, self.dropout)

        # convolve
        if not self.featureless: # if it has features x
            if self.is_sparse_inputs:
                xw = torch.sparse.mm(x, self.weight)
            else:
                xw = torch.mm(x, self.weight)
        else:
            xw = self.weight
        out = torch.mm(support, xw)

        if self.bias is not None:
            out += self.bias

        return self.activation(out), support


class NERLSTM_CRF_GCN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout, word2id, tag2id):
        super(NERLSTM_CRF_GCN, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2id) + 1
        self.tag_to_ix = tag2id
        self.tagset_size = len(tag2id)

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # CRF
        # self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True,
        #                     batch_first=False)

        self.lstm = nn.GRU(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True,
                            batch_first=False)
        #
        # self.hidden2tag = nn.Linear(self.hidden_dim+100, self.tagset_size)
        self.fct = nn.Linear(self.hidden_dim+100, 100)
        self.hidden2tag = nn.Linear(100, self.tagset_size)
        # self.hidden2tag = nn.Linear(self.hidden_dim+100, self.tagset_size)

        self.crf = CRF(self.tagset_size)

        ### GCN
        input_dim = 300
        self.gcn_hidden_dim = hidden_dim
        self.layers = nn.Sequential(GraphConvolution(input_dim, self.gcn_hidden_dim, num_features_nonzero=0,
                                                     activation=F.relu,
                                                     dropout=dropout,
                                                     is_sparse_inputs=False),

                                    GraphConvolution(self.gcn_hidden_dim, self.gcn_hidden_dim-100, num_features_nonzero=0,
                                                     activation=F.relu,
                                                     dropout=dropout,
                                                     is_sparse_inputs=False),

                                    )

        ### 注意力机制
        # self.atten = MultiHeadSelfAttention(self.hidden_dim+100, self.hidden_dim+100, self.hidden_dim+100, 6)


    def forward(self, x, inputs):
        # CRF
        x = x.transpose(0, 1)
        batch_size = x.size(1)
        sent_len = x.size(0)
        feature, support = inputs

        embedding = self.word_embeds(x)
        # gcn_embed = self.layers((feature, support))[0]
        # gcn_embed = gcn_embed[x]
        # embedding = torch.cat([embedding, gcn_embed], -1)

        outputs, hidden = self.lstm(embedding)
        gcn_embed = self.layers((feature, support))[0]
        gcn_embed = gcn_embed[x]
        outputs = torch.cat([outputs, gcn_embed], -1)

        outputs = self.dropout(outputs)
        # outputs = self.atten(outputs)
        outputs = self.fct(outputs)
        outputs = self.hidden2tag(outputs)
        # CRF
        outputs = self.crf.decode(outputs)
        return outputs

    def log_likelihood(self, x, tags, inputs):
        x = x.transpose(0, 1)
        batch_size = x.size(1)
        sent_len = x.size(0)
        tags = tags.transpose(0, 1)
        feature, support = inputs[0], inputs[1]

        embedding = self.word_embeds(x)

        # gcn_embed = self.layers((feature, support))[0]
        # gcn_embed = gcn_embed[x]
        # embedding = torch.cat([embedding, gcn_embed], -1)

        outputs, hidden = self.lstm(embedding)

        gcn_embed = self.layers((feature, support))[0]
        gcn_embed = gcn_embed[x]
        outputs = torch.cat([outputs, gcn_embed], -1)

        outputs = self.dropout(outputs)
        # outputs = self.atten(outputs)

        outputs = self.fct(outputs)


        outputs = self.hidden2tag(outputs)
        return - self.crf(outputs, tags)