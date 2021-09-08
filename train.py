# -*- coding: utf-8 -*-
# @Time    : 2021/1/5 15:40
# @Author  : hw
# @FileName: train.py
# @Software: PyCharm

import pickle
import pdb
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from config import opt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from model.LSTM import NERLSTM
from model.LSTM_CRF import NERLSTM_CRF
from model.LSTM_CRF_GCN import NERLSTM_CRF_GCN
import prettytable as pt
from data import batch_yield, read_corpus, pad_sequences, tag2label

with open('data/word2id.pkl', 'rb') as fp:
    word2id = pickle.load(fp)
with open('data/vocab.pkl', 'rb') as fp:
    vocab = pickle.load(fp)
train_data = read_corpus('data/train_data')
test_data = read_corpus('data/test_data')
test_size = len(test_data)

model = NERLSTM_CRF_GCN[opt.model](opt.embedding_dim, opt.hidden_dim, opt.dropout, word2id, tag2label).cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

for epoch in range(opt.max_epoch):
    model.train()
    for step, (seqs, labels) in enumerate(batch_yield(train_data, opt.batch_size, word2id, tag2label)):
        optimizer.zero_grad()
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)
        labels, labels_len = pad_sequences(labels, pad_mark=0)
        X = torch.tensor(word2id, dtype=torch.long).cuda()
        y = torch.tensor(labels, dtype=torch.long).cuda()
        feature, adj = torch.tensor(feature, dtype=torch.float).cuda(), torch.tensor(adj, dtype=torch.float).cuda()
        # print(X.shape)
        # CRF
        loss = model.log_likelihood(X, y, (feature, adj))
        loss.backward()
        # CRF
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)

        optimizer.step()
        if index % 200 == 0:
            print('epoch:%04d,------------loss:%f' % (epoch, loss.item()))

    aver_loss = 0
    preds, labels = [], []
    for index, batch in enumerate(valid_dataloader):
        model.eval()
        val_x, val_y = torch.tensor(batch['x'], dtype=torch.long).cuda(), torch.tensor(batch['y'],
                                                                                       dtype=torch.long).cuda()
        predict = model(val_x, (feature, adj))
        # CRF
        loss = model.log_likelihood(val_x, val_y, (feature, adj))
        aver_loss += loss.item()
        # 统计非0的，也就是真实标签的长度
        leng = []
        for i in val_y.cpu():
            tmp = []
            for j in i:
                if j.item() > 0:
                    tmp.append(j.item())
            leng.append(tmp)

        for index, i in enumerate(predict):
            preds += i[:len(leng[index])]

        for index, i in enumerate(val_y.tolist()):
            labels += i[:len(leng[index])]
    aver_loss /= (len(valid_dataloader) * 64)
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    print(precision, recall)
    f1 = f1_score(labels, preds, average='macro')
    # report = classification_report(labels, preds)
    # print(report)
    tb = pt.PrettyTable()
    tb.field_names = ["精确度", "召回率", "f1分数"]
    tb.add_row([precision, recall, f1])
    print(tb)