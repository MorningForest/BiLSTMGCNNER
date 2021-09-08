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
from tensorboardX import SummaryWriter

with open(opt.pickle_path, 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)
        x_valid = pickle.load(inp)
        y_valid = pickle.load(inp)
print("train len:", len(x_train))
print("test len:", len(x_test))
print("valid len", len(x_valid))
print("word nums", len(word2id))
print(word2id)
print(tag2id)

with open('gcndata.pkl', 'rb') as inp:
    feature = pickle.load(inp)
    adj = pickle.load(inp).toarray()

print('feature', feature.shape)
print('adj', adj.shape)

class NERDataset(Dataset):

    def __init__(self, X, Y, *args, **kwargs):
        self.data = [{'x': X[i], 'y': Y[i]} for i in range(X.shape[0])]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

train_dataset = NERDataset(x_train, y_train)
valid_dataset = NERDataset(x_valid, y_valid)
test_dataset = NERDataset(x_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory= True)
valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory= True)
test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory= True)


models = {'NERLSTM': NERLSTM,
          'NERLSTM_CRF': NERLSTM_CRF,
          'NERLSTM_CRF_GCN': NERLSTM_CRF_GCN}

model = models[opt.model](opt.embedding_dim, opt.hidden_dim, opt.dropout, word2id, tag2id).cuda()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
writer = SummaryWriter(log_dir='result')

if opt.model == 'NERLSTM':
    for epoch in range(opt.max_epoch):
        model.train()
        # for batch in enumerate(train_dataloader):
        for index, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            X = batch['x'].cuda()
            y = batch['y'].cuda()
            print(type(X))
            y = y.view(-1, 1)
            y = y.squeeze(-1)
            pred = model(X)
            print(pred.shape)
            pred = pred.view(-1, pred.size(-1))
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            if index % 200 == 0:
                print('epoch:%04d,------------loss:%f'%(epoch,loss.item()))

        aver_loss = 0
        preds, labels = [], []
        for index, batch in enumerate(valid_dataloader):
            model.eval()
            val_x,val_y = batch['x'].cuda(), batch['y'].cuda()
            predict = model(val_x)
            predict = torch.argmax(predict, dim=-1)
            if index % 500 == 0:
                print([id2word[i.item()] for i in val_x[0].cpu() if i.item()>0])
                length = [id2tag[i.item()] for i in val_y[0].cpu() if i.item()>0]
                print(length)
                print([id2tag[i.item()] for i in predict[0][:len(length)].cpu() if i.item()>0])
            
            # 统计非0的，也就是真实标签的长度
            leng = []
            for i in val_y.cpu():
                tmp = []
                for j in i:
                    if j.item()>0:
                        tmp.append(j.item())
                leng.append(tmp)

            # 提取真实长度的预测标签
            for index, i in enumerate(predict.tolist()):
                preds.extend(i[:len(leng[index])])

            # 提取真实长度的真实标签
            for index, i in enumerate(val_y.tolist()):
                labels.extend(i[:len(leng[index])])
            
        precision = precision_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        f1 = f1_score(labels, preds, average='macro')
        report = classification_report(labels, preds)
        print(report)

elif opt.model == 'NERLSTM_CRF':
    for epoch in range(opt.max_epoch):
        model.train()
        for index, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            X = torch.tensor(batch['x'], dtype=torch.long).cuda()
            y = torch.tensor(batch['y'], dtype=torch.long).cuda()
            # print(X.shape)
            #CRF
            loss = model.log_likelihood(X, y)
            loss.backward()
            #CRF
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)

            optimizer.step()
            if index % 200 == 0:
                print('epoch:%04d,------------loss:%f'%(epoch,loss.item()))

        aver_loss = 0
        preds, labels = [], []
        for index, batch in enumerate(valid_dataloader):
            model.eval()
            val_x, val_y = torch.tensor(batch['x'], dtype=torch.long).cuda(), torch.tensor(batch['y'], dtype=torch.long).cuda()
            predict = model(val_x)
            #CRF
            loss = model.log_likelihood(val_x, val_y)
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
        f1 = f1_score(labels, preds, average='macro')
        report = classification_report(labels, preds)
        print(report)

elif opt.model == 'NERLSTM_CRF_GCN':
    for epoch in range(opt.max_epoch):
        model.train()
        for index, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            X = torch.tensor(batch['x'], dtype=torch.long).cuda()
            y = torch.tensor(batch['y'], dtype=torch.long).cuda()

            feature, adj = torch.tensor(feature, dtype=torch.float).cuda(), torch.tensor(adj, dtype=torch.float).cuda()
            # print(X.shape)
            #CRF
            loss = model.log_likelihood(X, y, (feature, adj))
            loss.backward()
            #CRF
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)

            optimizer.step()
            if index % 200 == 0:
                print('epoch:%04d,------------loss:%f' % (epoch, loss.item()))

        aver_loss = 0
        preds, labels = [], []
        for index, batch in enumerate(valid_dataloader):
            model.eval()
            val_x, val_y = torch.tensor(batch['x'], dtype=torch.long).cuda(), torch.tensor(batch['y'], dtype=torch.long).cuda()
            predict = model(val_x, (feature, adj))
            #CRF
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

        with open('result.txt', 'a+') as fp:
            fp.write(str(epoch)+'\t'+str(precision)+'\t'+str(recall)+'\t'+str(f1)+'\t'+str(aver_loss)+'\n')



        writer.add_scalar('precision', precision, epoch)
        writer.add_scalar('recall', recall, epoch)
        writer.add_scalar('f1', f1, epoch)
        # report = classification_report(labels, preds)
        # print(report)
        tb = pt.PrettyTable()
        tb.field_names = ["精确度", "召回率", "f1分数"]
        tb.add_row([precision, recall, f1])
        print(tb)
        # precision = precision_score(labels, preds, average='micro')
        # recall = recall_score(labels, preds, average='micro')
        # f1 = f1_score(labels, preds, average='micro')
        # report = classification_report(labels, preds)
        # print(report)
    aver_loss = 0
    preds, labels = [], []
    for index, batch in enumerate(test_dataloader):
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
    with open('result.txt', 'a+') as fp:
        fp.write("测试数据\n")
        fp.write(
            str(precision) + '\t' + str(recall) + '\t' + str(f1) + '\t' + str(aver_loss) + '\n')

    # # report = classification_report(labels, preds)
    # # print(report)
    tb = pt.PrettyTable()
    tb.field_names = ["test精确度", "test召回率", "testf1分数"]
    tb.add_row([precision, recall, f1])
    print(tb)
    torch.save(model, 'result/model.pkl')

writer.close()
