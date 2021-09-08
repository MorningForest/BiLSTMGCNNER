# -*- coding: utf-8 -*-
# @Time    : 2021/1/8 19:24
# @Author  : hw
# @FileName: data_process.py
# @Software: PyCharm

import codecs
import pandas as pd
import numpy as np
import collections
from sklearn.model_selection import train_test_split
import pickle

sentences = []
labels = []
max_len = 20

with codecs.open('files/Genia4ERtask1.iob2', 'r', encoding='utf8') as fp:
    sent, label = [], []
    for line in fp:
        line = line.strip('\n').strip().split()
        if len(line) > 0:
            sent.append(line[0])
            label.append(line[1])
        else:
            sentences.append(sent)
            labels.append(label)
            sent, label = [], []
test_count = 0
with codecs.open('files/Genia4EReval2.iob2', 'r', 'utf8') as fp:
    sent, label = [], []
    for line in fp:
        line = line.strip('\n').strip().split()
        if len(line) == 2:
            sent.append(line[0])
            label.append(line[1])
        elif len(line) == 0:
            sentences.append(sent)
            labels.append(label)
            sent, label = [], []
            test_count += 1


with open("Genia4ERtask.txt", 'w', encoding='utf8') as fp:
    for sent in sentences:
        copous = ''
        for item in sent:
            copous += item + ' '
        copous += '\n'
        fp.write(copous)
def flat_gen(x):
    def iselement(e):
        return not(isinstance(e, collections.Iterable) and not isinstance(e, str))
    for el in x:
        if iselement(el):
            yield el
        else:
            yield from flat_gen(el)

def X_padding(words):
    ids = list(word2id[words])
    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids)))
    return ids

def y_padding(tags):
    ids = list(tag2id[tags])
    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids)))
    return ids
### get word2id
all_words = [i for i in flat_gen(sentences)]
sr_allwords = pd.Series(all_words)
sr_allwords = sr_allwords.value_counts()
set_words = sr_allwords.index
set_ids = range(1, len(set_words)+1)
word2id = pd.Series(set_ids, index=set_words)
id2word = pd.Series(set_words, index=set_ids)
word2id["unknow"] = len(word2id)+1
id2word[len(word2id)] = "unknow"
### get tag2id
tags = set(i for i in flat_gen(labels))
tags = [i for i in tags]
tag_ids = range(len(tags))
tag2id = pd.Series(tag_ids, index=tags)
id2tag = pd.Series(tags, index=tag_ids)


### get word, label
df_data = pd.DataFrame({'words': sentences, 'tags': labels}, index=range(len(sentences)))
df_data['x'] = df_data['words'].apply(X_padding)
df_data['y'] = df_data['tags'].apply(y_padding)
x = np.asarray(list(df_data['x'].values))
y = np.asarray(list(df_data['y'].values))
x_test, y_test = x[-test_count:], y[-test_count:]
x, y = x[:-test_count], y[:-test_count]
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=43)
# x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,  test_size=0.2, random_state=43)
print("x_train", x_train.shape)
print("x_test", x_test.shape)
print("y_train", y_train.shape)
print("y_test", y_test.shape)

with open('../Genia4ERdata.pkl', 'wb') as outp:
    pickle.dump(word2id, outp)
    pickle.dump(id2word, outp)
    pickle.dump(tag2id, outp)
    pickle.dump(id2tag, outp)
    pickle.dump(x_train, outp)
    pickle.dump(y_train, outp)
    pickle.dump(x_test, outp)
    pickle.dump(y_test, outp)
    pickle.dump(x_valid, outp)
    pickle.dump(y_valid, outp)

print('** Finished saving the data.')






