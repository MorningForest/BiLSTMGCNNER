# -*- coding: utf-8 -*-
# @Time    : 2021/1/4 14:28
# @Author  : hw
# @FileName: build_graph.py
# @Software: PyCharm

import codecs
import collections
import itertools
from math import log
import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix
import pickle

def get_co_occurrence_matrix(doc_train_list, word_id_map, word_freq, word_set):
    window_size = 10
    windows = []
    ## 滑动窗口
    wins = []
    for doc_words in doc_train_list:
        words = doc_words
        length = len(words)
        if length <= window_size:
            wins.append(words)
        else:
            # print(length, length - window_size + 1)
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                wins.append(window)
    ## 获取共现矩阵
    for doc_words in doc_train_list:
        words = doc_words.split(' ')
        length = len(words)
        if length <= window_size:
            # windows.append(words)
            pass
        else:
            # print(length, length - window_size + 1)
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]

                if len(window) == 2:
                    count_pair = (window[0], window[1])
                    # windows.append(window)
                    windows.append(count_pair)

    unqi_wind = set(windows)

    row = []
    col = []
    weight = []
    count = len(windows)
    counter = collections.Counter(windows)
    ## 计算单词之间边
    for word in itertools.chain(unqi_wind):
        nums = counter[word]
        word_i_id = word_id_map[word[0]]
        word_j_id = word_id_map[word[1]]
        word_freq_i = word_freq[word[0]]
        word_freq_j = word_freq[word[1]]
        # print(word_freq_i, word_freq_j, nums)
        pmi = log((1.0 * nums / count) /
                  (1.0 * word_freq_i * word_freq_j / (count * count)))
        if pmi < 0:
            continue
        row.append(word_i_id)
        col.append(word_j_id)
        weight.append(pmi)

    ### 获取句子和单词得关系
    # word_count = len(word_set)
    # for id, sentence in enumerate(doc_train_list):
    #     for word in sentence:
    #         row.append(word_count + id)
    #         col.append(word_id_map[word])
    #         weight.append(1.0)

    adj = coo_matrix((weight, (row, col)), shape=(len(word_id_map)+1, len(word_id_map)+1))
    ## 获取字符得特征
    word_feature_map = {}
    words_list = []
    with codecs.open('char_word2vec_size300_win5.txt', 'r', 'utf8') as fp:
        for line in fp:
            line = line.split(' ')
            embedd = np.asarray(line[1:], dtype='float32')
            word_feature_map[line[0]] = embedd
            words_list.append(line[0])
        fp.close()
    feature = []
    feature.append(np.random.normal(0, 1.0, 300))
    for word, id in word_id_map.items():
        if word in words_list:
            feature.append(word_feature_map[word])
        else:
            feature.append(np.random.normal(0, 1.0, 300))

    # for _ in range(len(doc_train_list)):
    #     feature.append(np.random.normal(0, 1.0, 300))

    features = np.array(feature)
    return features, adj


def build_graph():
    ### 获取文档句子
    sentences = []
    with open("Genia4ERtask.txt", encoding='utf8') as fp:
        for line in fp:
            if len(line.strip()) > 0:
                # line = re.split('[，。！？、‘’“”:]/[O]', ''.join(item for item in line.strip('\n')))
                # print(line[0])
                sentences.append(line.strip())

    ### 获取字符集合和词频
    word_freq = {}
    word_set = set()
    for doc_words in sentences:
        for word in doc_words.split(' '):
            word_set.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    # with open('../renmindata.pkl', 'rb') as fp:
    #     word2id = pickle.load(fp)
    with open('../Genia4ERdata.pkl', 'rb') as fp:
        word2id = pickle.load(fp)
    word_id_map = {}
    for word, id in word2id.items():
        word_id_map[word] = id + 1
    for id, word in enumerate(word_set):
        if word_id_map.get(word, -1) == -1:
            word_id_map[word] = len(word_id_map) + 1
    print(len(word_id_map))
    # id_word_map = {id: word for id, word in enumerate(word_set)}
    features, adj = get_co_occurrence_matrix(sentences, word_id_map, word_freq, word_set)
    with open('../gcndata.pkl', 'wb') as outp:
        pickle.dump(features, outp)
        pickle.dump(adj, outp)
    return features, adj.toarray()
    # word_size = len(word_set)
    # all_size = len(sentences)
    # train_size = int(all_size * 0.6)
    # valid_size = int(all_size * 0.2)
    # test_size = all_size - train_size - valid_size
    # # ids_train = range(word_size, word_size+train_size)
    # return features, adj.toarray(), (train_size, valid_size, test_size), word_id_map, word_size

if __name__ == '__main__':
    feature, adj = build_graph()
    print(feature.shape, adj.shape)