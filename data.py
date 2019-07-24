#!env python
# -*- coding: UTF-8 -*-

import codecs
import pandas as pd
import numpy as np
import re
import collections
import random


def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def data2pkl():
    datas = list()
    labels = list()
    tags = set()
    input_data = codecs.open('./data/train', 'r', 'utf-8')
    for line in input_data.readlines():
        line = line.strip().split('\t')
        linedata = list(line[0].split())
        linelabel = list(line[1].split())
        set1 = set(linelabel)
        tags = tags.union(set1)
        datas.append(linedata)
        labels.append(linelabel)

    input_data.close()
    all_words = flatten(datas)   # 将二维的datas放入一维的all_words
    sr_allwords = pd.Series(all_words)    # 一维的标签矩阵，类似于字典   index:words
    sr_allwords = sr_allwords.value_counts()    # 统计字的词频，使词频高的在前 word:count
    set_words = sr_allwords.index     # 真正的words set
    set_ids = range(1, len(set_words) + 1)   # id范围

    tags = [i for i in tags]
    tag_ids = range(len(tags))
    word2id = pd.Series(set_ids, index=set_words)  # word:id   id从1开始
    id2word = pd.Series(set_words, index=set_ids)  # id:word
    tag2id = pd.Series(tag_ids, index=tags)        # tag:id id从0开始
    id2tag = pd.Series(tags, index=tag_ids)        # id:tag

    word2id["unknow"] = len(word2id) + 1
    # print(word2id)
    max_len = 60

    def X_padding(words):
        ids = list(word2id[words])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0] * (max_len - len(ids)))
        return ids

    def y_padding(tags):
        ids = list(tag2id[tags])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0] * (max_len - len(ids)))
        return ids

    df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))  # index:datas:labels
    df_data['x'] = df_data['words'].apply(X_padding)    # 对每一行的word进行word-id转换并填充,df_data['x']形式为index:x,object
    df_data['y'] = df_data['tags'].apply(y_padding)     # 对每一行的label进行tag_id转换并填充
    x = np.asarray(list(df_data['x'].values))    # 去掉index将df_data['x']转换为二维的list类型，line_count*max_len
    y = np.asarray(list(df_data['y'].values))    #

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=43)    # 划分训练集，测试集
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=43)  # 训练集，验证集

    import pickle
    with open('./data/data.pkl', 'wb') as outp:  # 序列化
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


# 对文本进行标注，实体标注为BIE的格式，其他标注为0
def origin2tag():
    input_data = codecs.open('./origindata.txt', 'r', 'utf-8')
    output_data = codecs.open('./wordtag.txt', 'w', 'utf-8')
    for line in input_data.readlines():
        line = line.strip()
        i = 0
        while i < len(line):
            if line[i] == '{':
                i += 2
                temp = ""
                while line[i] != '}':
                    temp += line[i]
                    i += 1
                i += 2
                word = temp.split(':')
                sen = word[1]
                output_data.write(sen[0] + "/B_" + word[0] + " ")
                for j in sen[1:len(sen) - 1]:
                    output_data.write(j + "/M_" + word[0] + " ")
                output_data.write(sen[-1] + "/E_" + word[0] + " ")
            else:
                output_data.write(line[i] + "/O ")
                i += 1
        output_data.write('\n')
    input_data.close()
    output_data.close()


def tagsplit():
    with open('./wordtag.txt', 'rb') as inp:
        texts = inp.read().decode('utf-8')
    sentences = re.split('[，。！？、‘’“”（）]/[O]', texts)
    output_data = codecs.open('./wordtagsplit.txt', 'w', 'utf-8')
    for sentence in sentences:
        if sentence != " ":
            output_data.write(sentence.strip() + '\n')
    output_data.close()

def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        train_batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp

        yield train_batch

# origin2tag()
# tagsplit()
data2pkl()
