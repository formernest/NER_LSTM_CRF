#!env python
# -*- coding: UTF-8 -*-
import codecs
import pandas as pd
import numpy as np
import collections
from sklearn.model_selection import train_test_split
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import datetime
import torch.optim as optim

#############
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100  # 嵌入层维度
HIDDEN_DIM = 200  # 隐藏层数量
EPOCHS = 5  # 迭代几次
BATCH_SIZE = 64  # 每个batch的大小
SEQ_LEN = 60
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        torch.manual_seed(1)
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    @staticmethod
    def argmax(vec):
        # return the argmax as a python int
        _, idx = torch.max(vec, 1)
        return idx.item()

    @staticmethod
    def prepare_sequence(seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)

    # Compute log sum exp in a numerically stable way for the forward algorithm
    def log_sum_exp(self, vec):
        max_score = vec[0, self.argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + \
               torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = self.log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)  # batch_size*len(sentence)*embed_dim to len(sentence)*1*embedding_dim 1应该是batch_size
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)  # 初始化隐藏层,经过lstm输出和状态 len(sentence)*1*hidden_dim 1*1*hidden_size
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)  # lstm_out格式转化为len(sentence)*hidden_dim
        lstm_feats = self.hidden2tag(lstm_out)  # 线性转换hidden_dim to tag_size
        return lstm_feats   # len(sentence)*tag_size

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):  # len(sentence)*tag_size
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)   # 1*tagsize [-10000.,......]
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0  # [-10000.,....,0,-10000.]

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:  # len(sentence)*tag_size
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = self.argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = self.argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)   # len(sentence)*tag_size
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def batch_loss(self, sentences, labels):
        result = torch.zeros(len(sentences))
        for i in range(len(sentences)):
            result[i] = self.neg_log_likelihood(sentences[i], labels[i])
        return torch.mean(result)

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)  # len(sentence)*tag_size
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


class CommModelTrain():

    def __init__(self):
        self.word2id = dict()
        self.id2word = dict()
        self.tag2id = dict()
        self.id2tag = dict()
        data_path = './data/train'
        self.data2pkl(data_path)
        self.tag2id[START_TAG] = len(self.tag2id)
        self.tag2id[STOP_TAG] = len(self.tag2id)
        self.test_size = 0.1
        pass

    def flatten(self, x):
        result = []
        for el in x:
            if isinstance(x, collections.Iterable) and not isinstance(el, str):
                result.extend(self.flatten(el))
            else:
                result.append(el)
        return result

    def X_padding(self, words):
        ids = list(self.word2id[words])
        if len(ids) >= SEQ_LEN:
            return ids[:SEQ_LEN]
        ids.extend([0] * (SEQ_LEN - len(ids)))
        return ids

    def y_padding(self, tags):
        ids = list(self.tag2id[tags])
        if len(ids) >= SEQ_LEN:
            return ids[:SEQ_LEN]
        ids.extend([0] * (SEQ_LEN - len(ids)))
        return ids

    def calculate(self, x, y, res=[]):
        x = x.numpy()
        entity = []
        for j in range(len(x)):
            try:
                if x[j] == 0 or y[j] == 0:
                    continue
                if self.id2tag[y[j]][0] == 'B':
                    entity = [self.id2word[x[j]] + '/' + self.id2tag[y[j]]]
                elif self.id2tag[y[j]][0] == 'I' and len(entity) != 0 and entity[-1].split('/')[1][1:] == self.id2tag[y[j]][1:]:
                    entity.append(self.id2word[x[j]] + '/' + self.id2tag[y[j]])
                else:
                    if len(entity) != 0:
                        res.append(entity)
                        entity = []
            except Exception as e:
                if len(entity) != 0:
                    res.append(entity)
                    entity = []
        return res

    def calculate_batch(self, x, y, id2word, id2tag, res=[]):
        size = len(x)
        for i in range(size):
            res = (self.calculate(x[i], y[i], id2word, id2tag, res))
        return res

    def data2pkl(self, path):
        datas = list()
        labels = list()
        tags = set()
        # input_data = codecs.open('./data/train', 'r', 'utf-8')
        input_data = codecs.open(path, 'r', 'utf-8')
        for line in input_data.readlines():
            line = line.strip().split('\t')
            linedata = list(line[0].split())
            linelabel = list(line[1].split())
            set1 = set(linelabel)
            tags = tags.union(set1)
            datas.append(linedata)
            labels.append(linelabel)

        input_data.close()
        all_words = self.flatten(datas)   # 将二维的datas放入一维的all_words
        sr_allwords = pd.Series(all_words)    # 一维的标签矩阵，类似于字典   index:words
        sr_allwords = sr_allwords.value_counts()    # 统计字的词频，使词频高的在前 word:count
        set_words = sr_allwords.index     # 真正的words set
        set_ids = range(1, len(set_words) + 1)   # id范围

        tags = [i for i in tags]
        tag_ids = range(len(tags))
        self.word2id = pd.Series(set_ids, index=set_words)  # word:id   id从1开始
        self.id2word = pd.Series(set_words, index=set_ids)  # id:word
        self.tag2id = pd.Series(tag_ids, index=tags)        # tag:id id从0开始
        self.id2tag = pd.Series(tags, index=tag_ids)        # id:tag
        self.word2id["unknow"] = len(self.word2id) + 1

        df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))  # index:datas:labels
        df_data['x'] = df_data['words'].apply(self.X_padding)    # 对每一行的word进行word-id转换并填充,df_data['x']形式为index:x,object
        df_data['y'] = df_data['tags'].apply(self.y_padding)     # 对每一行的label进行tag_id转换并填充
        # df_data['x'] = self.X_padding(df_data['words'], self.word2id, SEQ_LEN)    # 对每一行的word进行word-id转换并填充,df_data['x']形式为index:x,object
        # df_data['y'] = self.y_padding(df_data['tags'], self.tag2id, SEQ_LEN)   # 对每一行的label进行tag_id转换并填充
        x = np.asarray(list(df_data['x'].values))    # 去掉index将df_data['x']转换为二维的list类型，line_count*max_len
        y = np.asarray(list(df_data['y'].values))    #
        # split train and test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=43)    # 划分训练集，测试集
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=43)  # 训练集，验证集

        # 序列化
        with open('./data/data.pkl', 'wb') as outp:
            pickle.dump(self.word2id, outp)
            pickle.dump(self.id2word, outp)
            pickle.dump(self.tag2id, outp)
            pickle.dump(self.id2tag, outp)
            pickle.dump(x_train, outp)
            pickle.dump(y_train, outp)
            pickle.dump(x_test, outp)
            pickle.dump(y_test, outp)
            pickle.dump(x_valid, outp)
            pickle.dump(y_valid, outp)
        #
        print('** Finished saving the data.')
        # return x_train, y_train, x_test, y_test

    def load_data(self):
        # 加载数据
        with open('./data/data.pkl', 'rb') as inp:
            word2id = pickle.load(inp)
            id2word = pickle.load(inp)
            tag2id = pickle.load(inp)
            id2tag = pickle.load(inp)
            x_train = pickle.load(inp)
            y_train = pickle.load(inp)
            x_test = pickle.load(inp)
            y_test = pickle.load(inp)
            # x_valid = pickle.load(inp)
            # y_valid = pickle.load(inp)
        # print("train len:", len(x_train))
        # print("test len:", len(x_test))
        # print("valid len", len(x_valid))

        # 将START_TAG和STOP_TAG加入tag字典中
        # tag2id[START_TAG] = len(tag2id)
        # tag2id[STOP_TAG] = len(tag2id)
        x_train = x_train[:10000]
        y_train = y_train[:10000]

        x_test = x_test[:1000]
        y_test = y_test[:1000]
        return x_train, y_train, x_test, y_test

    # train
    def fit(self):
        # path = './data/train'
        # word2id, id2word, tag2id, id2tag, x_train, y_train, x_test, y_test = self.data2pkl(path)
        x_train, y_train, x_test, y_test = self.load_data()
        model = BiLSTM_CRF(len(self.word2id) + 1, self.tag2id, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)  # 实例化模型
        optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=1e-4)  # 随机梯度下降优化算法
        # train_data
        sentence = torch.tensor(x_train, dtype=torch.long).to(DEVICE)
        y_train1 = [[self.tag2id[y_train[i][j]] for j in range(len(y_train[i]))] for i in range(len(y_train))]
        tags = torch.tensor(y_train1, dtype=torch.long).to(DEVICE)
        trainloader = DataLoader(TensorDataset(sentence, tags), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        # test_data
        sentence = torch.tensor(x_test, dtype=torch.long).to(DEVICE)
        y_test1 = [[self.tag2id[y_test[i][j]] for j in range(len(y_test[i]))] for i in range(len(y_test))]
        tags = torch.tensor(y_test1, dtype=torch.long).to(DEVICE)
        testloader = DataLoader(TensorDataset(sentence, tags), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        print("train start:", datetime.datetime.now())

        # 训练
        for epoch in range(EPOCHS):
            for batch, data in enumerate(trainloader, 0):
                sentences, labels = data
                model.zero_grad()  # 清空梯度
                loss = model.batch_loss(sentences, labels)
                loss.backward()
                optimizer.step()
                if batch % 20 == 0 and batch > 0:
                    print("epoch:", epoch, "batch:", batch, 'current time:', datetime.datetime.now())

            entityres = []
            entityall = []
            # 每个epoch后测试一下
            print('test start:', datetime.datetime.now())
            for batch, data in enumerate(testloader, 0):
                sentences, labels = data
                for sentence, label in zip(sentences, labels):
                    score, predict = model(sentence)
                    entityres = self.calculate(sentence, predict, self.id2word, self.id2tag, entityres)
                    entityall = self.calculate(sentence, label, self.id2word, self.id2tag, entityall)

            jiaoji = [i for i in entityres if i in entityall]
            if len(jiaoji) != 0:
                zhun = float(len(jiaoji)) / len(entityres)  # 准确率
                zhao = float(len(jiaoji)) / len(entityall)  # 召回率
                print("test:")
                print("precision:", zhun)
                print("recall:", zhao)
                print("F:", (2 * zhun * zhao) / (zhun + zhao))
            else:
                print("precision:", 0)
            print('test end:', datetime.datetime.now())
            path_name = "./model/model" + str(epoch) + ".pkl"
            print(path_name)
            torch.save(model, path_name)
            print("model has been saved")

        print('train finished:', datetime.datetime.now())

    # predict
    def predict(self, sentence, path):
        # word2id, id2word, tag2id, id2tag = self.load_date()
        path = './model/model.pkl'
        model = torch.load(path)
        sentence = self.X_padding(sentence)
        score, predict = model(sentence)
        res = self.calculate(sentence, predict, self.id2word, self.id2tag)
        print(res)


def applyModelTrain():
    train_model = CommModelTrain()
    train_model.fit()

if __name__ == '__main__':
    applyModelTrain()

