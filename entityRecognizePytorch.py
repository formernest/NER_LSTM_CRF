# -*- coding:utf-8 -*-
# create on 2018.09.27
#
import os
import re
import sys
import jieba
import traceback
import time
import math
import numpy as np
import pandas as pd
import source.util as util
import source.log as log
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from gensim.corpora import Dictionary
from pypinyin import lazy_pinyin
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from Levenshtein import distance
from random import shuffle, choice
from source.api import ModelTrain
from source.api import dic2str


class Cnn(nn.Module):
    def __init__(self, embed_mat, class_num):
        super(Cnn, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()
        self.win_len = 7
        self.class_num = class_num
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.conv = nn.Conv1d(embed_len, 128, kernel_size=self.win_len, padding=0)
        self.gate = nn.Conv1d(embed_len, 128, kernel_size=self.win_len, padding=0)
        self.la = nn.Sequential(nn.Linear(128, 200),
                                nn.ReLU())
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(200, class_num))

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 2, 1)
        g = torch.sigmoid(self.gate(x))
        x = self.conv(x)
        x = x * g
        x = x.permute(0, 2, 1)
        x = self.la(x)
        return self.dl(x)


class Rnn(nn.Module):
    def __init__(self, embed_mat, class_num):  # embed_mat：字tensor，seq_len：每句50个字，class_num：5个分类
        super(Rnn, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()  # 字数，每个字的维度：每个字200维
        self.class_num = class_num
        self.embed = nn.Embedding(self.vocab_num, self.embed_len, _weight=embed_mat)  # 嵌入层
        self.hidden_size = 200
        self.ra = nn.LSTM(self.embed_len, 200, batch_first=True, bidirectional=True)  # 输入特征维度为字向量的维度,隐状态特征维度200，1个lstm层，batch_first=True表示顺序[batch_size, time_step, input_size]，双向
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(400, self.class_num))  # Sequential时序容器，Linear：线性变换y=Ax+b，全连接层

    def forward(self, x):  # 每次执行的 计算步骤 二维，32行，每行50个字
        x = self.embed(x)  # 转向量的张量 三维，32行，每行50个字，每个字200维
        h, hc_n = self.ra(x)  # 输入：input（batch_size, time_step, input_size），(h_0, c_0)  输出：output , (h_n隐藏层状态, c_n记忆层状态)
        return self.dl(h)

    def attention(self, hidden, encoder_outputs, encoder_maskings):
        """
        hidden : 1,B,D
        encoder_outputs : B,T,D
        encoder_maskings : B,T
        output : B,1,D
        """
        # (1,B,D) -> (B,D,1)
        hidden = hidden.squeeze(0).unsqueeze(2)
        # B
        batch_size = encoder_outputs.size(0)
        # T
        max_len = encoder_outputs.size(1)
        # (B*T,D) -> (B*T,D)
        energies = self.attn(encoder_outputs.contiguous().view(batch_size * max_len, -1))
        # (B*T,D) -> B,T,D
        energies = energies.view(batch_size, max_len, -1)
        # (B,T,D) * (B,D,1) -> (B,1,T)
        attn_energies = energies.bmm(hidden).transpose(1, 2)
        # PAD masking
        attn_energies = attn_energies.squeeze(1).masked_fill(encoder_maskings, -1e12)

        # B,T
        alpha = F.softmax(attn_energies, dim=1)
        # B,1,T
        alpha = alpha.unsqueeze(1)
        # B,1,T * B,T,D => B,1,D
        context = alpha.bmm(encoder_outputs)

        return context

class CommModelTrain(ModelTrain):
    def __init__(self, nntype, datapath, threshold, modelparmvalue, trainloger):
        self.type = nntype       # model类型
        self.datapath = datapath  # 数据路径
        self.threshold = '0.85|32|10'.split('|')  # 阈值
        self.modelparmvalue = modelparmvalue  # 模型参数
        self.trainloger = trainloger  # 训练日志
        self.intent_sentences = {}  # 意图
        self.featureWordsWeight = None  # 特征权重
        self.include_intents_list = list()
        self.levenshtein_threshold = 0.40  # 编辑距离阈值
        self.jaccard_threshold = 0.45
        self.embed_len = 200  # 字维度？
        self.max_vocab = 5000  # 最高词频
        self.min_freq = 1  # 最低词频
        self.seq_len = 50  # 每句50字
        self.lr = 1e-3  # 学习率
        self.validation_split = 0.2  # 验证集
        self.ind_label = dict()
        self.word_inds = None  # word-id
        self.embed_mat = None   # 字向量
        self.load_model = None  # 使用的模型
        self.label_inds = None
        if threshold != None and len(threshold) > 0:
            self.threshold = threshold.split('|')
        self.stopwords_re, self.homonymwords, self.synonymwords = util.load_model_depend_file(self.datapath)
        if self.type == '5.0.1' or self.type == '5.0.2':
            if len(self.threshold) != 3:
                self.__err('invalid threshold:', self.threshold)
        if self.modelparmvalue is not None and type(self.modelparmvalue) == type({}):
            if self.modelparmvalue.get('class_lr', None) is not None:
                self.lr = float(self.modelparmvalue['class_lr'])
            if self.modelparmvalue.get('class_validation_split', None) is not None:
                self.validation_split = float(self.modelparmvalue['class_validation_split'])
            if self.modelparmvalue.get('class_embed_len', None) is not None:
                self.embed_len = int(self.modelparmvalue['class_embed_len'])
            if self.modelparmvalue.get('class_max_vocab', None) is not None:
                self.max_vocab = int(self.modelparmvalue['class_max_vocab'])
            if self.modelparmvalue.get('class_seq_len', None) is not None:
                self.seq_len = int(self.modelparmvalue['class_seq_len'])

    def __log(self, *args):
        info = 'entityRecognize_pytorch '
        for a in args:
            info += str(a)
        log.log(info)
        if self.trainloger is not None:
            self.trainloger.info(info)

    def __err(self, *args):
        info = 'entityRecognize_pytorch '
        for a in args:
            info += str(a)
        log.err(info)
        if self.trainloger is not None:
            self.trainloger.info(info)

    def fit(self):
        try:
            word_vecs = util.load_model_word_vec(self.datapath)
            line_list, label_list = self._generate(self._loadsyntax(), self._loadslots(), 5000)
            self.__log('########line_list', line_list)
            self.__log('########label_list', label_list)

            self.__id2vecs(word_vecs, line_list)
            self.__label2ind(label_list)
            pad_seqs, inds = self.__align(line_list, label_list, False)    # line index padding, label index padding
            self.__compile_fit(int(self.threshold[1]), int(self.threshold[2]), pad_seqs, inds)
            return True, set(), set(), [], ''
        except Exception as e:
            self.__err(str(Exception), ':', repr(e), '\n', traceback.format_exc())
            return False, None

    def load(self):
        self.device = torch.device('cpu')
        if self.type == '5.0.1':
            if self.load_model is None:
                self.load_model = torch.load(self.datapath + '/cnnclass.pkl')
        elif self.type == '5.0.2':
            if self.load_model is None:
                self.load_model = torch.load(self.datapath + '/rnnclass.pkl')

    def predict(self, content):
        try:
            text = content.strip()
            pad_seq = self.__sent2ind(list(text), keep_oov=True)
            sent = torch.LongTensor([pad_seq]).to(self.device)

            with torch.no_grad(): #在上下文环境中切断梯度计算
                self.load_model.eval()  # 切换到测试模式
                probs = F.softmax(self.load_model(sent), dim=-1)  # softmax转概率分布输出
            probs = probs.numpy()[0]
            inds = np.argmax(probs, axis=1)
            preds = [self.ind_label[ind] for ind in inds[-len(text):]]
            pairs = list()
            for word, pred in zip(text, preds):
                pairs.append((word, pred))

            label = ''
            entity = ''
            entity_list = []
            intent_slots = []
            for id in range(len(pairs)):
                if self.label_inds.get(pairs[id][1], None):
                    label = re.findall('.-(.*)',pairs[id][1])
                    if re.findall('B-(.*)', pairs[id][1]):
                        if len(entity) > 0:
                            entity_list.append(entity + ':' + intent_slots[-1])
                        entity = pairs[id][0]
                        intent_slots.append(label[0])
                    elif re.findall('I-(.*)', pairs[id][1]):
                        entity += pairs[id][0]
            if len(entity) > 0:
                entity_list.append(entity + ':' + intent_slots[-1])

            predict = {}
            predict['content'] = content
            if len(intent_slots) == 0:
                predict['thresholdintent'] = '其他'
                predict['maxprob'] = 0.0
            else:
                predict['thresholdintent'] = '|'.join(intent_slots)
                predict['entity'] = '|'.join(entity_list)
                predict['pairs'] = str(pairs)
                predict['maxprob'] = 1.0

            return predict
        except Exception as e:
            self.__err(str(Exception),':', repr(e), '\n', traceback.format_exc())
            return None

    def _loadsyntax(self):
        temps = list()
        pathfile = os.path.join(self.datapath + '/syntax/template.txt')
        if os.path.exists(pathfile) == True:
            with open(pathfile) as f:
                for line in f:
                    if len(line) == 0:
                        continue
                    parts = line.strip().split()
                    temps.append(parts)
        self.__log('#########temps',temps)
        return temps

    def _loadslots(self):
        entity_dic = {}
        files = os.listdir(os.path.join(self.datapath, 'slot'))
        for file in files:
            slot = os.path.splitext(file)[0]
            entity_dic[slot] = self._loadslot(self.datapath+'/slot/' + file)
        self.__log('###########entity_dic:',entity_dic)
        return entity_dic

    def _loadslot(self, pathfile):
        slot_entity_list = []
        entity_strs = pd.read_csv(pathfile).values
        for entity_str in entity_strs:
            entity_str0 = '' if pd.isnull(entity_str[0]) else entity_str[0]
            entity_str1 = '' if pd.isnull(entity_str[1]) else entity_str[1]
            if len(entity_str1) > 0:
                slot_entity_list.append([entity_str0,entity_str1.strip().split()])
            else:
                slot_entity_list.append([entity_str0, None])
        return slot_entity_list

    def _generate(self, temps, slots, num):
        word_mat = []
        label_mat = []
        for i in range(num):
            parts = choice(temps)
            words, labels = list(), list()
            for part in parts:
                if slots.get(part, None):
                    entity = choice(slots[part])
                    words.extend(entity[0])
                    labels.append('B-' + part)
                    if len(entity[0]) > 1:
                        labels.extend(['I-' + part] * (len(entity[0]) - 1))
                else:
                    words.extend(part)
                    labels.extend(['O'] * len(part))
            word_mat.append(words)
            label_mat.append(labels)
        return word_mat, label_mat

    def _loadtrain(self, path, intentname, regretest, intent_sentences, line_list, label_list):
        files = os.listdir(path)
        for file in files:
            pathfile = os.path.join(path, file)
            if os.path.isfile(pathfile):
                if not file.endswith('.csv'):
                    continue
                intent = os.path.splitext(file)[0]
                if len(intentname) > 0:
                    intent = (intentname + '/' + intent).split('/')[0]
                if len(self.include_intents_list) > 0 and intent not in self.include_intents_list:
                    intent = '其他'
                with open(pathfile) as f:
                    for line in f:
                        regretest.append(intent + ',' + line.strip())
                        line = util.sentence_repair(line, self.stopwords_re, self.homonymwords, self.synonymwords)
                        if len(line.strip()) == 0:
                            continue
                        if intent_sentences.get(intent, None) is None:
                            intent_sentences[intent] = list()
                        intent_sentences[intent].append((line, len(line), ''.join(lazy_pinyin(line)), len(''.join(lazy_pinyin(line)))))
                        if line not in line_list:
                            line_list.append(line)
                            label_list.append(intent)
            elif os.path.isdir(pathfile):
                if file == '__pycache__':
                    continue
                intent = file
                if len(intentname) > 0:
                    intent = (intentname + '/' + intent).split('/')[0]
                self._loadtrain(pathfile, intent, regretest, intent_sentences, line_list, label_list)

    def __id2vecs(self, word_vecs, sent_words):
        model = Dictionary(sent_words)
        model.filter_extremes(no_below=self.min_freq, no_above=1.0, keep_n=self.max_vocab)  # 去掉高频、低频词
        self.word_inds = model.token2id  # 存放的是单词-id key-value对
        self.word_inds = {word: ind + 2 for word, ind in self.word_inds.items()}  # 对每一个value执行value+2
        self.embed_mat = np.zeros((min(self.max_vocab + 2, len(self.word_inds) + 2), self.embed_len))   # 初始化矩阵embed_mat
        for word, ind in self.word_inds.items():
            if word in word_vecs.vocab:
                if ind < self.max_vocab:
                    self.embed_mat[ind] = word_vecs[word]    # ？
        self.__log('###id2vecs ', len(self.embed_mat), 'x', len(self.embed_mat[0]), '\n', self.embed_mat)

    def __label2ind(self, labels):
        labels_list = []
        for i in range(len(labels)):
            labels_list.extend(labels[i])
        self.label_inds = {}
        self.label_inds['N'] = 0
        labels = sorted(list(set(labels_list)))
        for i in range(len(labels)):
            self.ind_label[i + 1] = labels[i]
            self.label_inds[labels[i]] = i + 1

    def __pad(self, seq):
        if len(seq) < self.seq_len:
            return [0] * (self.seq_len - len(seq)) + seq
        else:
            return seq[-self.seq_len:]

    def __sent2ind(self, words, keep_oov):
        seq = list()
        for word in words:
            if word in self.word_inds:
                seq.append(self.word_inds[word])
            elif keep_oov:
                seq.append(1)
        return self.__pad(seq)

    def __add_buf(self, seqs):
        buf = [0] * int((win_len - 1) / 2)
        buf_seqs = list()
        for seq in seqs:
            buf_seqs.append(buf + seq + buf)
        return buf_seqs

    def __align(self, sent_words, labels, extra):
        pad_seqs = list()
        for words in sent_words:
            pad_seq = self.__sent2ind(words, keep_oov=True)
            pad_seqs.append(pad_seq)
        if extra:
            pad_seqs = self.__add_buf(pad_seqs)
        pad_seqs = np.array(pad_seqs)

        inds = list()
        for label in labels:
            ls = []
            for item in label:
                ls.append(self.label_inds[item])
            inds.append(self.__pad(ls))
        inds = np.array(inds)
        self.__log('###align ', len(pad_seqs), 'x', len(pad_seqs[0]), ' ',len(inds), '\n', pad_seqs, '\n', inds)
        return pad_seqs, inds

    def __get_metric(self, model, loss_func, pairs):
        sents, labels = pairs
        labels = labels.view(-1)
        num = (labels > 0).sum().item()
        prods = model(sents)  # 返回是一个Variable，里面包含grad_fn才能实现反向梯度传递[[-0.0785, -0.9034, -0.3025,  2.0512, -1.0713], [-0.0614, -0.8819,  0.3913,  1.0619, -0.7799],
        prods = prods.view(-1, prods.size(-1))
        preds = torch.max(prods, 1)[1]  # 取最大的预测
        loss = loss_func(prods, labels)  # tensor(1.3219, grad_fn=<NllLossBackward>)
        acc = (preds == labels).sum().item()  # 预测与结果有多少是一致的
        return loss, acc, num

    def __batch_train(self, model, loss_func, loader, optim):  # 训练数据
        total_loss, total_acc, total_num = [0] * 3
        for step, pairs in enumerate(loader):
            batch_loss, batch_acc, batch_num = self.__get_metric(model, loss_func, pairs)
            optim.zero_grad()
            batch_loss.backward()
            optim.step()
            total_loss = total_loss + batch_loss.item()
            total_acc, total_num = total_acc + batch_acc, total_num + batch_num
            self.__log('{} {} - loss: {:.3f} - acc: {:.3f}'.format('step', step + 1, batch_loss/batch_num, batch_acc/batch_num))
        return total_loss / total_num, total_acc / total_num

    def __batch_dev(self, model, loss_func, loader):
        total_loss, total_acc, total_num = [0] * 3
        for step, pairs in enumerate(loader):
            batch_loss, batch_acc, batch_num = self.__get_metric(model, loss_func, pairs)
            total_loss = total_loss + batch_loss.item()
            total_acc, total_num = total_acc + batch_acc, total_num + batch_num
        return total_loss / total_num, total_acc / total_num

    def __compile_fit(self, batch_size, epochs, lines, labels):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设置GPU
        bound = int(len(lines)*(1-self.validation_split))
        train_sents = torch.LongTensor(lines[:bound]).to(device)
        train_labels = torch.LongTensor(labels[:bound]).to(device)
        dev_sents = torch.LongTensor(lines[bound:]).to(device)
        dev_labels = torch.LongTensor(labels[bound:]).to(device)

        train_loader = DataLoader(TensorDataset(train_sents, train_labels), batch_size=batch_size, shuffle=True)
        dev_loader = DataLoader(TensorDataset(dev_sents, dev_labels), batch_size=batch_size, shuffle=True)
        embed_mat_tensor = torch.Tensor(self.embed_mat)  # torch.FlaotTensor 字向量转为tensor

        class_num = len(self.label_inds)
        model = None
        filepath = ''
        if self.type == '5.0.1':
            model = Cnn(embed_mat_tensor, class_num).to(device)
            filepath = self.datapath + '/cnnclass.pkl'
        elif self.type == '5.0.2':
            model = Rnn(embed_mat_tensor, class_num).to(device)
            filepath = self.datapath + '/rnnclass.pkl'
        loss_func = CrossEntropyLoss(ignore_index=0, reduction='sum')  # 交叉熵损失函数LogSoftMax和NLLLoss的集成，一般用于分类

        self.__log('###compile_fit adam(lr=', self.lr, ') batch_size=', int(batch_size),' epochs=', int(epochs),
                   ' validation_split=', self.validation_split, '\n{}'.format(model))

        min_dev_loss = float('inf')  # 正无穷
        for i in range(epochs):
            model.train()
            optimizer = Adam(model.parameters(), lr=self.lr)  # optim.Adam 随机优化方法
            start = time.time()
            train_loss, train_acc = self.__batch_train(model, loss_func, train_loader, optimizer)
            delta = time.time() - start
            with torch.no_grad():  # 在上下文环境中切断梯度计算
                model.eval()  # 切换到测试模式
                dev_loss, dev_acc = self.__batch_dev(model, loss_func, dev_loader)
            extra = ''
            if dev_loss < min_dev_loss:
                extra = ', val_loss reduce {:.3f}'.format(min_dev_loss - dev_loss)
                torch.save(model, filepath)
                min_dev_loss = dev_loss
            self.__log('{} {} - {:.2f}s - loss: {:.3f} - acc: {:.3f} - val_loss: {:.3f} - val_acc: {:.3f}'.format(
                'epoch', i, delta, train_loss, train_acc, dev_loss, dev_acc) + extra)


def applyModelTrain(type, datapath, threshold, modelparmvalue, trainloger):
    return CommModelTrain(type, datapath, threshold, modelparmvalue, trainloger)