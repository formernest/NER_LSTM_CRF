# !env python
# coding=utf-8
import pickle
import torch
import torch.optim as optim
from LSTM_CRF import BiLSTM_CRF
from resultCal import *
from torch.utils.data import TensorDataset, DataLoader
import datetime

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
    x_valid = pickle.load(inp)
    y_valid = pickle.load(inp)

print("train len:", len(x_train))
print("test len:", len(x_test))
print("valid len", len(x_valid))

#############
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100  # 嵌入层维度
HIDDEN_DIM = 200  # 隐藏层数量
EPOCHS = 5    # 迭代几次
batch_size = 50    # 每个batch的大小
# 将START_TAG和STOP_TAG加入tag字典中
tag2id[START_TAG] = len(tag2id)
tag2id[STOP_TAG] = len(tag2id)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

x_train = x_train[:10000]
y_train = y_train[:10000]

x_test = x_test[:1000]
y_test = y_test[:1000]


def train():

    model = BiLSTM_CRF(len(word2id) + 1, tag2id, EMBEDDING_DIM, HIDDEN_DIM).to(device)  # 实例化模型
    optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=1e-4)  # 随机梯度下降优化算法

    # train_data
    sentence = torch.tensor(x_train, dtype=torch.long).to(device)
    y_train1 = [[tag2id[y_train[i][j]] for j in range(len(y_train[i]))] for i in range(len(y_train))]
    tags = torch.tensor(y_train1, dtype=torch.long).to(device)
    trainloader = DataLoader(TensorDataset(sentence, tags), batch_size=batch_size, shuffle=True, drop_last=True)

    # test_data
    sentence = torch.tensor(x_test, dtype=torch.long).to(device)
    y_test1 = [[tag2id[y_test[i][j]] for j in range(len(y_test[i]))] for i in range(len(y_test))]
    tags = torch.tensor(y_test1, dtype=torch.long).to(device)
    testloader = DataLoader(TensorDataset(sentence, tags), batch_size=batch_size, shuffle=True, drop_last=True)
    print("train start:", datetime.datetime.now())
    # 训练
    for epoch in range(EPOCHS):
        for batch, data in enumerate(trainloader, 0):
            sentences, labels = data
            model.zero_grad()       # 清空梯度
            loss = model.batch_loss(sentences, labels)
            loss.backward()
            optimizer.step()
            if batch % 20 == 0 and batch > 0:
                print("epoch:", epoch, "batch:", batch, 'current time:', datetime.datetime.now())

        # 用来保存测试结果
        entityres = []
        entityall = []
        # 每个epoch后测试一下
        print('test start:', datetime.datetime.now())
        for batch, data in enumerate(testloader, 0):
            sentences, labels = data
            for sentence, label in zip(sentences, labels):
                score, predict = model(sentence)
                entityres = calculate(sentence, predict, id2word, id2tag, entityres)
                entityall = calculate(sentence, label, id2word, id2tag, entityall)

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

    print('train finished:', datetime.datetime.now())

    path_name = "./model/model.pkl"
    print(path_name)
    torch.save(model, path_name)
    print("model has been saved")


def test():
    path_name = "./model/model.pkl"
    model = torch.load(path_name)
    entityres = []
    entityall = []
    # 测试
    for sentence, tags in zip(x_test, y_test):
        sentence = torch.tensor(sentence, dtype=torch.long)
        score, predict = model(sentence)
        entityres = calculate(sentence, predict, id2word, id2tag, entityres)
        entityall = calculate(sentence, tags, id2word, id2tag, entityall)
    jiaoji = [i for i in entityres if i in entityall]
    if len(jiaoji) != 0:
        zhun = float(len(jiaoji)) / len(entityres)  # 准确率
        zhao = float(len(jiaoji)) / len(entityall)  # 召回率
        print("test:")
        print("zhun:", zhun)
        print("zhao:", zhao)
        print("f:", (2 * zhun * zhao) / (zhun + zhao))
    else:
        print("zhun:", 0)


def predict(sentence):
    path = './model/model.pkl'
    model = torch.load(path)



train()
