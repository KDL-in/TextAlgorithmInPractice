import os
from random import shuffle

import fasttext
import numpy as np
import pandas as pd
import sklearn.utils as shuffle

from common import DocTokenizer


# 切分dataframe数据集
def split_dataframe(data, rate=0.8, shffule=True):
    train_n = int(len(data) * rate)
    if shffule:
        data = shuffle.shuffle(data, random_state=0)
    return data.iloc[:train_n], data.iloc[train_n:]


# 转化为fasttext格式
def to_fasttext_data(data):
    return [f"__label__{row['label'].strip()} {row['content'].strip()}\n" for _, row in data.iterrows()]


# 训练模型
def train_model(data_path=None, save_path=None, pretrained_vec_path=None, dim=100, epoch=5, lr=0.1, loss='softmax'):
    """
      训练一个模型
      @:param data_path: 数据路径
      @:param save_path: 模型保存路径，如果模型存在，则直接加载
      @:param pretrained_vec_path: 如果使用预训练向量，则提供路径，注意维度必须和dim一致。处理预训练向量，请参考README
      @:param dim 词向量化的维度
      @:param epoch 训练轮次
      @:param lr 学习率
      @:param loss 损失函数
    """
    if os.path.isfile(save_path):
        return fasttext.load_model(save_path)
    """
      训练一个监督模型, 返回一个模型对象

      @param input:           训练数据文件路径
      @param lr:              学习率
      @param dim:             向量维度
      @param ws:              cbow模型时使用
      @param epoch:           次数
      @param minCount:        词频阈值, 小于该值在初始化时会过滤掉
      @param minCountLabel:   类别阈值，类别小于该值初始化时会过滤掉
      @param minn:            构造subword时最小char个数
      @param maxn:            构造subword时最大char个数
      @param neg:             负采样
      @param wordNgrams:      n-gram个数
      @param loss:            损失函数类型, softmax, ns: 负采样, hs: 分层softmax
      @param bucket:          词扩充大小, [A, B]: A语料中包含的词向量, B不在语料中的词向量
      @param thread:          线程个数, 每个线程处理输入数据的一段, 0号线程负责loss输出
      @param lrUpdateRate:    学习率更新
      @param t:               负采样阈值
      @param label:           类别前缀
      @param verbose:         ??
      @param pretrainedVectors: 预训练的词向量文件路径, 如果word出现在文件夹中初始化不再随机
      @return model object
    """
    if pretrained_vec_path is not None and os.path.isfile(pretrained_vec_path):
        classifier = fasttext.train_supervised(data_path, label='__label__', dim=dim, epoch=epoch,
                                               lr=lr, wordNgrams=2, loss=loss, pretrainedVectors=pretrained_vec_path)
    else:
        classifier = fasttext.train_supervised(data_path, label='__label__', dim=dim, epoch=epoch,
                                               lr=lr, wordNgrams=2, loss=loss)
    if save_path is not None:
        classifier.save_model(save_path)
    return classifier


# predict
def predict(model, data_test_path='data/data_test.txt'):
    x = []
    y = []
    with open(data_test_path, 'r', encoding='utf-8') as f:
        for line in f:
            items = line[9:].strip('\n').split(' ')
            y.append(items[0])
            x.append(' '.join(items[1:]))
    pred_y = [model.predict(s)[0][0][9:] for s in x]
    return pred_y, y


# evaluate
def evaluate(pred_y, y):
    pred_y = np.array(pred_y)
    y = np.array(y)

    categories = np.unique(y)
    p_dict = dict([(x, 0) for x in categories])
    r_dict = dict([(x, 0) for x in categories])
    t_dict = dict([(x, 0) for x in categories])

    for pred, label in zip(pred_y, y):
        t_dict[label] += 1
        r_dict[pred] += 1
        if pred == label:
            p_dict[label] += 1

    result = {}
    for c in categories:
        precision = p_dict[c] / t_dict[c]
        recall = p_dict[c] / r_dict[c]
        f1 = (2 * precision * recall) / (precision + recall)
        print(f"{c}: precision {precision}, recall {recall}, f1 {f1}")
        result[c] = {'precision': precision, 'recall': recall, 'f1': f1}

    accuracy = (pred_y == y).sum() / len(y)
    precision = np.average([result[c]['precision'] for c in categories])
    recall = np.average([result[c]['recall'] for c in categories])
    f1 = np.average([result[c]['f1'] for c in categories])

    result['total'] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    print(f"overall: accuracy {accuracy}, precision {precision}, recall {recall}, f1 {f1}")

    return result


def test_model(model, data_test_path='./data/data_test.txt'):
    pred_y, y = predict(model, data_test_path=data_test_path)
    return evaluate(pred_y, y)


# 数据预处理
tokenizer = DocTokenizer(stop_words_path='data/stopwords.txt')
data = pd.read_csv('./data/data.csv')
data['content'] = data['content'].apply(lambda sentence: tokenizer.seg(sentence))
# 数据切分

train_data, test_data = split_dataframe(data.dropna())
with open('data/data_train.txt', 'w', encoding='utf-8') as train, open('data/data_test.txt', 'w',
                                                                       encoding='utf-8') as test:
    for s in to_fasttext_data(train_data):
        train.write(s)
    for s in to_fasttext_data(test_data):
        test.write(s)

# 训练
dim = 100
lr = 0.5
epoch = 5
save_path = f'model/data_dim{str(dim)}_lr0{str(lr)}_iter{str(epoch)}.model'
data_path = 'data/data_train.txt'
data_test_path = 'data/data_test.txt'

model = train_model(data_path, save_path=save_path, dim=dim, lr=lr, epoch=epoch)
result = test_model(model)

print(f"overall: {result['total']}")
