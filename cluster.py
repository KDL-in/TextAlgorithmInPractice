from types import MethodType, FunctionType
import pandas as pd
import numpy as np
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from TextRank4ZH.textrank4zh import TextRank4Keyword, TextRank4Sentence
from common import DocTokenizer,Vectorizer


'''
基于TFIDF的向量化工具，它需要读取所有数据统计词频
'''
class DefaultVectorizer(Vectorizer):
    def __init__(self, data_path='data/data.csv', stop_words_path='data/stop_words.txt'):
        super().__init__()
        self.tokenizer = DocTokenizer(stop_words_path=stop_words_path)
        texts = self.getTexts(data_path)
        self.tfidf = TfidfVectorizer(stop_words=self.tokenizer.stop_words, max_df=0.5, max_features=100)
        self.tfidf.fit(texts)

    def getTexts(self, data_path):
        data = pd.read_csv(data_path)
        return [self.tokenizer.seg(doc) for doc in data['content']]

    def vectorize(self, doc):
        return self.tfidf.transform([self.tokenizer.seg(doc)]).todense().tolist()[0]

'''
基于Fasttext的向量化工具，它需要训练好的fastext模型
'''
class FasttextVectorizer(Vectorizer):

    def vectorize(self, doc):
        return self.model.get_sentence_vector(self.tokenizer.seg(doc))

    def __init__(self, model_path='model/data_dim100_lr00.5_iter5.model', stop_words_path='data/stop_words.txt') :
        super().__init__()
        self.tokenizer = DocTokenizer(stop_words_path=stop_words_path)
        self.model = fasttext.load_model(model_path)



'''
OnePass文本聚类器
'''
class SinglePassCluster():
    def __init__(self, vectorizer=None, center_func=None, threshold=0.1):
        # 聚类, 中心idx -> clusters
        self.clusters = {}
        # 聚类中心
        self.centers = []
        self.threshold = threshold
        # 向量化工具
        if not vectorizer:
            self.vectorizer = DefaultVectorizer()
        else:
            self.vectorizer = vectorizer
        # cluster中心计算工具
        if not center_func or not self.is_valid_func(center_func):
            self.center_func = lambda vecs: vecs[0]
        else:
            self.center_func = center_func

    def single_pass(self, doc):
        vec = self.vectorizer.vectorize(doc)
        max_sim, max_idx = self.cal_max_cos_sim(vec)
        # 相似度小于阈值，则新建为一个cluster
        # print(max_sim)
        if max_sim < self.threshold:
            idx = self.add_center(vec)
            self.clusters[idx] = [vec]
            return idx
        else:
            # 添加到最相似的cluster中，并更新中心vec
            self.clusters[max_idx].append(vec)
            self.centers[max_idx] = self.center_func(self.clusters[max_idx])
            return max_idx

    # 计算最大的余弦距离，及center idx
    def cal_max_cos_sim(self, vec):
        # 如果没有cluster中心，则返回余弦相似度为0
        if not self.centers:
            return 0, -1
        sims = cosine_similarity(np.array([vec]), np.array(self.centers))
        max_idx = np.argmax(sims, axis=1)[0]
        max_sim = sims[0][max_idx]
        return max_sim, max_idx

    def add_center(self, vec):
        self.centers.append(vec)
        return len(self.centers) - 1

    def is_valid_func(self, func):
        return (isinstance(func, FunctionType) or isinstance(func, MethodType))


stop_words_path = 'data/stopwords.txt'
data_path = 'data/data.csv'
model_path = 'model/data_dim100_lr00.5_iter10.model'
# 如果采用FasttextVectorizer，阈值建议设置成0.8
threshold = 0.1
vectorizer = DefaultVectorizer(data_path=data_path, stop_words_path=stop_words_path)
# vectorizer = FasttextVectorizer(model_path=model_path, stop_words_path=stop_words_path)
cluster = SinglePassCluster(vectorizer=vectorizer, threshold=threshold, center_func=lambda vecs:np.average(np.array(vecs), axis=0))


result = {}
data = pd.read_csv(data_path)
for doc in data[data['label']=='game']['content']:
    cluster_id = cluster.single_pass(doc)
    # print(f"{cluster_id}: {doc}")
    docs = result.get(cluster_id, [])
    docs.append(doc)
    result[cluster_id] = docs

tr4w = TextRank4Keyword()
tr4s = TextRank4Sentence()



for cluster_id, docs in sorted(result.items(), key=lambda item: len(item[1]), reverse=True):
    tr4w.analyze('\n'.join(docs), window=5, lower=True)
    tr4s.analyze('\n'.join(docs), lower=True)
    words = [x['word'] for x in tr4w.get_keywords(num=10, word_min_len=2)]
    sentences = [x['sentence'] for x in tr4s.get_key_sentences(num=4, sentence_min_len=5)]
    print(f"【主题】：{cluster_id} 句子数量: {len(docs)}")
    print(f"【关键词】：{','.join(words)}")
    print(f"【中心句】:")
    print('\n'.join(sentences))
    print("------------------------------------")


