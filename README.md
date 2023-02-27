# 介绍
本项目包含三种中文文本（短文本）算法：文本分类、文本聚类、文本摘要（关键词、中心句）

- 文本分类：基于fasttext实现
- 文本聚类：采用One-Pass Cluster实现
- 文本摘要：对文本聚类结果进行摘要
# 细节
## 测试数据
项目中采用数据为中文新闻标题数据集，来源为互联网。具体使用什么数据不是特别重要，仅作为测试使用。
> [https://blog.csdn.net/qq_36291847/article/details/115455226](https://blog.csdn.net/qq_36291847/article/details/115455226)

## 分词
项目中均采用结巴分词。可以自由替换stopwords。
> [https://github.com/fxsjy/jieba](https://github.com/fxsjy/jieba)

## 文本分类（classification.py）
基于fasttext实现的高效文本分类方案，并提供各个分类、总体的评估指标计算（accuracy，precision，recall，F1）。
测试数据新闻标题比较短数据归类的难度比较大，所以总体的指标并不算太高。但是可以看到部分分类的指标都还行。
```
label: evaluation
baby: precision 0.8728179551122195, recall 0.7675438596491229, f1 0.8168028004667446
beauty: precision 0.837696335078534, recall 0.8672086720867209, f1 0.8521970705725699
car: precision 0.8791469194312796, recall 0.8983050847457628, f1 0.8886227544910179
...
overall: {'accuracy': 0.7590710040663121, 'precision': 0.7569864906813206, 'recall': 0.7620159648481086, 'f1': 0.7578451905253383}
```
> 如果分类类别很多，可以考虑使用hierarchical softmax
> fasttext的细节请参考
> [https://github.com/facebookresearch/fastText/tree/main/python](https://github.com/facebookresearch/fastText/tree/main/python)

### 使用fasttext预训练向量
如果想使用fasttext预训练的向量，可以去这里下载
> [https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md](https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md)

注意fastext提供的向量`.vec`全是300维的，如果直接使用，需要保证fasttext模型参数也是300。否则需要下载`.bin`模型，自己进行降维。方法如下:
```python
import fasttext.util
model = fasttext.load_model('model/cc.zh.300.bin')
fasttext.util.reduce_model(model, 100)
words = model.get_words()
with open('model/pretrained.100.vec', 'w') as file_out:
    file_out.write(str(len(words)) + " " + str(model.get_dimension()) + "\n")
    for w in words:
        v = model.get_word_vector(w)
        vstr = ""
        for vi in v:
            vstr += " " + str(vi)
        try:
            file_out.write(w + vstr + '\n')
        except:
            pass
```
使用也很简单，`pretrained_vec_path`传入参数`pretrained_vec_path`。
## 文本聚类（cluster.py）
文本聚类基于One-Pass Cluster实现。
### One-Pass Cluster
这个聚类算法非常简洁，速度也快。首先它基于文本向量的[余弦相似度](https://zh.wikipedia.org/zh-hans/%E4%BD%99%E5%BC%A6%E7%9B%B8%E4%BC%BC%E6%80%A7)：

1. 读入一个文本向量
2. 计算与当前所有clusters的余弦相似度，找到最大的余弦相似度
3. 如果余弦相似度大于等于某个阈值，则说明该文本和该cluster非常相似，加入该cluster
4. 如果余弦相似度小于某个阈值，则说明该文本与所有clusters都不相似，则新建一个cluster

### 向量化方法
支持自定义不同的文本向量化方式（继承`Vectorizer`），项目中集成了：

- TFIDF向量化，[https://zh.wikipedia.org/zh/Tf-idf](https://zh.wikipedia.org/zh/Tf-idf)
- Fastext向量化，使用训练好的模型向量化

建议使用TFIDF，词频聚类文本看起来效果还是比较好的，缺点是TFIDF都计算时间比较久。
```python
主题】：0 句子数量: 123
【关键词】：游戏,玩家,e3,手游,世界,中国,背景,截图,分享,世界杯
【中心句】:
谁是你的游戏style
腾讯游戏平台
圣域三国游戏特色
《吞噬苍穹》游戏资料――摆摊
------------------------------------
【主题】：37 句子数量: 75
【关键词】：视频,英雄传,集锦,小鸟,关卡,攻略,愤怒,回顾,通关,剑灵
【中心句】:
愤怒的小鸟英雄传北海-1关卡视频攻略
愤怒的小鸟英雄传东沙岛关卡视频攻略
愤怒的小鸟英雄传红河谷-4关卡视频攻略
愤怒的小鸟英雄传山猪城堡关卡视频攻略
------------------------------------
【主题】：10 句子数量: 71
【关键词】：世界,魔兽,坦克,视频,boss,战争,仙侠,开启,测试,pk
【中心句】:
魔兽世界新坐骑戈隆曝光
魔兽世界官方各职业教学视频 德鲁伊篇
新《仙侠世界定制版》产品六大升级
魔兽世界欧服官网6.0爆料
------------------------------------
【主题】：36 句子数量: 33
【关键词】：曝光,视频,cg,原画,幻想,玩法,公测,副本,明日,传说
【中心句】:
《龙武》6月27日公测 灵魂藏地新副本曝光
《神魔大陆2》首部公测CG视频曝光
《龙武》6月27日公测 完整版剧情CG曝光
《神魔大陆2》首部公测CG曝光 史诗副本前瞻
```
### 聚类中心向量自定义
cluster中的向量求取cluster中心向量可以自定义，默认采用第一个向量成员作为中心，但推荐计算平均值。
```python
SinglePassCluster(center_func=lambda vecs:np.average(np.array(vecs), axis=0))
```
### 聚类关键词、中心句提取

关键词、中心句提取基于TextRank4ZH，原理是大名鼎鼎但PageRank算法。但由于该工具比较老，新版本networkx有报错，这里集成了一下源码。
> https://github.com/letiantian/TextRank4ZH

# LICENSE
[MIT](LICENSE)