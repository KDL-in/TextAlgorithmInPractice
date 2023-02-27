import re
from types import MethodType, FunctionType
import jieba

'''
文本分词工具
'''
class DocTokenizer():
    def __init__(self, stop_words_path='./data/stop_words.txt'):
        self.pattern = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fa5]+")
        self.stop_words = self.stop_words(stop_words_path)

    def clean_txt(self, raw):
        return self.sub(' ', raw)

    def stop_words(self, stop_words_path):
        with open(stop_words_path, 'r', encoding='utf-8') as swf:
            return [line.strip() for line in swf]

    def seg(self, sentence):
        return self.seg(sentence, apply=self.clean_txt)

    def seg(self, sentence, apply=None):
        if isinstance(apply, FunctionType) or isinstance(apply, MethodType):
            sentence = apply(sentence)
        return ' '.join([i for i in jieba.cut(sentence) if i.strip() and i not in self.stop_words])
