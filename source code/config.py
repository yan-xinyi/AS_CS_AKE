# -*- coding: utf-8 -*-
# @Time    : 2026/1/20
# @Author  : leizhao150, Xinyi Yan
import torch
from nltk import PorterStemmer
from nltk.corpus import stopwords

bert_path = '../scibert-pr-model'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

valid_pos = ['NN', 'NNP', 'JJ', 'NNS']
stops_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
punctuations = [',', '.', ':', ';', '``','?', '（','）','(', ')', '[', ']',
                '&', '!', '*', '@', '#', '$', '%', '\\','\"','}','{', '-',
                '–', '—', '..', '/', '=', '∣', '…', '′', '⋅', '×', '+', '•',
                '<', '>', '’', "''"]

# 摘要句子数量
seq_max_length = 256
