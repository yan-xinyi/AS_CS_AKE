# -*- coding: utf-8 -*-
# @Time    : 2026/1/20
# @Author  : leizhao150, Xinyi Yan
import json
from collections import defaultdict
from nltk import word_tokenize, sent_tokenize
from tqdm import tqdm
from config import seq_max_length, punctuations

for corpus in ['corpus-ph']:

    for dn in ['test', 'train']:

        datas = json.load(open('./abs-datas/re/%s/%s.json'%(corpus, dn), 'r', encoding='utf-8'))

        # 读取原始数据
        origin_datas = json.load(open('./dataset/%s/%s.json'%(corpus, dn), 'r', encoding='utf-8'))

        for index, sentences in enumerate(tqdm(datas)):

            origin_data = origin_datas[index]

            def inner_fun(text):
                texts = text.split("\n")
                res = []
                for text in texts:
                    text = text.strip()
                    if len(text) == 0: continue
                    res.extend([sen.strip() for sen in sent_tokenize(text) if len(sen.strip()) != 0])
                return res


            # 遍历句子
            for field in ['ib', 'rw', 'md', 'er', 'dc']:

                texts = inner_fun(origin_data[field])

                sub_sentences = []
                for text in texts:
                    for sen in sentences:
                        if text == sen[0]:
                            sub_sentences.append(sen)

                sub_sentences = [sentence + [rank] for rank, sentence in enumerate(sub_sentences)]
                key_sens = sorted(sub_sentences, key=lambda item: item[1], reverse=True)

                # 遍历句子
                num, temp = 0, []
                for sen in key_sens:

                    if sen[0] == '无标题': continue
                    if str(sen[0]).isupper(): continue

                    words = word_tokenize(sen[0])

                    words = [word.strip() for word in words if len(word.strip()) > 0] # word.strip() not in punctuations and
                    while len(words) > 0 and words[0] in punctuations:words = words[1: ]
                    if len(words) == 0: continue

                    curr_length = num + len(words)
                    if curr_length < seq_max_length:
                        temp.append([" ".join(words), sen[1], sen[2]])
                        num += len(words)
                    else:
                        break

                assert num <= seq_max_length

                key_sens = sorted(temp, key=lambda item: item[2], reverse=False)
                text = " ".join([sen[0] for sen in key_sens])

                origin_datas[index]["%s(r-%s)"%(field, seq_max_length)] = text


        # 保存数据
        json.dump(origin_datas, open('./dataset/%s/%s.json'%(corpus, dn), 'w', encoding='utf-8'))