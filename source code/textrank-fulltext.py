# -*- coding: utf-8 -*-
# @Time    : 2026/1/20
# @Author  : leizhao150, Xinyi Yan
import json
from nltk import word_tokenize
from tqdm import tqdm
from config import punctuations, seq_max_length, stops_words, stemmer
from utils import textSummarization


def generate_summarization():

    for dt in ['train', 'test']:

        for corpus in ['corpus-ph']:

            # 读取数据
            rv_datas = []
            datas = json.load(open('./ft-datas/%s/ft.json' % corpus, 'r', encoding='utf-8'))[dt]

            for index, document in enumerate(tqdm(datas)):

                # 句子数量
                sen_nums = len(document)
                sentences_list, words_list = [], []

                # 每篇文本
                for item in document:
                    sentence = item['sentence']
                    feature = item['feature']
                    label = item['label']

                    temp = []
                    for word in word_tokenize(sentence):
                        if word in stops_words or word in punctuations: continue
                        word = stemmer.stem(word)
                        temp.append(word)
                    words_list.append(temp)
                    sentences_list.append(sentence)

                scores = textSummarization(words_list)

                # 根据textrank值对未处理的句子进行排序
                ranked_sentences = [(s, scores[i]) for i, s in enumerate(sentences_list)]

                rv_datas.append(ranked_sentences)

            # 保存数据
            json.dump(rv_datas, open('./abs-datas/tr/%s/%s.json' % (corpus, dt), 'w', encoding='utf-8'))


def main():

    for corpus in ['corpus-ph']:

        for dn in ['test', 'train']:

            datas = json.load(open('./abs-datas/tr/%s/%s.json'%(corpus, dn), 'r', encoding='utf-8'))

            # 读取原始数据
            origin_datas = json.load(open('./dataset/%s/%s.json'%(corpus, dn), 'r', encoding='utf-8'))

            for index, sentences in enumerate(tqdm(datas)):

                sentences = [sentence + [rank] for rank, sentence in enumerate(sentences)]
                key_sens = sorted(sentences, key=lambda item: item[1], reverse=True)

                # 遍历句子
                num, temp = 0, []
                for sen in key_sens:

                    if sen[0] == '无标题': continue
                    if str(sen[0]).isupper(): continue

                    words = word_tokenize(sen[0])

                    words = [word.strip() for word in words if len(word.strip()) > 0]  # word.strip() not in punctuations and
                    while len(words) > 0 and words[0] in punctuations: words = words[1:]
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

                origin_datas[index]['TS(Tr)-%s'%seq_max_length] = text
                origin_datas[index]['AS(Tr)-%s' % seq_max_length] = origin_datas[index]['ab'] + " " + text

            # 保存数据
            json.dump(origin_datas, open('./dataset/%s/%s.json' % (corpus, dn), 'w', encoding='utf-8'))



if __name__ == '__main__':

    main()
