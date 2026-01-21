# -*- coding: utf-8 -*-
# @Time    : 2026/1/20
# @Author  : leizhao150, Xinyi Yan
import json
from nltk import word_tokenize
from config import punctuations, seq_max_length
from spacy.lang.en import English
from tqdm import tqdm
from TransformerSum.src.extractive import ExtractiveSummarizer
from config import device

nlp = English()
nlp.add_pipe("sentencizer")
model = ExtractiveSummarizer.load_from_checkpoint("./TransformerSum/models/epoch=3.ckpt", strict=False).to(device)

def generate_summarization():

    for corpus in ['corpus-ph']:

        for dn in ['test', 'train']:

            # 读取原始数据
            datas = json.load(open('./ft-datas/%s/ft.json' % corpus, 'r', encoding='utf-8'))[dn]
            rv_datas = []

            for index, document in enumerate(tqdm(datas)):

                # 每篇文本
                word_nums = 0
                rv_data, seq_list = [], []
                for item in document:

                    sentence = item['sentence']

                    seg_words = [token.text for token in nlp(sentence) if str(token) != "."] + ["."]

                    curr_length = word_nums + len(seg_words)

                    if curr_length <= 256:
                        seq_list.append(sentence)
                        word_nums += len(seg_words)
                    else:
                        score = model.predict_sentences(seq_list, raw_scores=True)
                        temp = [[se, so[1]] for se, so in zip(seq_list, score)]

                        rv_data.extend(temp)

                        word_nums = len(seg_words)
                        seq_list = [sentence]

                rv_datas.append(rv_data)

            # 保存数据
            json.dump(rv_datas, open('./abs-datas/ex/%s/%s.json'%(corpus, dn), 'w', encoding='utf-8'))


def main():

    for corpus in ['corpus-ph']:

        for dn in ['test', 'train']:

            datas = json.load(open('./abs-datas/ex/%s/%s.json' % (corpus, dn), 'r', encoding='utf-8'))

            # 读取原始数据
            origin_datas = json.load(open('./dataset/%s/%s.json' % (corpus, dn), 'r', encoding='utf-8'))

            for index, sentences in enumerate(tqdm(datas)):

                sentences = [sentence + [rank] for rank, sentence in enumerate(sentences)]
                key_sens = sorted(sentences, key=lambda item: item[1], reverse=True)

                # 遍历句子
                num, temp = 0, []
                for sen in key_sens:

                    if sen[0] == '无标题': continue
                    if str(sen[0]).isupper(): continue

                    words = word_tokenize(sen[0])

                    words = [word.strip() for word in words if
                             len(word.strip()) > 0]  # word.strip() not in punctuations and
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

                origin_datas[index]['TS(Ex)-%s' % seq_max_length] = text
                origin_datas[index]['AS(Ex)-%s' % seq_max_length] = origin_datas[index]['ab'] + " " + text

            # 保存数据
            json.dump(origin_datas, open('./dataset/%s/%s.json' % (corpus, dn), 'w', encoding='utf-8'))


if __name__ == '__main__':

    main()





