# -*- coding: utf-8 -*-
# @Time    : 2026/1/20
# @Author  : leizhao150, Xinyi Yan
import json
import torch
from tqdm import tqdm
from transformers import LEDForConditionalGeneration, LEDTokenizer
from config import seq_max_length

tokenizer = LEDTokenizer.from_pretrained("patrickvonplaten/led-large-16384-pubmed")
model = LEDForConditionalGeneration.from_pretrained("patrickvonplaten/led-large-16384-pubmed",
                                                    return_dict_in_generate=True).to("cuda")

for corpus in ['corpus-ph']:

    for dn in ['test', 'train']:

        # 读取原始数据
        origin_datas = json.load(open('./dataset/%s/%s.json'%(corpus, dn), 'r', encoding='utf-8'))

        for index, document in enumerate(tqdm(origin_datas)):

            text = [i.strip() for i in document['ft'].split("\n") if len(i.strip()) > 0 and i.strip() != '无标题']

            text = " ".join(text)

            input_ids = tokenizer(text,
                                  max_length=4096,
                                  truncation=True,
                                  return_tensors="pt").input_ids.to("cuda")
            global_attention_mask = torch.zeros_like(input_ids)

            global_attention_mask[:, 0] = 1
            summary_ids = model.generate(input_ids,
                                         global_attention_mask=global_attention_mask,
                                         max_length=seq_max_length).sequences
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

            origin_datas[index]['TS(Ge)-%s'%seq_max_length] = summary
            origin_datas[index]['AS(Ge)-%s'%seq_max_length] = origin_datas[index]['ab'] + " " + summary

        # 保存数据
        json.dump(origin_datas, open('./dataset/%s/%s.json'%(corpus, dn), 'w', encoding='utf-8'))