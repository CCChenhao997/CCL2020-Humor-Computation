'''
@Description: 
@version: 
@Author: chenhao
@Date: 2020-06-23 20:06:58
@LastEditors: chenhao
@LastEditTime: 2020-06-23 20:07:06
'''

import os
import nltk
import re
import json
import pickle
import numpy as np
import pandas as pd
from pytorch_transformers import BertTokenizer
# from transformers import BertTokenizer
from torch.utils.data import Dataset


def parse_data(data_path):
    df = pd.read_csv(data_path, sep='\t', header=None, encoding='utf-8', engine='python')
    all_data = []
    for index, line in df.iterrows():
        speaker = line[1].lower().strip()
        sentence = line[2].lower().strip()
        polarity = int(line[3])

        data = {'speaker': speaker, 'sentence': sentence, 'polarity': polarity}
        all_data.append(data)

    return all_data


class Tokenizer4Bert(object):
    def __init__(self, max_length, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_length = max_length

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return Tokenizer4Bert.pad_sequence(sequence, pad_id=0, maxlen=self.max_length, 
                                    padding=padding, truncating=truncating)

    @staticmethod
    def pad_sequence(sequence, pad_id, maxlen, dtype='int64', padding='post', truncating='post'):
        # x = (np.zeros(maxlen) + pad_id).astype(dtype)   # 长度为maxlen的数组中的元素全为pad_id，也就是0
        x = (np.ones(maxlen) * pad_id).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]  # 把过长的句子前面部分截断
        else:
            trunc = sequence[:maxlen]   # 把过长的句子尾部截断
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc      # 在句子尾部打padding
        else:
            x[-len(trunc):] = trunc     # 在句子前面打padding
        return x

    @staticmethod
    def split_text(text):
        # for ch in ["\'s", "\'ve", "n\'t", "\'re", "\'m", "\'d", "\'ll", ",", ".", "!", "*", "/", "?", "(", ")", "\"", "-", ":"]:
        #     text = text.replace(ch, " "+ch+" ")
        return text.strip().split()


class BertSentenceDataset(Dataset):
    ''' PyTorch standard dataset class '''
    def __init__(self, fname, tokenizer, target_dim, opt):

        parse = parse_data
        data = list()

        for obj in parse(fname):
            text_raw_indices = tokenizer.text_to_sequence(obj['sentence'])
            text_raw_bert_indices = tokenizer.text_to_sequence("[CLS] " + obj['sentence'] + " [SEP]")
            text_bert_indices = tokenizer.text_to_sequence("[CLS] " + obj['sentence'] + " [SEP] " + obj['speaker'] + " [SEP]")
            speaker_indices = tokenizer.text_to_sequence(obj['speaker'])
            speaker_len = np.sum(speaker_indices != 0)
            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (speaker_len + 1))
            bert_segments_ids = tokenizer.pad_sequence(bert_segments_ids, 0, tokenizer.max_length)
            polarity = obj['polarity']
            data.append(
                {
                    'text_raw_bert_indices': text_raw_bert_indices,
                    'bert_segments_ids': bert_segments_ids,
                    'text_bert_indices': text_bert_indices,
                    'polarity': polarity
                }
            )

        self._data = data
    
    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)
