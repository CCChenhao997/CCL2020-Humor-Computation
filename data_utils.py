import os
import nltk
import re
import json
import pickle
import numpy as np
import pandas as pd
from transformers import BertTokenizer, AutoTokenizer, XLMRobertaTokenizer, GPT2Tokenizer, RobertaTokenizer
from torch.utils.data import Dataset


def parse_data(data_path):
    df = pd.read_csv(data_path, sep='\t', header=None, encoding='utf-8', engine='python')
    all_data = []
    for index, line in df.iterrows():
        speaker = line[1].lower().strip()
        sentence = line[2].lower().strip()
        try:
            polarity = int(line[3])
        except:
            polarity = 0

        data = {'speaker': speaker, 'sentence': sentence, 'polarity': polarity}
        all_data.append(data)

    return all_data


def parse_transdata(data_path):
    df = pd.read_csv(data_path, sep='\t', header=None, encoding='utf-8', engine='python')
    all_data = []
    for index, line in df.iterrows():
        sentence_pre = line[3].lower().strip()
        sentence_post = line[4].lower().strip()
        try:
            polarity = int(line[5])
        except:
            polarity = 0

        data = {'sentence_pre': sentence_pre, 'sentence_post': sentence_post, 'polarity': polarity}
        all_data.append(data)

    return all_data


class Tokenizer4Bert(object):
    def __init__(self, max_length, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        # self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_bert_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)
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
        data = list()

        if opt.transdara:
            parse = parse_transdata
            for obj in parse(fname):
                sentence_pre_indices = tokenizer.text_to_sequence(obj['sentence_pre'])
                sentence_post_indices = tokenizer.text_to_sequence(obj['sentence_post'])
                sentence_post_bert_indices = tokenizer.text_to_sequence("[CLS] " + obj['sentence_post'] + " [SEP]")
                sentence_pair_bert_indices = tokenizer.text_to_sequence("[CLS] " + obj['sentence_pre'] + " [SEP] " + obj['sentence_post'] + " [SEP]")
                # sentence_post_len = np.sum(sentence_post_indices != 0)
                bert_segments_ids = np.asarray([0] * (np.sum(sentence_pre_indices != 0) + 2) + [1] * (np.sum(sentence_post_indices != 0) + 1))
                bert_segments_ids = tokenizer.pad_sequence(bert_segments_ids, 0, tokenizer.max_length)
                polarity = obj['polarity']

                attention_mask = np.asarray([1] * np.sum(sentence_post_bert_indices != 0) + [0] * (opt.max_length - np.sum(sentence_post_bert_indices != 0)))
                attention_mask_pair = np.asarray([1] * np.sum(sentence_pair_bert_indices != 0) + [0] * (opt.max_length - np.sum(sentence_pair_bert_indices != 0)))
                data.append(
                    {
                        'sentence_bert_indices': sentence_post_bert_indices,
                        'sentence_pair_bert_indices': sentence_pair_bert_indices,
                        'bert_segments_ids': bert_segments_ids,
                        'attention_mask': attention_mask,
                        'attention_mask_pair': attention_mask_pair,
                        'polarity': polarity
                    }
                )

        else:
            parse = parse_data

            for obj in parse(fname):
                sentence_indices = tokenizer.text_to_sequence(obj['sentence'])
                sentence_bert_indices = tokenizer.text_to_sequence("[CLS] " + obj['sentence'] + " [SEP]")
                sentence_speaker_bert_indices = tokenizer.text_to_sequence("[CLS] " + obj['speaker'] + " [SEP] " + obj['sentence'] + " [SEP]")
                speaker_indices = tokenizer.text_to_sequence(obj['speaker'])
                bert_segments_ids = np.asarray([0] * (np.sum(speaker_indices != 0) + 2) + [1] * (np.sum(sentence_indices != 0) + 1))
                bert_segments_ids = tokenizer.pad_sequence(bert_segments_ids, 0, tokenizer.max_length)
                polarity = obj['polarity']

                attention_mask = np.asarray([1] * np.sum(sentence_bert_indices != 0) + [0] * (opt.max_length - np.sum(sentence_bert_indices != 0)))
                attention_mask_pair = np.asarray([1] * np.sum(sentence_speaker_bert_indices != 0) + [0] * (opt.max_length - np.sum(sentence_speaker_bert_indices != 0)))
                data.append(
                    {
                        'sentence_bert_indices': sentence_bert_indices,
                        'sentence_pair_bert_indices': sentence_speaker_bert_indices,
                        'bert_segments_ids': bert_segments_ids,
                        'attention_mask': attention_mask,
                        'attention_mask_pair': attention_mask_pair,
                        'polarity': polarity
                    }
                )

        self._data = data
    
    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)
