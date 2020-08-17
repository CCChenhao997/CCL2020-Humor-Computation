import os
import nltk
import re
import json
import pickle
import time
import torch
from datetime import timedelta
import numpy as np
import pandas as pd
from config import logger, opt
from transformers import BertTokenizer, AutoTokenizer, RobertaTokenizer, AlbertTokenizer
from torch.utils.data import Dataset
from pprint import pprint

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def parse_data(data_path):
    df = pd.read_csv(data_path, sep='\t', header=None, encoding='utf-8', engine='python')
    all_data = []
    for index, line in df.iterrows():
        dialogue_id = int(line[0])
        speaker = line[1].lower().strip()
        sentence = line[2].lower().strip()
        if 'en' in opt.dataset:
            sentence = re.sub('\x92', '\'', sentence)
            sentence = Tokenizer4Bert.split_text(sentence)
            sentence = re.sub('\"', '', sentence)
        else:
            sentence = re.sub('-{1,}', ' ', sentence)
            sentence = re.sub(' {2,}', ' ', sentence)
        try:
            polarity = int(line[3])
        except:
            polarity = 0

        data = {'dialogue_id': dialogue_id, 'speaker': speaker, 'sentence': sentence, 'polarity': polarity}
        all_data.append(data)

    return all_data


def parse_rawdata(data_path):
    df = pd.read_csv(data_path, encoding='utf-8', engine='python')
    all_data = []
    for index, line in df.iterrows():
        dialogue_id = int(line[1])
        # utterance_id = int(line[2])
        speaker = line[3].lower().strip()
        sentence = line[4].lower().strip()
        if 'en' in opt.dataset:
            sentence = re.sub('\x92', '\'', sentence)
            sentence = Tokenizer4Bert.split_text(sentence)
            sentence = re.sub('\"', '', sentence)
        else:
            sentence = re.sub('-{1,}', ' ', sentence)
            sentence = re.sub(' {2,}', ' ', sentence)
        try:
            polarity = int(line[5])
        except:
            polarity = 0

        data = {'dialogue_id': dialogue_id, 'speaker': speaker, 'sentence': sentence, 'polarity': polarity}
        all_data.append(data)

    return all_data


def parse_transdata(data_path):
    df = pd.read_csv(data_path, sep='\t', header=None, encoding='utf-8', engine='python')
    all_data = []
    for index, line in df.iterrows():
        dialogue_id = int(line[0])
        speaker_pre = line[1].lower().strip()
        speaker_post = line[2].lower().strip()
        sentence_pre = line[3].lower().strip() 
        sentence_post = line[4].lower().strip()
        if 'en' in opt.dataset:
            sentence_pre = re.sub('\x92', '\'', sentence_pre)
            sentence_pre = Tokenizer4Bert.split_text(sentence_pre)
            sentence_pre = re.sub('\"', '', sentence_pre)
            sentence_post = re.sub('\x92', '\'', sentence_post)
            sentence_post = Tokenizer4Bert.split_text(sentence_post)
            sentence_post = re.sub('\"', '', sentence_post)
        try:
            polarity = int(line[5])
        except:
            polarity = 0

        data = {'dialogue_id': dialogue_id, 'speaker_pre': speaker_pre, 'speaker_post': speaker_post,
                'sentence_pre': sentence_pre, 'sentence_post': sentence_post, 'polarity': polarity}
        all_data.append(data)

    return all_data


def parse_rawdata_dialogue(data_path):
    df = pd.read_csv(data_path, encoding='utf-8', engine='python')
    all_data = dict()
    prelogue_id, global_id, count = 0, 0, 0 
    for index, line in df.iterrows():
        dialogue_id = int(line[1])
        if dialogue_id == prelogue_id:
            count += 1
            if count >= opt.dia_maxlength:
                global_id += 1
                count = 0
        else:
            global_id += 1
            count = 0
            prelogue_id = dialogue_id
            
        utterance_id = int(line[2])
        speaker = line[3].lower().strip()
        sentence = line[4].lower().strip()
        if 'en' in opt.dataset:
            sentence = re.sub('\x92', '\'', sentence)
            sentence = Tokenizer4Bert.split_text(sentence)
            sentence = re.sub('\"', '', sentence)
        else:
            sentence = re.sub('-{1,}', ' ', sentence)
            sentence = re.sub(' {2,}', ' ', sentence)
        try:
            polarity = int(line[5])
        except:
            polarity = 0

        data = {'speaker': speaker, 'sentence': sentence, 'polarity': polarity}
        if global_id not in all_data:
            all_data[global_id] = []
            all_data[global_id].append(data)
        else:
            all_data[global_id].append(data)
        # data = {'dialogue_id': dialogue_id, 'speaker': speaker, 'sentence': sentence, 'polarity': polarity}
        # all_data.append(data)
    # pprint(all_data)
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
        for ch in ["\'s", "\'ve", "n\'t", "\'re", "\'m", "\'d", "\'ll", ",", ".", "!", "*", "/", "?", "(", ")", "\"", "-", ":"]:
            text = text.replace(ch, " "+ch+" ")
        # return text.strip().split()
        return text


class BertSentenceDataset(Dataset):
    ''' PyTorch standard dataset class '''
    def __init__(self, fname, tokenizer, target_dim, opt):
        data = list()
        if opt.datatype == 'transdata':
            logger.info('datatype:{}'.format(opt.datatype))
            parse = parse_transdata
            for obj in parse(fname):
                speaker_bert_indices = tokenizer.text_to_sequence("[CLS] " + obj['sentence_post'] + " [SEP]")
                # speaker_pre_indices = tokenizer.text_to_sequence(obj['speaker_pre'])
                # speaker_post_indices = tokenizer.text_to_sequence(obj['speaker_post'])
                sentence_pre_indices = tokenizer.text_to_sequence(obj['sentence_pre'])
                sentence_post_indices = tokenizer.text_to_sequence(obj['sentence_post'])
                sentence_post_bert_indices = tokenizer.text_to_sequence("[CLS] " + obj['sentence_post'] + " [SEP]")
                sentence_pair_bert_indices = tokenizer.text_to_sequence("[CLS] " + obj['sentence_pre'] + " [SEP] " + obj['sentence_post'] + " [SEP]")
                sentence_pair_bert_indices_speaker = tokenizer.text_to_sequence("[CLS] " + obj['speaker_pre'] + ' ' + obj['sentence_pre'] + " [SEP] " + obj['speaker_post'] + ' ' + obj['sentence_post'] + " [SEP]")
                sentence_pair_bert_indices_speaker_reverse = tokenizer.text_to_sequence("[CLS] " + obj['speaker_post'] + ' ' + obj['sentence_post'] + " [SEP] " + obj['speaker_pre'] + ' ' + obj['sentence_pre'] + " [SEP]")
                # sentence_post_len = np.sum(sentence_post_indices != 0)
                bert_segments_ids = np.asarray([0] * (np.sum(sentence_pre_indices != 0) + 2) + [1] * (np.sum(sentence_post_indices != 0) + 1))
                bert_segments_ids = tokenizer.pad_sequence(bert_segments_ids, 0, tokenizer.max_length)
                bert_segments_ids_reverse = np.asarray([0] * (np.sum(sentence_post_indices != 0) + 2) + [1] * (np.sum(sentence_pre_indices != 0) + 1))
                bert_segments_ids_reverse = tokenizer.pad_sequence(bert_segments_ids_reverse, 0, tokenizer.max_length)
                polarity = obj['polarity']
                dialogue_id = obj['dialogue_id']
                attention_mask = np.asarray([1] * np.sum(sentence_post_bert_indices != 0) + [0] * (opt.max_length - np.sum(sentence_post_bert_indices != 0)))
                attention_mask_pair = np.asarray([1] * np.sum(sentence_pair_bert_indices != 0) + [0] * (opt.max_length - np.sum(sentence_pair_bert_indices != 0)))
                data.append(
                    {
                        'sentence_bert_indices': sentence_post_bert_indices,
                        'sentence_pair_bert_indices': sentence_pair_bert_indices,
                        'sentence_pair_bert_indeces_reverse': sentence_pair_bert_indices_speaker_reverse,
                        'bert_segments_ids': bert_segments_ids,
                        'bert_segments_ids_reverse': bert_segments_ids_reverse,
                        'attention_mask': attention_mask,
                        'attention_mask_pair': attention_mask_pair,
                        'polarity': polarity,
                        'dialogue_id': dialogue_id,
                        'speaker_bert_indices': speaker_bert_indices
                    }
                )

        elif opt.datatype == 'diadata':
            logger.info('datatype:{}'.format(opt.datatype))
            parse = parse_rawdata_dialogue
            for key, value in parse(fname).items():
                dialogue_data = []
                dialogue_id = key
                for term in value:
                    sentence_indices = tokenizer.text_to_sequence(term['sentence'])
                    sentence_bert_indices = tokenizer.text_to_sequence("[CLS] " + term['sentence'] + " [SEP]")
                    speaker_bert_indices = tokenizer.text_to_sequence("[CLS] " + term['speaker'] + " [SEP]")
                    sentence_speaker_bert_indices = tokenizer.text_to_sequence("[CLS] " + term['speaker'] + " [SEP] " + term['sentence'] + " [SEP]")
                    sentence_speaker_bert_indices_reverse = tokenizer.text_to_sequence("[CLS] " + term['sentence'] + " [SEP] " + term['speaker'] + " [SEP]")
                    speaker_indices = tokenizer.text_to_sequence(term['speaker'])
                    bert_segments_ids = np.asarray([0] * (np.sum(speaker_indices != 0) + 2) + [1] * (np.sum(sentence_indices != 0) + 1))
                    bert_segments_ids = tokenizer.pad_sequence(bert_segments_ids, 0, tokenizer.max_length)
                    bert_segments_ids_reverse = np.asarray([0] * (np.sum(sentence_indices != 0) + 2) + [1] * (np.sum(speaker_indices != 0) + 1))
                    bert_segments_ids_reverse = tokenizer.pad_sequence(bert_segments_ids_reverse, 0, tokenizer.max_length)
                    polarity = term['polarity']
                    # dialogue_id = term['dialogue_id']
                    attention_mask = np.asarray([1] * np.sum(sentence_bert_indices != 0) + [0] * (opt.max_length - np.sum(sentence_bert_indices != 0)))
                    attention_mask_pair = np.asarray([1] * np.sum(sentence_speaker_bert_indices != 0) + [0] * (opt.max_length - np.sum(sentence_speaker_bert_indices != 0)))
                    dialogue_data.append(
                        {
                            'sentence_bert_indices': sentence_bert_indices,
                            'sentence_pair_bert_indices': sentence_speaker_bert_indices,
                            'sentence_pair_bert_indeces_reverse': sentence_speaker_bert_indices_reverse,
                            'bert_segments_ids': bert_segments_ids,
                            'bert_segments_ids_reverse': bert_segments_ids_reverse,
                            'attention_mask': attention_mask,
                            'attention_mask_pair': attention_mask_pair,
                            'polarity': polarity,
                            # 'dialogue_id': dialogue_id,
                            'speaker_bert_indices': speaker_bert_indices
                        }
                    )

                data.append(dialogue_data)

        elif opt.datatype == 'raw':
            logger.info('datatype:{}'.format(opt.datatype))
            parse = parse_rawdata
            for obj in parse(fname):
                sentence_indices = tokenizer.text_to_sequence(obj['sentence'])
                sentence_bert_indices = tokenizer.text_to_sequence("[CLS] " + obj['sentence'] + " [SEP]")
                speaker_bert_indices = tokenizer.text_to_sequence("[CLS] " + obj['speaker'] + " [SEP]")
                sentence_speaker_bert_indices = tokenizer.text_to_sequence("[CLS] " + obj['speaker'] + " [SEP] " + obj['sentence'] + " [SEP]")
                sentence_speaker_bert_indices_reverse = tokenizer.text_to_sequence("[CLS] " + obj['sentence'] + " [SEP] " + obj['speaker'] + " [SEP]")
                speaker_indices = tokenizer.text_to_sequence(obj['speaker'])
                bert_segments_ids = np.asarray([0] * (np.sum(speaker_indices != 0) + 2) + [1] * (np.sum(sentence_indices != 0) + 1))
                bert_segments_ids = tokenizer.pad_sequence(bert_segments_ids, 0, tokenizer.max_length)
                bert_segments_ids_reverse = np.asarray([0] * (np.sum(sentence_indices != 0) + 2) + [1] * (np.sum(speaker_indices != 0) + 1))
                bert_segments_ids_reverse = tokenizer.pad_sequence(bert_segments_ids_reverse, 0, tokenizer.max_length)
                polarity = obj['polarity']
                dialogue_id = obj['dialogue_id']

                attention_mask = np.asarray([1] * np.sum(sentence_bert_indices != 0) + [0] * (opt.max_length - np.sum(sentence_bert_indices != 0)))
                attention_mask_pair = np.asarray([1] * np.sum(sentence_speaker_bert_indices != 0) + [0] * (opt.max_length - np.sum(sentence_speaker_bert_indices != 0)))
                data.append(
                    {
                        'sentence_bert_indices': sentence_bert_indices,
                        'sentence_pair_bert_indices': sentence_speaker_bert_indices,
                        'sentence_pair_bert_indeces_reverse': sentence_speaker_bert_indices_reverse,
                        'bert_segments_ids': bert_segments_ids,
                        'bert_segments_ids_reverse': bert_segments_ids_reverse,
                        'attention_mask': attention_mask,
                        'attention_mask_pair': attention_mask_pair,
                        'polarity': polarity,
                        'dialogue_id': dialogue_id,
                        'speaker_bert_indices': speaker_bert_indices
                    }
                )

        else:
            logger.info('datatype:{}'.format(opt.datatype))
            parse = parse_data
            for obj in parse(fname):
                sentence_indices = tokenizer.text_to_sequence(obj['sentence'])
                sentence_bert_indices = tokenizer.text_to_sequence("[CLS] " + obj['sentence'] + " [SEP]")
                speaker_bert_indices = tokenizer.text_to_sequence("[CLS] " + obj['speaker'] + " [SEP]")
                sentence_speaker_bert_indices = tokenizer.text_to_sequence("[CLS] " + obj['speaker'] + " [SEP] " + obj['sentence'] + " [SEP]")
                sentence_speaker_bert_indices_reverse = tokenizer.text_to_sequence("[CLS] " + obj['sentence'] + " [SEP] " + obj['speaker'] + " [SEP]")
                speaker_indices = tokenizer.text_to_sequence(obj['speaker'])
                # bert_segments_ids = np.asarray([0] * (np.sum(speaker_indices != 0) + 2) + [1] * (np.sum(sentence_indices != 0) + 1))
                bert_segments_ids = np.asarray([0] * (np.sum(speaker_indices != 0) + 2) + [1] * (np.sum(sentence_indices != 0) + 1))
                bert_segments_ids = tokenizer.pad_sequence(bert_segments_ids, 0, tokenizer.max_length)
                bert_segments_ids_reverse = np.asarray([0] * (np.sum(sentence_indices != 0) + 2) + [1] * (np.sum(speaker_indices != 0) + 1))
                bert_segments_ids_reverse = tokenizer.pad_sequence(bert_segments_ids_reverse, 0, tokenizer.max_length)
                polarity = obj['polarity']
                dialogue_id = obj['dialogue_id']

                attention_mask = np.asarray([1] * np.sum(sentence_bert_indices != 0) + [0] * (opt.max_length - np.sum(sentence_bert_indices != 0)))
                attention_mask_pair = np.asarray([1] * np.sum(sentence_speaker_bert_indices != 0) + [0] * (opt.max_length - np.sum(sentence_speaker_bert_indices != 0)))
                data.append(
                    {
                        'sentence_bert_indices': sentence_bert_indices,
                        'sentence_pair_bert_indices': sentence_speaker_bert_indices,
                        'sentence_pair_bert_indeces_reverse': sentence_speaker_bert_indices_reverse,
                        'bert_segments_ids': bert_segments_ids,
                        'bert_segments_ids_reverse': bert_segments_ids_reverse,
                        'attention_mask': attention_mask,
                        'attention_mask_pair': attention_mask_pair,
                        'polarity': polarity,
                        'dialogue_id': dialogue_id,
                        'speaker_bert_indices': speaker_bert_indices
                    }
                )

        self._data = data
    
    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)


def collate_wrapper(batch):
    sentence_bert_indices = torch.LongTensor([item['sentence_bert_indices'] for item in batch[0]]).detach()
    sentence_pair_bert_indices = torch.LongTensor([item['sentence_pair_bert_indices'] for item in batch[0]]).detach()
    sentence_pair_bert_indeces_reverse = torch.LongTensor([item['sentence_pair_bert_indeces_reverse'] for item in batch[0]]).detach()
    bert_segments_ids = torch.LongTensor([item['bert_segments_ids'] for item in batch[0]]).detach()
    attention_mask = torch.LongTensor([item['attention_mask'] for item in batch[0]]).detach()
    attention_mask_pair = torch.LongTensor([item['attention_mask_pair'] for item in batch[0]]).detach()
    polarity = torch.LongTensor([item['polarity'] for item in batch[0]]).detach()
    speaker_bert_indices = torch.LongTensor([item['speaker_bert_indices'] for item in batch[0]]).detach()
    data = {
            'sentence_bert_indices': sentence_bert_indices,
            'sentence_pair_bert_indices': sentence_pair_bert_indices,
            'sentence_pair_bert_indeces_reverse': sentence_pair_bert_indeces_reverse,
            'bert_segments_ids': bert_segments_ids,
            'attention_mask': attention_mask,
            'attention_mask_pair': attention_mask_pair,
            'polarity': polarity,
            # 'dialogue_id': dialogue_id,
            'speaker_bert_indices': speaker_bert_indices
        }
    return data