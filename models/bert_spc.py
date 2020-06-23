'''
@Description: 
@version: 
@Author: chenhao
@Date: 2020-06-23 20:06:21
@LastEditors: chenhao
@LastEditTime: 2020-06-23 20:06:32
'''

import torch
import torch.nn as nn


class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        # self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        layers = [nn.Linear(
            opt.bert_dim, 256), nn.ReLU(), nn.Linear(256, opt.polarities_dim)]
        self.dense = nn.Sequential(*layers)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        _, pooled_output = self.bert(text_bert_indices, bert_segments_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits