import torch
import torch.nn as nn
from models_utils.linear_helper import Linear


class BERT(nn.Module):
    def __init__(self, bert, opt):
        super(BERT, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        # self.dense = Linear(opt.bert_dim, opt.polarities_dim)
        layers = [Linear(
            opt.bert_dim, 256), nn.ReLU(), Linear(256, opt.polarities_dim)]
        self.dense = nn.Sequential(*layers)

    def forward(self, inputs):
        text_bert_indices, attention_mask = inputs[0], inputs[1]
        _, pooled_output, all_hidden_states = self.bert(text_bert_indices, attention_mask=attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        
        return logits