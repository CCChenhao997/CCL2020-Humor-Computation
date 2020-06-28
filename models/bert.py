import torch
import torch.nn as nn


class BERT(nn.Module):
    def __init__(self, bert, opt):
        super(BERT, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        # self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        layers = [nn.Linear(
            opt.bert_dim, 256), nn.ReLU(), nn.Linear(256, opt.polarities_dim)]
        self.dense = nn.Sequential(*layers)

    def forward(self, inputs):
        text_bert_indices, attention_mask = inputs[0], inputs[1]
        _, pooled_output = self.bert(text_bert_indices, attention_mask=attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits