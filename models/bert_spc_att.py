import torch
import torch.nn as nn
import torch.nn.functional as F
from models_utils.attention_helper import SelfAttention
from models_utils.rnn_helper import DynamicLSTM, SqueezeEmbedding


class BERT_SPC_Att(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC_Att, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.attention = SelfAttention(opt.bert_dim, 5)
        self.squeeze_embedding = SqueezeEmbedding()
        # self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        layers = [nn.Linear(
            opt.bert_dim*2, opt.bert_dim), nn.ReLU(), nn.Linear(opt.bert_dim, opt.polarities_dim)]
        self.dense = nn.Sequential(*layers)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask = inputs[0], inputs[1], inputs[2]
        text_len = torch.sum(text_bert_indices != 0, dim=-1)
        sentence_output, pooled_output = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)
        sentence_output = self.dropout(sentence_output)
        pooled_output = self.dropout(pooled_output)
        sentence_output_mask = bert_segments_ids.unsqueeze(-1) * sentence_output
        att_hidden, attn_weights = self.attention(sentence_output_mask, mask=bert_segments_ids)
        final_hidden = torch.cat((pooled_output, att_hidden), dim=-1)
        logits = self.dense(final_hidden)
        return logits