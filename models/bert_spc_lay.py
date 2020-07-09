import torch
import torch.nn as nn
import torch.nn.functional as F
from models_utils.linear_helper import Linear
from models_utils.attention_helper import SelfAttention
from models_utils.rnn_helper import DynamicLSTM, SqueezeEmbedding


class BERT_SPC_Lay(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC_Lay, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.pos_embed = nn.Embedding(opt.max_length, opt.position_dim)
        self.attention = SelfAttention(opt.bert_dim+opt.position_dim, 5)
        self.squeeze_embedding = SqueezeEmbedding()
        # self.dense = Linear(opt.bert_dim, opt.polarities_dim)
        layers = [Linear(
            opt.bert_dim*3, opt.bert_dim), nn.ReLU(), Linear(opt.bert_dim, opt.polarities_dim)]
        self.dense = nn.Sequential(*layers)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask = inputs[0], inputs[1], inputs[2]
        text_len = torch.sum(text_bert_indices != 0, dim=-1)
        position = self.pos_embed(bert_segments_ids)
        sentence_output, pooled_output, all_hidden_states = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)
        # sentence_output = self.dropout(sentence_output)
        # pooled_output = self.dropout(pooled_output)

        last_three_hidden_states = all_hidden_states[-3:]
        concated_layers = torch.cat(last_three_hidden_states, dim=-1)
        concated_layers_cls = concated_layers[:, 0, :]
        concated_layers_cls = self.dropout(concated_layers_cls)

        logits = self.dense(concated_layers_cls)
        return logits