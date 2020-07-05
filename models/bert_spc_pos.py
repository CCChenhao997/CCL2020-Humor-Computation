import torch
import torch.nn as nn
import torch.nn.functional as F
from models_utils.linear_helper import Linear
from models_utils.attention_helper import SelfAttention
from models_utils.rnn_helper import DynamicLSTM, SqueezeEmbedding


class BERT_SPC_Pos(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC_Pos, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.pos_embed = nn.Embedding(opt.max_length, opt.position_dim)
        self.attention = SelfAttention(opt.bert_dim+opt.position_dim, 5)
        self.squeeze_embedding = SqueezeEmbedding()
        # self.dense = Linear(opt.bert_dim, opt.polarities_dim)
        layers = [Linear(
            opt.bert_dim*2+opt.position_dim, opt.bert_dim), nn.ReLU(), Linear(opt.bert_dim, opt.polarities_dim)]
        self.dense = nn.Sequential(*layers)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask = inputs[0], inputs[1], inputs[2]
        text_len = torch.sum(text_bert_indices != 0, dim=-1)
        position = self.pos_embed(bert_segments_ids)
        sentence_output, pooled_output, all_hidden_states = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)
        # sentence_output = self.dropout(sentence_output)
        pooled_output = self.dropout(pooled_output)
        
        p_sentence = torch.cat((sentence_output, position), dim=-1)
        p_sentence = self.squeeze_embedding(p_sentence, text_len)
        p_sentence = self.dropout(p_sentence)

        mask = self.squeeze_embedding(attention_mask, text_len)
        att_hidden, attn_weights = self.attention(p_sentence, mask)
        final_hidden = torch.cat((pooled_output, att_hidden), dim=-1)

        logits = self.dense(final_hidden)
        return logits