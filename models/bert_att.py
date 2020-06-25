import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, r):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.r = r
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(True),
            # nn.Tanh(),
            nn.Linear(128, self.r)
        )
        # self.weight_n = nn.Parameter(torch.Tensor(opt.hidden_dim*2, 128))
        # self.bias_n = nn.Parameter(torch.Tensor(1))
        # self.fc = nn.Linear(10, 1)

    def forward(self, encoder_outputs): # torch.Size([32, 66, 400])

        energy = self.projection(encoder_outputs)       # torch.Size([32, 66, 5])

        weights = F.softmax(energy.squeeze(-1), dim=1)  # torch.Size([32, 66, 5])
        weights = weights.transpose(1,2)              # torch.Size([32, 5, 66])
        outputs = weights @ encoder_outputs           # torch.Size([32, 5, 400])

        # outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)  # torch.Size([32, 5, 400])
        outputs = torch.sum(outputs, 1) / self.r          # torch.Size([32, 400])
       
        return outputs, weights


class BERT_Att(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_Att, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.attention = SelfAttention(opt.bert_dim, 5)
        # self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        layers = [nn.Linear(
            opt.bert_dim*2, opt.bert_dim), nn.ReLU(), nn.Linear(opt.bert_dim, opt.polarities_dim)]
        self.dense = nn.Sequential(*layers)

    def forward(self, inputs):
        text_bert_indices, attention_mask = inputs[0], inputs[1]
        sentence_output, pooled_output = self.bert(text_bert_indices, attention_mask=attention_mask)
        sentence_output = self.dropout(sentence_output)
        pooled_output = self.dropout(pooled_output)

        att_hidden, attn_weights = self.attention(sentence_output)
        final_hidden = torch.cat((pooled_output, pooled_output), dim=-1)
        logits = self.dense(final_hidden)
        return logits