import os
import sys
import torch
import torch.nn as nn
import argparse
import logging
from models.bert import BERT
from models.bert_spc import BERT_SPC
from models.bert_att import BERT_Att
from models.bert_spc_att import BERT_SPC_Att
from models.bert_spc_pos import BERT_SPC_Pos
from models.bert_spc_cap import BERT_SPC_Cap
from models.bert_spc_lay import BERT_SPC_Lay
from models.bert_spc_rnn import BERT_SPC_RNN
from models.bert_spc_cnn import BERT_SPC_CNN
from models.lcf_bert import LCF_BERT
from models.bert_bag_cnn import BERT_BAG_CNN

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

model_classes = {
        'bert': BERT,
        'bert_att': BERT_Att,
        'bert_spc': BERT_SPC,
        'bert_spc_att': BERT_SPC_Att,
        'bert_spc_pos': BERT_SPC_Pos,
        'bert_spc_cap': BERT_SPC_Cap,
        'bert_spc_lay': BERT_SPC_Lay,
        'bert_spc_rnn': BERT_SPC_RNN,
        'bert_spc_cnn': BERT_SPC_CNN,
        'lcf_bert': LCF_BERT,
        'bert_bag_cnn':BERT_BAG_CNN
    }

input_colses = {
        'bert': ['sentence_bert_indices', 'attention_mask'],
        'bert_att': ['sentence_bert_indices', 'attention_mask'],
        'bert_spc': ['sentence_pair_bert_indices', 'bert_segments_ids', 'attention_mask_pair'],
        'bert_spc_att': ['sentence_pair_bert_indices', 'bert_segments_ids', 'attention_mask_pair'],
        'bert_spc_pos': ['sentence_pair_bert_indices', 'bert_segments_ids', 'attention_mask_pair'],
        'bert_spc_cap': ['sentence_pair_bert_indices', 'bert_segments_ids', 'attention_mask_pair'],
        'bert_spc_lay': ['sentence_pair_bert_indices', 'bert_segments_ids', 'attention_mask_pair'],
        'bert_spc_rnn': ['sentence_pair_bert_indices', 'bert_segments_ids', 'attention_mask_pair'],
        'bert_spc_cnn': ['sentence_pair_bert_indices', 'bert_segments_ids', 'attention_mask_pair'],
        'bert_bag_cnn': ['sentence_pair_bert_indices', 'bert_segments_ids', 'attention_mask_pair'],
        'lcf_bert': ['sentence_pair_bert_indices', 'bert_segments_ids', 'sentence_bert_indices', 'speaker_bert_indices', 'attention_mask_pair'],
    }
    
initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,    # default lr=0.01
        'adam': torch.optim.Adam,          # default lr=0.001
        'adamax': torch.optim.Adamax,      # default lr=0.002
        'asgd': torch.optim.ASGD,          # default lr=0.01
        'rmsprop': torch.optim.RMSprop,    # default lr=0.01
        'sgd': torch.optim.SGD,
    }


# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='bert_spc', type=str, help=', '.join(model_classes.keys()))
parser.add_argument('--dataset', default='cn', type=str)
parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
parser.add_argument('--learning_rate', default=0.002, type=float)    # 1e-3
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--l2reg', default=1e-5, type=float)    # 1e-5
parser.add_argument('--num_epoch', default=20, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--log_step', default=5, type=int)
# parser.add_argument('--embed_dim', default=300, type=int)
parser.add_argument('--hidden_dim', default=300, type=int)
parser.add_argument('--position_dim', default=100, type=int)
parser.add_argument('--polarities_dim', default=2, type=int, help='2')
parser.add_argument('--max_length', default=80, type=int)
parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
parser.add_argument('--repeats', default=1, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--bert_dim', default=768, type=int)
parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
parser.add_argument("--weight_decay", default=0.00, type=float, help="Weight deay if we apply some.") # 0.01
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument('--cross_val_fold', default=5, type=int, help='k-fold cross validation')
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='clip gradients at this value')
parser.add_argument('--cuda', default=0, type=str)
parser.add_argument('--datatype', default=None, type=str, choices=['transdata', 'diadata'])
parser.add_argument('--attention_hops', default=5, type=int)
parser.add_argument('--adv_type', default=None, type=str, help='fgm, pgd')
parser.add_argument('--criterion', default=None, type=str, help='loss choice', choices=['focalloss', None])
parser.add_argument('--fp16', default=False, type=bool)
parser.add_argument('--fp16_opt_level', default='O1', help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
parser.add_argument('--filter_sizes', default=(2, 3, 4), type=tuple)
parser.add_argument('--num_filters', default=256, type=int)
parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')
parser.add_argument('--date', default='date', type=str)
parser.add_argument('--fold_n', default=0, type=int)
parser.add_argument('--pseudo', default='False', type=str)
parser.add_argument('--rnntype', default='LSTM', type=str, choices=['LSTM', 'GRU', 'RNN'])
opt = parser.parse_args()
opt.model_class = model_classes[opt.model_name]
opt.inputs_cols = input_colses[opt.model_name]
opt.initializer = initializers[opt.initializer]
opt.optimizer = optimizers[opt.optimizer]
os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)