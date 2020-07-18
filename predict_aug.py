import os
import torch
import torch.nn as nn
from time import strftime, localtime
import argparse
import random
from sklearn import metrics
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import BertModel, AutoModel, XLMRobertaModel, GPT2Model, RobertaModel
from models.bert import BERT
from models.bert_spc import BERT_SPC
from models.bert_att import BERT_Att
from models.bert_spc_att import BERT_SPC_Att
from models.bert_spc_pos import BERT_SPC_Pos
from models.bert_spc_cap import BERT_SPC_Cap
from models.bert_spc_lay import BERT_SPC_Lay
from models.bert_spc_rnn import BERT_SPC_RNN
from models.bert_spc_cnn import BERT_SPC_CNN
from data_utils import Tokenizer4Bert, BertSentenceDataset


class Inferer:
    ''' Model training and evaluation '''
    def __init__(self, opt):
        self.opt = opt
        tokenizer = Tokenizer4Bert(opt.max_length, opt.pretrained_bert_name)
        bert_model = BertModel.from_pretrained(opt.pretrained_bert_name, output_hidden_states=True)
        self.pretrained_bert_state_dict = bert_model.state_dict()
        self.model = opt.model_class(bert_model, opt).to(opt.device)

        print('loading model {0} ...'.format(opt.model_name))
        
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model.to(opt.device)
        
        torch.autograd.set_grad_enabled(False)

        testset = BertSentenceDataset(opt.dataset_file['test'], tokenizer, target_dim=self.opt.polarities_dim, opt=opt)
        self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size, shuffle=False)

    def evaluate(self):
        self.model.eval()
        predict = []
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_dataloader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_outputs = self.model(t_inputs)
                predict.extend(torch.argmax(t_outputs, -1).cpu().numpy().tolist())
        return predict


def get_parameters():
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_spc', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='en', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--date', default='date', type=str)
    parser.add_argument('--fold_n', default=0, type=str)
    parser.add_argument('--pseudo', default='False', type=str)
    
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
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    # parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--cross_val_fold', default=5, type=int, help='k-fold cross validation')
    # parser.add_argument('--grad_clip', type=float, default=10, help='clip gradients at this value')
    parser.add_argument('--cuda', default=0, type=str)
    parser.add_argument('--transdata', default=False, type=bool)
    parser.add_argument('--attention_hops', default=5, type=int)
    parser.add_argument('--adv_type', default=None, type=str, help='fgm, pgd')
    parser.add_argument('--fp16', default=False, type=bool)
    parser.add_argument('--fp16_opt_level', default='O1', help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
    parser.add_argument('--filter_sizes', default=(2, 3, 4), type=tuple)
    parser.add_argument('--num_filters', default=256, type=int)
    opt = parser.parse_args()
    return opt


if __name__=="__main__":
    # torch.set_printoptions(precision=3, threshold=float("inf"), edgeitems=None, linewidth=300, profile=None)
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
    }

    model_state_dict_paths = {
        
        'en':{
            '0': './recorder/第五周/en_uuu_ordered/bert_spc_cnn_0718_epoch=5/bert_spc_cnn_en_fold_0_uuu_f1_0.6718_f1_0_0.8318_f1_1_0.5117_acc_0.7498_score_1.2616',
            '1': './recorder/第五周/en_uuu_ordered/bert_spc_cnn_0718_epoch=5/bert_spc_cnn_en_fold_1_uuu_f1_0.6661_f1_0_0.8437_f1_1_0.4886_acc_0.7605_score_1.2491',
            '2': './recorder/第五周/en_uuu_ordered/bert_spc_cnn_0718_epoch=5/bert_spc_cnn_en_fold_2_uuu_f1_0.6798_f1_0_0.8463_f1_1_0.5132_acc_0.7664_score_1.2796',
            '3': './recorder/第五周/en_uuu_ordered/bert_spc_cnn_0718_epoch=5/bert_spc_cnn_en_fold_3_uuu_f1_0.6781_f1_0_0.8548_f1_1_0.5015_acc_0.7751_score_1.2766',
            '4': './recorder/第五周/en_uuu_ordered/bert_spc_cnn_0718_epoch=5/bert_spc_cnn_en_fold_4_uuu_f1_0.6779_f1_0_0.8620_f1_1_0.4937_acc_0.7831_score_1.2769',
        },

        'cn':{
            '0': './recorder/pick/cn/lay_adv_711/bert_spc_lay_cn_fold_0_f1_0.6606_f1_0_0.7744_f1_1_0.5469',
            '1': './recorder/pick/cn/lay_adv_711/bert_spc_lay_cn_fold_1_f1_0.6394_f1_0_0.7496_f1_1_0.5292',
            '2': './recorder/pick/cn/lay_adv_711/bert_spc_lay_cn_fold_2_f1_0.6421_f1_0_0.7605_f1_1_0.5236',
            '3': './recorder/pick/cn/lay_adv_711/bert_spc_lay_cn_fold_3_f1_0.6237_f1_0_0.7242_f1_1_0.5231',
            '4': './recorder/pick/cn/lay_adv_711/bert_spc_lay_cn_fold_4_f1_0.6528_f1_0_0.7800_f1_1_0.5255',
        }
    }

    dataset_files = {
        'cn': {
            'test': './data/preprocess/cn_aug.tsv'
        },
        'en': {
            'test': './data/preprocess/en_aug.tsv'
        }
    }

    pretrained_bert_names = {
        'cn':  './pretrain_models/ERNIE_cn',
        'en':  'bert-base-uncased'
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
    }

    opt = get_parameters()
    opt.dataset_file = dataset_files[opt.dataset]
    opt.pretrained_bert_name = pretrained_bert_names[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.model_class = model_classes[opt.model_name]
    opt.state_dict_path = model_state_dict_paths[opt.dataset][opt.fold_n]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inf = Inferer(opt)
    predict_label = inf.evaluate()
    id = [i for i in range(len(predict_label))]

    predict_df = pd.DataFrame(list(zip(id, predict_label)))

    if opt.pseudo != "False":
        save_path = "./predict_data/aug_data/{}_{}_pseudo/{}".format(opt.model_name, opt.date, opt.dataset)
    else:
        save_path = "./predict_data/aug_data/{}_{}/{}".format(opt.model_name, opt.date, opt.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path, mode=0o777)
    
    file_path = "{}/{}-{}-fold-{}.csv".format(save_path, opt.model_name, opt.dataset, opt.fold_n)
    predict_df.to_csv(file_path, index=None, header=['ID', 'Label'])