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
    # parser.add_argument('--dataset', default='en', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--learning_rate', default=0.002, type=float)    # 1e-3
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--l2reg', default=1e-5, type=float)    # 1e-5
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    # parser.add_argument('--embed_dim', default=300, type=int)
    # parser.add_argument('--hidden_dim', default=200, type=int)
    # parser.add_argument('--position_dim', default=100, type=int)
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
    }

    model_state_dict_paths = {
        # 'bert_spc': './recorder/bert_spc_0703_adv/en/bert_spc_en_fold_0_score_1.4303',
        # 'bert_spc': './recorder/bert_spc_0703_adv/en/bert_spc_en_fold_1_score_1.4434',
        # 'bert_spc': './recorder/bert_spc_0703_adv/en/bert_spc_en_fold_2_score_1.4481',
        # 'bert_spc': './recorder/bert_spc_0703_adv/en/bert_spc_en_fold_3_score_1.4569',
        # 'bert_spc': './recorder/bert_spc_0703_adv/en/bert_spc_en_fold_4_score_1.4566',

        # 'bert_spc': './recorder/bert_spc_0703_adv/cn/bert_spc_cn_fold_0_score_1.3849',
        # 'bert_spc': './recorder/bert_spc_0703_adv/cn/bert_spc_cn_fold_1_score_1.3520',
        # 'bert_spc': './recorder/bert_spc_0703_adv/cn/bert_spc_cn_fold_2_score_1.3619',
        # 'bert_spc': './recorder/bert_spc_0703_adv/cn/bert_spc_cn_fold_3_score_1.3773',
        # 'bert_spc': './recorder/bert_spc_0703_adv/cn/bert_spc_cn_fold_4_score_1.3710',

        # * 伪标签模型
        # 'bert_spc': './recorder/bert_spc_0703_adv_pseudo/en/bert_spc_en_fold_0_score_1.4734',
        # 'bert_spc': './recorder/bert_spc_0703_adv_pseudo/en/bert_spc_en_fold_1_score_1.4553',
        # 'bert_spc': './recorder/bert_spc_0703_adv_pseudo/en/bert_spc_en_fold_2_score_1.4593',
        # 'bert_spc': './recorder/bert_spc_0703_adv_pseudo/en/bert_spc_en_fold_3_score_1.4274',
        # 'bert_spc': './recorder/bert_spc_0703_adv_pseudo/en/bert_spc_en_fold_4_score_1.4384',

        # 'bert_spc': './recorder/bert_spc_0703_adv_pseudo/cn/bert_spc_cn_fold_0_score_1.3920',
        # 'bert_spc': './recorder/bert_spc_0703_adv_pseudo/cn/bert_spc_cn_fold_1_score_1.4059',
        # 'bert_spc': './recorder/bert_spc_0703_adv_pseudo/cn/bert_spc_cn_fold_2_score_1.4094',
        # 'bert_spc': './recorder/bert_spc_0703_adv_pseudo/cn/bert_spc_cn_fold_3_score_1.3736',
        # 'bert_spc': './recorder/bert_spc_0703_adv_pseudo/cn/bert_spc_cn_fold_4_score_1.4310',
        
    }

    dataset_files = {
        'cn': {
            'test': './data/preprocess/cn_test.tsv'
        },
        'en': {
            'test': './data/preprocess/en_test.tsv'
        }
    }
    
    input_colses = {
        'bert': ['sentence_bert_indices', 'attention_mask'],
        'bert_att': ['sentence_bert_indices', 'attention_mask'],
        'bert_spc': ['sentence_pair_bert_indices', 'bert_segments_ids', 'attention_mask_pair'],
        'bert_spc_att': ['sentence_pair_bert_indices', 'bert_segments_ids', 'attention_mask_pair'],
        'bert_spc_pos': ['sentence_pair_bert_indices', 'bert_segments_ids', 'attention_mask_pair'],
        'bert_spc_cap': ['sentence_pair_bert_indices', 'bert_segments_ids', 'attention_mask_pair'],
    }

    opt = get_parameters()
    #! 注意
    opt.dataset = 'cn'
    date = "0705"
    fold_n = 4
    pseudo = True
    #! 模型路径
    # opt.pretrained_bert_name = 'bert-base-uncased'
    opt.pretrained_bert_name = './pretrain_models/ERNIE_cn'

    #! max_length
    if opt.dataset == 'cn':
        opt.max_length = 128

    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.model_class = model_classes[opt.model_name]
    opt.state_dict_path = model_state_dict_paths[opt.model_name]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inf = Inferer(opt)
    predict_label = inf.evaluate()
    id = [i for i in range(len(predict_label))]

    predict_df = pd.DataFrame(list(zip(id, predict_label)))

    if pseudo:
        save_path = "./predict_data/{}_{}_pseudo/{}".format(opt.model_name, date, opt.dataset)
    else:
        save_path = "./predict_data/{}_{}/{}".format(opt.model_name, date, opt.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path, mode=0o777)
    
    file_path = "{}/{}-{}-fold-{}.csv".format(save_path, opt.model_name, opt.dataset, fold_n)
    predict_df.to_csv(file_path, index=None, header=['ID', 'Label'])