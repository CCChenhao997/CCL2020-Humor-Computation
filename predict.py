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
from data_utils import Tokenizer4Bert, BertSentenceDataset
from config import model_classes, input_colses, initializers, optimizers, opt


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
            for batch, sample_batched in enumerate(self.test_dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                predict.extend(torch.argmax(outputs, -1).cpu().numpy().tolist())
        return predict


if __name__=="__main__":
    # torch.set_printoptions(precision=3, threshold=float("inf"), edgeitems=None, linewidth=300, profile=None)

    model_state_dict_paths = {
        
        'en':{
            '0': './recorder/第五周/en_aug_pseudo/bert_spc_0719_Linear/bert_spc_en_fold_0_aug_pseudo_f1_0.6877_f1_0_0.8413_f1_1_0.5342_acc_0.7632_score_1.2974',
            '1': './recorder/第五周/en_aug_pseudo/bert_spc_0719_Linear/bert_spc_en_fold_1_aug_pseudo_f1_0.6789_f1_0_0.8542_f1_1_0.5037_acc_0.7746_score_1.2783',
            '2': './recorder/第五周/en_aug_pseudo/bert_spc_0719_Linear/bert_spc_en_fold_2_aug_pseudo_f1_0.6864_f1_0_0.8352_f1_1_0.5376_acc_0.7570_score_1.2946',
            '3': './recorder/第五周/en_aug_pseudo/bert_spc_0719_Linear/bert_spc_en_fold_3_aug_pseudo_f1_0.7041_f1_0_0.8627_f1_1_0.5455_acc_0.7892_score_1.3346',
            '4': './recorder/第五周/en_aug_pseudo/bert_spc_0719_Linear/bert_spc_en_fold_4_aug_pseudo_f1_0.6957_f1_0_0.8709_f1_1_0.5205_acc_0.7965_score_1.3170',
        },

        'cn':{
            '0': './recorder/第五周/pick_0719/cn/after_pseudo/bert_spc_lay_cn_fold_0_uuu_f1_0.6627_f1_0_0.7759_f1_1_0.5496_score_1.2503',
            '1': './recorder/第五周/pick_0719/cn/after_pseudo/bert_spc_cnn_cn_fold_1_f1_0.6513_f1_0_0.7872_f1_1_0.5155_acc_0.7043_score_1.2198',
            '2': './recorder/第五周/pick_0719/cn/after_pseudo/bert_spc_lay_cn_fold_2_f1_0.6523_f1_0_0.7792_f1_1_0.5255_acc_0.6986_score_1.2241',
            '3': './recorder/第五周/pick_0719/cn/after_pseudo/bert_spc_lay_cn_fold_3_f1_0.6481_f1_0_0.7767_f1_1_0.5196_score_1.2147',
            '4': './recorder/第五周/pick_0719/cn/after_pseudo/bert_spc_lay_cn_fold_4_f1_0.6549_f1_0_0.7772_f1_1_0.5327_score_1.2309',
        }
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

    pretrained_bert_names = {
        'cn':  './pretrain_models/ERNIE_cn',
        'en':  'bert-base-uncased'
    }
    
    # opt = get_parameters()
    opt.dataset_file = dataset_files[opt.dataset]
    opt.pretrained_bert_name = pretrained_bert_names[opt.dataset]
    opt.state_dict_path = model_state_dict_paths[opt.dataset][opt.fold_n]

    inf = Inferer(opt)
    predict_label = inf.evaluate()
    id = [i for i in range(len(predict_label))]

    predict_df = pd.DataFrame(list(zip(id, predict_label)))

    if opt.pseudo != "False":
        save_path = "./predict_data/{}_{}_pseudo/{}".format(opt.model_name, opt.date, opt.dataset)
    else:
        save_path = "./predict_data/{}_{}/{}".format(opt.model_name, opt.date, opt.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path, mode=0o777)
    
    file_path = "{}/{}-{}-fold-{}.csv".format(save_path, opt.model_name, opt.dataset, opt.fold_n)
    predict_df.to_csv(file_path, index=None, header=['ID', 'Label'])