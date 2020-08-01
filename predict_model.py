import os
import torch
import torch.nn as nn
from time import strftime, localtime
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
        self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.eval_batch_size, shuffle=False)

    def evaluate(self):
        self.model.eval()
        predict = []
        targets_all, outputs_all = None, None
        with torch.no_grad():
            for batch, sample_batched in enumerate(self.test_dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)
                # predict.extend(torch.argmax(outputs, -1).cpu().numpy().tolist())
                targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
                outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs
        
        labels = targets_all.data.cpu()
        predic = torch.argmax(outputs_all, -1).cpu()
        acc = metrics.accuracy_score(labels, predic)
        f1_1 = metrics.f1_score(labels, predic, average='binary')
        return acc, f1_1


if __name__=="__main__":

    model_state_dict_paths = {
        
        'en':{
            '0': './recorder/第六周/en_aug_0721/bert_spc_batchsize=8_logstep=5/bert_spc_en_fold_0_aug_f1_0.6751_f1_0_0.8309_f1_1_0.5193_acc_0.7498_score_1.2691',
            '1': './recorder/第六周/en_aug_0721/bert_spc_batchsize=8_logstep=5/bert_spc_en_fold_1_aug_f1_0.6571_f1_0_0.7906_f1_1_0.5235_acc_0.7090_score_1.2326',
            '2': './recorder/第六周/en_aug_0721/bert_spc_batchsize=8_logstep=5/bert_spc_en_fold_2_aug_f1_0.6888_f1_0_0.8415_f1_1_0.5361_acc_0.7637_score_1.2999',
            '3': './recorder/第六周/en_aug_0721/bert_spc_batchsize=8_logstep=5/bert_spc_en_fold_3_aug_f1_0.6959_f1_0_0.8388_f1_1_0.5530_acc_0.7631_score_1.3161',
            '4': './recorder/第六周/en_aug_0721/bert_spc_batchsize=8_logstep=5/bert_spc_en_fold_4_aug_f1_0.6833_f1_0_0.8283_f1_1_0.5383_acc_0.7497_score_1.2879',
        },

        'cn':{
            '0': './recorder/第五周/pick_0719/cn/after_pseudo/bert_spc_lay_cn_fold_0_uuu_f1_0.6627_f1_0_0.7759_f1_1_0.5496_score_1.2503',
            '1': './recorder/第五周/pick_0719/cn/after_pseudo/bert_spc_cnn_cn_fold_1_f1_0.6513_f1_0_0.7872_f1_1_0.5155_acc_0.7043_score_1.2198',
            '2': './recorder/第五周/pick_0719/cn/after_pseudo/bert_spc_lay_cn_fold_2_f1_0.6523_f1_0_0.7792_f1_1_0.5255_acc_0.6986_score_1.2241',
            '3': './recorder/第五周/pick_0719/cn/after_pseudo/bert_spc_lay_cn_fold_3_f1_0.6481_f1_0_0.7767_f1_1_0.5196_score_1.2147',
            '4': './recorder/第五周/pick_0719/cn/after_pseudo/bert_spc_lay_cn_fold_4_f1_0.6549_f1_0_0.7772_f1_1_0.5327_score_1.2309',
        }
    }

    dataset_files = {
        'cn': {
            'test': './data/preprocess/cn_total.tsv'
        },
        'en': {
            'test': './data/preprocess/en_total.tsv'
        }
    }

    pretrained_bert_names = {
        'cn':  './pretrain_models/ERNIE_cn',
        'en':  'bert-base-uncased'
    }

    opt.dataset_file = dataset_files[opt.dataset]
    opt.pretrained_bert_name = pretrained_bert_names[opt.dataset]
    opt.state_dict_path = model_state_dict_paths[opt.dataset][opt.fold_n]

    inf = Inferer(opt)
    acc, f1_1 = inf.evaluate()
    print("fold: ", opt.fold_n)
    print("acc: ", acc)
    print("f1_1: ", f1_1)