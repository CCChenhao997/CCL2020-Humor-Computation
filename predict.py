import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import strftime, localtime
import argparse
import random
from sklearn import metrics
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import BertModel, AutoModel, XLMRobertaModel, GPT2Model, RobertaModel
from utils.data_utils import Tokenizer4Bert, BertSentenceDataset
from config import opt


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
        prob = []
        with torch.no_grad():
            for batch, sample_batched in enumerate(self.test_dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                outputs = F.softmax(outputs, dim=-1)
                predict.extend(torch.argmax(outputs, -1).cpu().numpy().tolist())
                outputs_array = outputs.cpu().numpy().tolist()
                for element in outputs_array:
                    # element = np.around(element, 4)
                    # prob.append(list(element))
                    prob.append(element)
        return predict, prob

    def save_prob_csv(self, id, predict, prob, save_path):
        df = pd.DataFrame({'ID':id,'Label':predict, 'Prob_0':[i[0] for i in prob], 'Prob_1':[i[1] for i in prob]})
        df['Prob_0'] = df['Prob_0'].round(4)
        df['Prob_1'] = df['Prob_1'].round(4)
        df.to_csv(save_path, index=False, sep=',')

if __name__=="__main__":
    # torch.set_printoptions(precision=3, threshold=float("inf"), edgeitems=None, linewidth=300, profile=None)

    model_state_dict_paths = {
        
        'en':{
            # '0': './recorder/第六周/en_aug_0721/bert_spc_batchsize=8_logstep=5/bert_spc_en_fold_0_aug_f1_0.6751_f1_0_0.8309_f1_1_0.5193_acc_0.7498_score_1.2691',
            # '1': './recorder/第六周/en_aug_0721/bert_spc_batchsize=8_logstep=5/bert_spc_en_fold_1_aug_f1_0.6571_f1_0_0.7906_f1_1_0.5235_acc_0.7090_score_1.2326',
            # '2': './recorder/第六周/en_aug_0721/bert_spc_batchsize=8_logstep=5/bert_spc_en_fold_2_aug_f1_0.6888_f1_0_0.8415_f1_1_0.5361_acc_0.7637_score_1.2999',
            # '3': './recorder/第六周/en_aug_0721/bert_spc_batchsize=8_logstep=5/bert_spc_en_fold_3_aug_f1_0.6959_f1_0_0.8388_f1_1_0.5530_acc_0.7631_score_1.3161',
            # '4': './recorder/第六周/en_aug_0721/bert_spc_batchsize=8_logstep=5/bert_spc_en_fold_4_aug_f1_0.6833_f1_0_0.8283_f1_1_0.5383_acc_0.7497_score_1.2879',
            # '0': './recorder/第六周/en_dia_aug/bert_spc_logstep=5_Linear/bert_spc_en_fold_0_dia_aug_f1_0.6982_f1_0_0.8512_f1_1_0.5452_acc_0.7758_score_1.3210',
            # '1': './recorder/第六周/en_dia_aug/bert_spc_logstep=5_Linear/bert_spc_en_fold_1_dia_aug_f1_0.6796_f1_0_0.8189_f1_1_0.5403_acc_0.7401_score_1.2804',
            # '2': './recorder/第六周/en_dia_aug/bert_spc_logstep=5_Linear/bert_spc_en_fold_2_dia_aug_f1_0.6600_f1_0_0.8175_f1_1_0.5024_acc_0.7330_score_1.2354',
            # '3': './recorder/第六周/en_dia_aug/bert_spc_logstep=5_Linear/bert_spc_en_fold_3_dia_aug_f1_0.6749_f1_0_0.8447_f1_1_0.5050_acc_0.7636_score_1.2686',
            # '4': './recorder/第六周/en_dia_aug/bert_spc_logstep=5_Linear/bert_spc_en_fold_4_dia_aug_f1_0.7035_f1_0_0.8346_f1_1_0.5724_acc_0.7615_score_1.3339',
            # '0': './recorder/第六周/en_dia_aug/bert_spc_logstep=5_nn.Linear_waccscore/bert_spc_en_fold_0_dia_aug_f1_0.6987_f1_0_0.8565_f1_1_0.5409_acc_0.7813_score_1.3223',
            # '1': './recorder/第六周/en_dia_aug/bert_spc_logstep=5_nn.Linear_waccscore/bert_spc_en_fold_1_dia_aug_f1_0.6780_f1_0_0.8132_f1_1_0.5427_acc_0.7348_score_1.2775',
            # '2': './recorder/第六周/en_dia_aug/bert_spc_logstep=5_nn.Linear_waccscore/bert_spc_en_fold_2_dia_aug_f1_0.6689_f1_0_0.8399_f1_1_0.4980_acc_0.7572_score_1.2552',
            # '3': './recorder/第六周/en_dia_aug/bert_spc_logstep=5_nn.Linear_waccscore/bert_spc_en_fold_3_dia_aug_f1_0.6718_f1_0_0.8428_f1_1_0.5007_acc_0.7609_score_1.2616',
            # '4': './recorder/第六周/en_dia_aug/bert_spc_logstep=5_nn.Linear_waccscore/bert_spc_en_fold_4_dia_aug_f1_0.7157_f1_0_0.8552_f1_1_0.5761_acc_0.7841_score_1.3602',
            '0': './recorder/第六周/en_dia_aug/bert_spc_logstep=5_nn.Linear_wd=0.01_waccscore/bert_spc_en_fold_0_dia_aug_f1_0.6985_f1_0_0.8448_f1_1_0.5521_acc_0.7695_score_1.3216',
            '1': './recorder/第六周/en_dia_aug/bert_spc_logstep=5_nn.Linear_wd=0.01_waccscore/bert_spc_en_fold_1_dia_aug_f1_0.6814_f1_0_0.8158_f1_1_0.5469_acc_0.7381_score_1.2850',
            # '2': './recorder/第六周/en_dia_aug/bert_spc_logstep=5_nn.Linear_waccscore/bert_spc_en_fold_2_dia_aug_f1_0.6689_f1_0_0.8399_f1_1_0.4980_acc_0.7572_score_1.2552',
            '3': './recorder/第六周/en_dia_aug/bert_spc_logstep=5_nn.Linear_wd=0.01_waccscore/bert_spc_en_fold_3_dia_aug_f1_0.6816_f1_0_0.8536_f1_1_0.5096_acc_0.7745_score_1.2842',
            '4': './recorder/第六周/en_dia_aug/bert_spc_logstep=5_nn.Linear_wd=0.01_waccscore/bert_spc_en_fold_4_dia_aug_f1_0.7180_f1_0_0.8673_f1_1_0.5687_acc_0.7970_score_1.3657',
        },

        'cn':{
            '0': './recorder/第六周/pick/cn/bert_spc_cn_fold_0_uuu_aug_f1_0.6662_f1_0_0.7927_f1_1_0.5397_acc_0.7141_score_1.2538',
            '1': './recorder/第六周/pick/cn/bert_spc_cn_fold_1_uuu_aug_f1_0.6535_f1_0_0.7897_f1_1_0.5172_acc_0.7070_score_1.2242',
            '2': './recorder/第六周/pick/cn/bert_spc_cn_fold_2_uuu_aug_f1_0.6541_f1_0_0.7851_f1_1_0.5232_acc_0.7037_score_1.2269',
             # '3': './recorder/第五周/pick_0719/cn/after_pseudo/bert_spc_lay_cn_fold_3_f1_0.6481_f1_0_0.7767_f1_1_0.5196_score_1.2147',
            '4': './recorder/第六周/pick/cn/bert_spc_cn_fold_4_uuu_aug_f1_0.6581_f1_0_0.7787_f1_1_0.5375_acc_0.7006_score_1.2381',
        }
        
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
    predict_label, prob = inf.evaluate()
    # print(prob)
    print(len(prob))
    # exit()

    id = [i for i in range(len(predict_label))]

    predict_df = pd.DataFrame(list(zip(id, predict_label)))

    if opt.pseudo != "False":
        save_path = "./predict_data/{}_{}_pseudo/{}".format(opt.model_name, opt.date, opt.dataset)
    else:
        save_path = "./predict_data/{}_{}/{}".format(opt.model_name, opt.date, opt.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path, mode=0o777)
    
    file_path = "{}/{}-{}-fold-{}.csv".format(save_path, opt.model_name, opt.dataset, opt.fold_n)
    save_prob_path = "{}/{}-{}-prob-{}.csv".format(save_path, opt.model_name, opt.dataset, opt.fold_n)
    # predict_df.to_csv(file_path, index=None, header=['ID', 'Label'])
    inf.save_prob_csv(id, predict_label, prob, save_prob_path)