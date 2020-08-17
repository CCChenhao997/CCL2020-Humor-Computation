import os
import torch
import torch.nn as nn
import argparse
import random
import math
import logging
import sys
import copy
import time
from time import strftime, localtime
from sklearn import metrics
import numpy as np
from transformers import BertModel, AutoModel, RobertaModel, AlbertModel, AutoTokenizer
from transformers import AdamW
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.nn.utils import clip_grad_norm_
from models_utils.loss_helper import FocalLoss
from models_utils.adv_helper import FGM, PGD
from utils.data_utils import Tokenizer4Bert, BertSentenceDataset, get_time_dif, collate_wrapper
from sklearn.model_selection import StratifiedKFold, KFold
from collections import defaultdict
from transformers import get_linear_schedule_with_warmup
from config import opt, logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Instructor:
    ''' Model training and evaluation '''
    def __init__(self, opt):
        self.opt = opt
        tokenizer = Tokenizer4Bert(opt.max_length, opt.pretrained_bert_name)
        # bert_model = AutoModel.from_pretrained(opt.pretrained_bert_name)
        bert_model = BertModel.from_pretrained(opt.pretrained_bert_name, output_hidden_states=True)
        # bert_model = AlbertModel.from_pretrained(opt.pretrained_bert_name)
        # self.pretrained_bert_state_dict = bert_model.state_dict()
        self.model = opt.model_class(bert_model, opt).to(opt.device)
        trainset = BertSentenceDataset(opt.dataset_file['train'], tokenizer, target_dim=self.opt.polarities_dim, opt=opt)
        testset = BertSentenceDataset(opt.dataset_file['test'], tokenizer, target_dim=self.opt.polarities_dim, opt=opt)
        if opt.datatype == 'diadata':
            self.train_dataloader = DataLoader(dataset=trainset, batch_size=opt.train_batch_size, shuffle=True, collate_fn=collate_wrapper)   # , drop_last=True
            self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.eval_batch_size, shuffle=False, collate_fn=collate_wrapper)
        else:
            self.train_dataloader = DataLoader(dataset=trainset, batch_size=opt.train_batch_size, shuffle=True)   # , drop_last=True
            self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.eval_batch_size, shuffle=False)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.opt.device.index)))
        self._print_args()
    
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))    # 计算参数量 torch.prod - Returns the product of all elements in the input tensor.
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('training arguments:')
        
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
    
    def _reset_params(self):
        for name, param in self.model.named_parameters():
            if 'bert' not in name:
                if param.requires_grad:
                    if len(param.shape) > 1:
                        self.opt.initializer(param)
                    else:
                        stdv = 1. / math.sqrt(param.shape[0])
                        torch.nn.init.uniform_(param, a=-stdv, b=stdv)

    def get_bert_optimizer(self, opt, model):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        diff_part = ["bert.embeddings", "bert.encoder"]

        if opt.diff_lr:
            logger.info("layered learning rate on")
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if
                            not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": opt.weight_decay,
                    "lr": opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                            any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                            not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": opt.weight_decay,
                    "lr": opt.layers_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                            any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": opt.layers_lr
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, eps=opt.adam_epsilon)

        else:
            logger.info("bert learning rate on")
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': opt.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters,
                        lr=opt.bert_lr, eps=opt.adam_epsilon)   #  weight_decay=self.opt.l2reg

        return optimizer
    
    def _train(self, model, optimizer, max_test_acc_overall=0, max_w_acc_overall=0, max_f1_overall=0, max_score_overall=0):
        # 对抗训练
        if self.opt.adv_type == 'fgm':
            fgm = FGM(self.model)
        elif self.opt.adv_type == 'pgd':
            pgd = PGD(self.model)
            K = 3

        if self.opt.criterion == 'focalloss':
            logger.info('criterion选择：focalloss')
            criterion = FocalLoss(num_class=opt.polarities_dim, alpha=opt.alpha, gamma=opt.gamma, smooth=opt.smooth)
        else:
            logger.info('criterion选择：CrossEntropyLoss')
            criterion = nn.CrossEntropyLoss()
        
        if 'bert' in self.opt.model_name:
            optimizer = self.get_bert_optimizer(self.opt, self.model)
        else:
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        if self.opt.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.opt.fp16_opt_level)

        if opt.scheduler:
            logger.info('使用scheduler')
            num_training_steps = len(self.train_dataloader) * self.opt.num_epoch
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_training_steps*0.1, num_training_steps=num_training_steps)

        max_test_acc, max_w_acc, max_f1, max_f1_1, max_score, global_step = 0, 0, 0, 0, 0, 0
        max_weight_score = 0
        
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 60)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                
                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets) 

                if self.opt.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # loss.backward()

                if self.opt.adv_type == 'fgm':
                    fgm.attack()  ##对抗训练
                    outputs = self.model(inputs)
                    loss_adv = criterion(outputs, targets)
                    loss_adv.backward()
                    fgm.restore()

                if self.opt.adv_type == 'pgd':
                    pgd.backup_grad()
                    for t in range(K):
                        pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                        if t != K-1:
                            self.model.zero_grad()
                        else:
                            pgd.restore_grad()
                        outputs = self.model(inputs)
                        loss_adv = criterion(outputs, targets)
                        loss_adv.backward()              # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    pgd.restore()                        # 恢复embedding参数

                optimizer.step()
                if opt.scheduler:
                    scheduler.step()
                
                if global_step % self.opt.log_step == 0:    # 每隔opt.log_step就输出日志
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total
                    test_acc, f1, f1_0, f1_1 = self._evaluate()
                    # score = w_acc*0.2 + f1
                    # score = test_acc*0.15 + f1_0*0.35 + f1_1
                    score = test_acc + f1_1
                    weight_score = test_acc*0.5 + f1_1

                    if test_acc > max_test_acc:
                        max_test_acc = test_acc

                    # if w_acc > max_w_acc:
                    #     max_w_acc = w_acc

                    if f1 > max_f1:
                        max_f1 = f1
                    
                    if score > max_score:
                        max_score = score

                    if weight_score > max_weight_score:
                        max_weight_score = weight_score
                        if not os.path.exists('state_dict'):
                            os.mkdir('state_dict')
                        # path = './state_dict/{0}_{1}_score_{2:.4f}_f1_{3:.4f}'.format(self.opt.model_name, self.opt.dataset, score, f1)
                        path = './state_dict/{0}_{1}_f1_{2:.4f}_f1_0_{3:.4f}_f1_1_{4:.4f}_acc_{5:.4f}_score_{6:.4f}'.format(self.opt.model_name, self.opt.dataset, f1, f1_0, f1_1, test_acc, score)
                        logger.info('>> The {0} has been promoted on {1} with score {2:.4f}'.format(self.opt.model_name, self.opt.dataset, score))
                        self.best_model = copy.deepcopy(self.model)

                    logger.info('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}, f1_0: {:.4f}, f1_1: {:.4f}, weight_score: {:.4f}, score: {:.4f}'\
                                .format(loss.item(), train_acc, test_acc, f1, f1_0, f1_1, weight_score, score))

        return max_test_acc, max_w_acc, max_f1, max_score, path
    
    def _evaluate(self, show_results=False):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all, ids_all = None, None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_dataloader):
                # ids = t_sample_batched['dialogue_id'].to(self.opt.device)
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)
                
                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)
                
                # ids_all = torch.cat((ids_all, ids), dim = 0) if ids_all is not None else ids
                t_targets_all = torch.cat((t_targets_all, t_targets), dim=0) if t_targets_all is not None else t_targets
                t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0) if t_outputs_all is not None else t_outputs
        test_acc = n_test_correct / n_test_total

        # dialog_id = ids_all.data.cpu()
        labels = t_targets_all.data.cpu()
        predic = torch.argmax(t_outputs_all, -1).cpu()

        f1 = metrics.f1_score(labels, predic, labels=[0, 1], average='macro')
        f1_0 = metrics.f1_score(labels==0, predic==0, labels=True)
        # f1_1 = metrics.f1_score(labels==1, predic==1, labels=True)
        f1_1 = metrics.f1_score(labels, predic, average='binary')
        # w_acc = self.weighted_acc(dialog_id, labels, predic)
        
        if show_results:
            # report = metrics.classification_report(labels, predic, digits=4)
            report = metrics.classification_report(labels, predic, digits=4)
            # confusion = metrics.confusion_matrix(labels, predic)
            confusion = metrics.confusion_matrix(labels, predic)
            return report, confusion, f1, f1_0, f1_1

        return test_acc, f1, f1_0, f1_1

    def weighted_acc(self, ids, labels, predict_labels):
        # compute the weighted_accuracy
        # ids, labels, predict_labels = ids.tolist(), labels.tolist(), predict_labels.tolist()
        n_test_total = len(ids)
        acc_dict = defaultdict(defaultdict)
        total_acc = 0
        for idx, label, predict_label in list(zip(ids, labels, predict_labels)):
            if idx not in acc_dict:
                acc_dict[idx]['total'] = 1
                acc_dict[idx]['true'] = 0
            else:
                acc_dict[idx]['total'] += 1
            if label == predict_label:
                acc_dict[idx]['true'] += 1
        for i, idx in acc_dict.items():
            acc_dict[i]['acc'] = idx['true'] / idx['total']
            total_acc += acc_dict[i]['acc']
            # print(idx['true'] / idx['total'])
        total_acc /= len(acc_dict)
        return total_acc


    def _test(self, model_path):
        # test
        # self.model.load_state_dict(torch.load(model_path))
        self.model = self.best_model
        self.model.eval()
        test_report, test_confusion, f1, f1_0, f1_1 = self._evaluate(show_results=True)
        logger.info("Precision, Recall and F1-Score...")
        logger.info(test_report)
        logger.info("Confusion Matrix...")
        logger.info(test_confusion)
        logger.info('f1: {:.4f},'.format(f1))
        logger.info('f1_0: {:.4f},'.format(f1_0))
        logger.info('f1_1: {:.4f},'.format(f1_1))

    
    def run(self, repeats=1):
        max_test_acc_overall = 0
        max_w_acc_overall = 0
        max_f1_overall = 0
        max_score_overall = 0
        for i in range(repeats):
            logger.info('repeat:{}'.format(i))
            # torch.cuda.empty_cache()
            # self._reset_params()
            max_test_acc, max_w_acc, max_f1, max_score, model_path = self._train(max_test_acc_overall, max_w_acc_overall, max_f1_overall, max_score_overall)
            logger.info('max_test_acc: {0:.4f}, max_w_acc: {1:.4f}, max_f1: {2:.4f}, max_score: {3:.4f}'.format(max_test_acc, max_w_acc, max_f1, max_score))
            max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
            max_w_acc_overall = max(max_w_acc, max_w_acc_overall)
            max_f1_overall = max(max_f1, max_f1_overall)
            max_score_overall = max(max_score, max_score_overall)
            # * 模型存储
            if opt.notsavemodel:
                txt_path = model_path + '.txt'
                os.mknod(txt_path)
            else:
                torch.save(self.best_model.state_dict(), model_path)
            logger.info('>> saved: {}'.format(model_path))
            logger.info('#' * 100)
        logger.info('max_test_acc_overall:{:.4f}'.format(max_test_acc_overall))
        logger.info('max_w_acc_overall:{:.4f}'.format(max_w_acc_overall))
        logger.info('max_f1_overall:{:.4f}'.format(max_f1_overall))
        logger.info('max_score_overall:{:.4f}'.format(max_score_overall))
        self._test(model_path)


def main():
    
    dataset_files = {
        # * cn-data
        'cn_fold_0': {
            'train': './data/data_StratifiedKFold_666/cn/data_fold_0/train.csv',
            'test': './data/data_StratifiedKFold_666/cn/data_fold_0/test.csv'
        },
        'cn_fold_1': {
            'train': './data/data_StratifiedKFold_666/cn/data_fold_1/train.csv',
            'test': './data/data_StratifiedKFold_666/cn/data_fold_1/test.csv'
        },
        'cn_fold_2': {
            'train': './data/data_StratifiedKFold_666/cn/data_fold_2/train.csv',
            'test': './data/data_StratifiedKFold_666/cn/data_fold_2/test.csv'
        },
        'cn_fold_3': {
            'train': './data/data_StratifiedKFold_666/cn/data_fold_3/train.csv',
            'test': './data/data_StratifiedKFold_666/cn/data_fold_3/test.csv'
        },
        'cn_fold_4': {
            'train': './data/data_StratifiedKFold_666/cn/data_fold_4/train.csv',
            'test': './data/data_StratifiedKFold_666/cn/data_fold_4/test.csv'
        },
        # pseudo
        'cn_fold_0_pseudo': {
            'train': './data/data_StratifiedKFold_666_pseudo_0627/cn/data_fold_0/train.csv',
            'test': './data/data_StratifiedKFold_666_pseudo_0627/cn/data_fold_0/test.csv'
        },
        'cn_fold_1_pseudo': {
            'train': './data/data_StratifiedKFold_666_pseudo_0627/cn/data_fold_1/train.csv',
            'test': './data/data_StratifiedKFold_666_pseudo_0627/cn/data_fold_1/test.csv'
        },
        'cn_fold_2_pseudo': {
            'train': './data/data_StratifiedKFold_666_pseudo_0627/cn/data_fold_2/train.csv',
            'test': './data/data_StratifiedKFold_666_pseudo_0627/cn/data_fold_2/test.csv'
        },
        'cn_fold_3_pseudo': {
            'train': './data/data_StratifiedKFold_666_pseudo_0627/cn/data_fold_3/train.csv',
            'test': './data/data_StratifiedKFold_666_pseudo_0627/cn/data_fold_3/test.csv'
        },
        'cn_fold_4_pseudo': {
            'train': './data/data_StratifiedKFold_666_pseudo_0627/cn/data_fold_4/train.csv',
            'test': './data/data_StratifiedKFold_666_pseudo_0627/cn/data_fold_4/test.csv'
        },

        'cn_fold_0_uuu': {
            'train': './data/data_StratifiedKFold_666_uuu/cn/data_fold_0/train.csv',
            'test': './data/data_StratifiedKFold_666_uuu/cn/data_fold_0/test.csv'
        },
        'cn_fold_1_uuu': {
            'train': './data/data_StratifiedKFold_666_uuu/cn/data_fold_1/train.csv',
            'test': './data/data_StratifiedKFold_666_uuu/cn/data_fold_1/test.csv'
        },
        'cn_fold_2_uuu': {
            'train': './data/data_StratifiedKFold_666_uuu/cn/data_fold_2/train.csv',
            'test': './data/data_StratifiedKFold_666_uuu/cn/data_fold_2/test.csv'
        },
        'cn_fold_3_uuu': {
            'train': './data/data_StratifiedKFold_666_uuu/cn/data_fold_3/train.csv',
            'test': './data/data_StratifiedKFold_666_uuu/cn/data_fold_3/test.csv'
        },
        'cn_fold_4_uuu': {
            'train': './data/data_StratifiedKFold_666_uuu/cn/data_fold_4/train.csv',
            'test': './data/data_StratifiedKFold_666_uuu/cn/data_fold_4/test.csv'
        },
        # * cn-data_aug
        'cn_fold_0_aug': {
            'train': './data/data_StratifiedKFold_666_aug/cn/data_fold_0/train.csv',
            'test': './data/data_StratifiedKFold_666_aug/cn/data_fold_0/test.csv'
        },
        'cn_fold_1_aug': {
            'train': './data/data_StratifiedKFold_666_aug/cn/data_fold_1/train.csv',
            'test': './data/data_StratifiedKFold_666_aug/cn/data_fold_1/test.csv'
        },
        'cn_fold_2_aug': {
            'train': './data/data_StratifiedKFold_666_aug/cn/data_fold_2/train.csv',
            'test': './data/data_StratifiedKFold_666_aug/cn/data_fold_2/test.csv'
        },
        'cn_fold_3_aug': {
            'train': './data/data_StratifiedKFold_666_aug/cn/data_fold_3/train.csv',
            'test': './data/data_StratifiedKFold_666_aug/cn/data_fold_3/test.csv'
        },
        'cn_fold_4_aug': {
            'train': './data/data_StratifiedKFold_666_aug/cn/data_fold_4/train.csv',
            'test': './data/data_StratifiedKFold_666_aug/cn/data_fold_4/test.csv'
        },

        'cn_fold_0_uuu_aug': {
            'train': './data/data_StratifiedKFold_666_uuu_aug/cn/data_fold_0/train.csv',
            'test': './data/data_StratifiedKFold_666_uuu_aug/cn/data_fold_0/test.csv'
        },
        'cn_fold_1_uuu_aug': {
            'train': './data/data_StratifiedKFold_666_uuu_aug/cn/data_fold_1/train.csv',
            'test': './data/data_StratifiedKFold_666_uuu_aug/cn/data_fold_1/test.csv'
        },
        'cn_fold_2_uuu_aug': {
            'train': './data/data_StratifiedKFold_666_uuu_aug/cn/data_fold_2/train.csv',
            'test': './data/data_StratifiedKFold_666_uuu_aug/cn/data_fold_2/test.csv'
        },
        'cn_fold_3_uuu_aug': {
            'train': './data/data_StratifiedKFold_666_uuu_aug/cn/data_fold_3/train.csv',
            'test': './data/data_StratifiedKFold_666_uuu_aug/cn/data_fold_3/test.csv'
        },
        'cn_fold_4_uuu_aug': {
            'train': './data/data_StratifiedKFold_666_uuu_aug/cn/data_fold_4/train.csv',
            'test': './data/data_StratifiedKFold_666_uuu_aug/cn/data_fold_4/test.csv'
        },
        # * cn-data_dia
        'cn_fold_0_dia': {
            'train': './data/data_StratifiedKFold_666_dia/cn/data_fold_0/train.csv',
            'test': './data/data_StratifiedKFold_666_dia/cn/data_fold_0/test.csv'
        },
        'cn_fold_1_dia': {
            'train': './data/data_StratifiedKFold_666_dia/cn/data_fold_1/train.csv',
            'test': './data/data_StratifiedKFold_666_dia/cn/data_fold_1/test.csv'
        },
        'cn_fold_2_dia': {
            'train': './data/data_StratifiedKFold_666_dia/cn/data_fold_2/train.csv',
            'test': './data/data_StratifiedKFold_666_dia/cn/data_fold_2/test.csv'
        },
        'cn_fold_3_dia': {
            'train': './data/data_StratifiedKFold_666_dia/cn/data_fold_3/train.csv',
            'test': './data/data_StratifiedKFold_666_dia/cn/data_fold_3/test.csv'
        },
        'cn_fold_4_dia': {
            'train': './data/data_StratifiedKFold_666_dia/cn/data_fold_4/train.csv',
            'test': './data/data_StratifiedKFold_666_dia/cn/data_fold_4/test.csv'
        },
        # * cn-data_filtered
        'cn_fold_0_fil': {
            'train': './data/data_StratifiedKFold_666_filtered/cn/data_fold_0/train.csv',
            'test': './data/data_StratifiedKFold_666_filtered/cn/data_fold_0/test.csv'
        },
        'cn_fold_1_fil': {
            'train': './data/data_StratifiedKFold_666_filtered/cn/data_fold_1/train.csv',
            'test': './data/data_StratifiedKFold_666_filtered/cn/data_fold_1/test.csv'
        },
        'cn_fold_2_fil': {
            'train': './data/data_StratifiedKFold_666_filtered/cn/data_fold_2/train.csv',
            'test': './data/data_StratifiedKFold_666_filtered/cn/data_fold_2/test.csv'
        },
        'cn_fold_3_fil': {
            'train': './data/data_StratifiedKFold_666_filtered/cn/data_fold_3/train.csv',
            'test': './data/data_StratifiedKFold_666_filtered/cn/data_fold_3/test.csv'
        },
        'cn_fold_4_fil': {
            'train': './data/data_StratifiedKFold_666_filtered/cn/data_fold_4/train.csv',
            'test': './data/data_StratifiedKFold_666_filtered/cn/data_fold_4/test.csv'
        },
    }
    
    opt.dataset_file = dataset_files[opt.dataset]
    
    # set random seed
    setup_seed(opt.seed)

    if not os.path.exists('log'):
        os.mkdir('log')
    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%Y-%m-%d_%H:%M:%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % ('./log', log_file)))

    # torch.set_printoptions(precision=None, threshold=float("inf"), edgeitems=None, linewidth=None, profile=None)
    start_time = time.time()
    ins = Instructor(opt)
    ins.run(opt.repeats)
    time_dif = get_time_dif(start_time)
    logger.info("Time usage: {}".format(time_dif))

if __name__ == '__main__':
    main()