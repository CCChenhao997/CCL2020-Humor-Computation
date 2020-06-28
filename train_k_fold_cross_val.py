import os
import torch
import torch.nn as nn
import argparse
import random
import math
import logging
import sys
import time
from time import strftime, localtime
from sklearn import metrics
import numpy as np
from transformers import BertModel, AutoModel, XLMRobertaModel, GPT2Model, RobertaModel
from transformers import AdamW
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.nn.utils import clip_grad_norm_
from models_utils.loss_helper import FocalLoss
from models_utils.adv_helper import FGM, PGD
from models.bert import BERT
from models.bert_spc import BERT_SPC
from models.bert_att import BERT_Att
from models.bert_spc_att import BERT_SPC_Att
from models.bert_spc_pos import BERT_SPC_Pos
from data_utils import Tokenizer4Bert, BertSentenceDataset, get_time_dif
from sklearn.model_selection import StratifiedKFold, KFold

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


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
        bert_model = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.pretrained_bert_state_dict = bert_model.state_dict()
        self.model = opt.model_class(bert_model, opt).to(opt.device)
        trainset = BertSentenceDataset(opt.dataset_file['train'], tokenizer, target_dim=self.opt.polarities_dim, opt=opt)
        testset = BertSentenceDataset(opt.dataset_file['test'], tokenizer, target_dim=self.opt.polarities_dim, opt=opt)
        self.train_dataloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)   # , drop_last=True
        self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size, shuffle=False)

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
        # for child in self.model.children():
        #     if type(child) != BertModel:  # skip bert params
        #         for p in child.parameters():
        #             if p.requires_grad:
        #                 if len(p.shape) > 1:
        #                     self.opt.initializer(p)
        #                 else:
        #                     stdv = 1. / math.sqrt(p.shape[0])
        #                     torch.nn.init.uniform_(p, a=-stdv, b=stdv)
        #     else:
        #         self.model.bert.load_state_dict(self.pretrained_bert_state_dict)
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
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': opt.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                        lr=opt.learning_rate, eps=opt.adam_epsilon, weight_decay=self.opt.l2reg)
        # scheduler = WarmupLinearSchedule(
        #     optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        return optimizer
    
    def _train(self, max_test_acc_overall=0, max_f1_overall=0):
        # 对抗训练
        if self.opt.adv_type == 'fgm':
            fgm = FGM(self.model)

        # criterion = nn.CrossEntropyLoss()
        criterion = FocalLoss(num_class=2, alpha=0.25, gamma=2, smooth=0.2)
        if 'bert' in self.opt.model_name:
            optimizer = self.get_bert_optimizer(self.opt, self.model)
        else:
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
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
                loss.backward()

                if self.opt.adv_type == 'fgm':
                    fgm.attack()  ##对抗训练
                    outputs = self.model(inputs)
                    loss_adv = criterion(outputs, targets)
                    loss_adv.backward()
                    fgm.restore()

                optimizer.step()
                
                if global_step % self.opt.log_step == 0:    # 每隔opt.log_step就输出日志
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total
                    test_acc, f1 = self._evaluate()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                    if f1 > max_f1:
                        max_f1 = f1
                        if f1 > max_f1_overall:
                            if not os.path.exists('state_dict'):
                                os.mkdir('state_dict')
                            path = './state_dict/{0}_{1}_f1_{2:.4f}'.format(self.opt.model_name, self.opt.dataset, f1)
                            torch.save(self.model.state_dict(), path)
                            logger.info('>> saved: {}'.format(path))

                    logger.info('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc, test_acc, f1))
        return max_test_acc, max_f1, path
    
    def _evaluate(self, show_results=False):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_dataloader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)
                
                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)
                
                t_targets_all = torch.cat((t_targets_all, t_targets), dim=0) if t_targets_all is not None else t_targets
                t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0) if t_outputs_all is not None else t_outputs
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1], average='macro')

        labels = t_targets_all.data.cpu().numpy()
        predic = torch.argmax(t_outputs_all, -1).cpu().numpy()
        if show_results:
            report = metrics.classification_report(labels, predic, digits=4)
            confusion = metrics.confusion_matrix(labels, predic)
            return report, confusion

        return test_acc, f1

    def _test(self, model_path):
        # test
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        # start_time = time.time()
        test_report, test_confusion = self._evaluate(show_results=True)
        logger.info("Precision, Recall and F1-Score...")
        logger.info(test_report)
        logger.info("Confusion Matrix...")
        logger.info(test_confusion)
        # time_dif = get_time_dif(start_time)
        # logger.info("Time usage:", time_dif)
    
    def run(self, repeats=1):
        max_test_acc_overall = 0
        max_f1_overall = 0
        for i in range(repeats):
            logger.info('repeat:{}'.format(i))
            # self._reset_params()
            max_test_acc, max_f1, model_path = self._train(max_test_acc_overall, max_f1_overall)
            logger.info('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
            max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
            max_f1_overall = max(max_f1, max_f1_overall)
            logger.info('#' * 60)
        logger.info('max_test_acc_overall:{}'.format(max_test_acc_overall))
        logger.info('max_f1_overall:{}'.format(max_f1_overall))
        self._test(model_path)


def main():
    
    model_classes = {
        'bert': BERT,
        'bert_att': BERT_Att,
        'bert_spc': BERT_SPC,
        'bert_spc_att': BERT_SPC_Att,
        'bert_spc_pos': BERT_SPC_Pos,
    }
    
    dataset_files = {
        # * cn-data
        'cn_fold_0': {
            'train': './data_StratifiedKFold_666/cn/data_fold_0/train.csv',
            'test': './data_StratifiedKFold_666/cn/data_fold_0/test.csv'
        },
        'cn_fold_1': {
            'train': './data_StratifiedKFold_666/cn/data_fold_1/train.csv',
            'test': './data_StratifiedKFold_666/cn/data_fold_1/test.csv'
        },
        'cn_fold_2': {
            'train': './data_StratifiedKFold_666/cn/data_fold_2/train.csv',
            'test': './data_StratifiedKFold_666/cn/data_fold_2/test.csv'
        },
        'cn_fold_3': {
            'train': './data_StratifiedKFold_666/cn/data_fold_3/train.csv',
            'test': './data_StratifiedKFold_666/cn/data_fold_3/test.csv'
        },
        'cn_fold_4': {
            'train': './data_StratifiedKFold_666/cn/data_fold_4/train.csv',
            'test': './data_StratifiedKFold_666/cn/data_fold_4/test.csv'
        },
        # * en-data
        'en_fold_0': {
            'train': './data_StratifiedKFold_666/en/data_fold_0/train.csv',
            'test': './data_StratifiedKFold_666/en/data_fold_0/test.csv'
        },
        'en_fold_1': {
            'train': './data_StratifiedKFold_666/en/data_fold_1/train.csv',
            'test': './data_StratifiedKFold_666/en/data_fold_1/test.csv'
        },
        'en_fold_2': {
            'train': './data_StratifiedKFold_666/en/data_fold_2/train.csv',
            'test': './data_StratifiedKFold_666/en/data_fold_2/test.csv'
        },
        'en_fold_3': {
            'train': './data_StratifiedKFold_666/en/data_fold_3/train.csv',
            'test': './data_StratifiedKFold_666/en/data_fold_3/test.csv'
        },
        'en_fold_4': {
            'train': './data_StratifiedKFold_666/en/data_fold_4/train.csv',
            'test': './data_StratifiedKFold_666/en/data_fold_4/test.csv'
        },

        # * cn-transdata
        'cn_trans_fold_0': {
            'train': './transdata_StratifiedKFold_666/cn/data_fold_0/train.csv',
            'test': './transdata_StratifiedKFold_666/cn/data_fold_0/test.csv'
        },
        'cn_trans_fold_1': {
            'train': './transdata_StratifiedKFold_666/cn/data_fold_1/train.csv',
            'test': './transdata_StratifiedKFold_666/cn/data_fold_1/test.csv'
        },
        'cn_trans_fold_2': {
            'train': './transdata_StratifiedKFold_666/cn/data_fold_2/train.csv',
            'test': './transdata_StratifiedKFold_666/cn/data_fold_2/test.csv'
        },
        'cn_trans_fold_3': {
            'train': './transdata_StratifiedKFold_666/cn/data_fold_3/train.csv',
            'test': './transdata_StratifiedKFold_666/cn/data_fold_3/test.csv'
        },
        'cn_trans_fold_4': {
            'train': './transdata_StratifiedKFold_666/cn/data_fold_4/train.csv',
            'test': './transdata_StratifiedKFold_666/cn/data_fold_4/test.csv'
        },
        # * en-transdata
        'en_trans_fold_0': {
            'train': './transdata_StratifiedKFold_666/en/data_fold_0/train.csv',
            'test': './transdata_StratifiedKFold_666/en/data_fold_0/test.csv'
        },
        'en_trans_fold_1': {
            'train': './transdata_StratifiedKFold_666/en/data_fold_1/train.csv',
            'test': './transdata_StratifiedKFold_666/en/data_fold_1/test.csv'
        },
        'en_trans_fold_2': {
            'train': './transdata_StratifiedKFold_666/en/data_fold_2/train.csv',
            'test': './transdata_StratifiedKFold_666/en/data_fold_2/test.csv'
        },
        'en_trans_fold_3': {
            'train': './transdata_StratifiedKFold_666/en/data_fold_3/train.csv',
            'test': './transdata_StratifiedKFold_666/en/data_fold_3/test.csv'
        },
        'en_trans_fold_4': {
            'train': './transdata_StratifiedKFold_666/en/data_fold_4/train.csv',
            'test': './transdata_StratifiedKFold_666/en/data_fold_4/test.csv'
        }
    }
    
    input_colses = {
        'bert': ['sentence_bert_indices', 'attention_mask'],
        'bert_att': ['sentence_bert_indices', 'attention_mask'],
        'bert_spc': ['sentence_pair_bert_indices', 'bert_segments_ids', 'attention_mask_pair'],
        'bert_spc_att': ['sentence_pair_bert_indices', 'bert_segments_ids', 'attention_mask_pair'],
        'bert_spc_pos': ['sentence_pair_bert_indices', 'bert_segments_ids', 'attention_mask_pair'],
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
    parser.add_argument('--dataset', default='cn', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
    parser.add_argument('--learning_rate', default=0.002, type=float)    # 1e-3
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--l2reg', default=1e-5, type=float)    # 1e-5
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    # parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--position_dim', default=100, type=int)
    parser.add_argument('--polarities_dim', default=2, type=int, help='2')
    parser.add_argument('--max_length', default=80, type=int)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--repeats', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.") # 0.01
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--cross_val_fold', default=5, type=int, help='k-fold cross validation')
    # parser.add_argument('--grad_clip', type=float, default=10, help='clip gradients at this value')
    parser.add_argument('--cuda', default=0, type=str)
    parser.add_argument('--transdara', default=False, type=bool)
    parser.add_argument('--attention_hops', default=5, type=int)
    parser.add_argument('--adv_type', default=None, type=str, help='fgm, pgd')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)
    
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