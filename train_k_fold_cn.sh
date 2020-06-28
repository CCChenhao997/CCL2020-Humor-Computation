#!/bin/bash

# bert_spc
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_0 --pretrained_bert_name ./pretrain_models/ERNIE_cn --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.5 --cuda 3 # bert_base
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_1 --pretrained_bert_name ./pretrain_models/ERNIE_cn --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.5 --cuda 3
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_2 --pretrained_bert_name ./pretrain_models/ERNIE_cn --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.5 --cuda 3
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_3 --pretrained_bert_name ./pretrain_models/ERNIE_cn --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.5 --cuda 3
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_4 --pretrained_bert_name ./pretrain_models/ERNIE_cn --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.5 --cuda 3

# 0.65+
# python train_k_fold_cross_val.py --model bert --dataset cn_fold_0 --pretrained_bert_name bert-base-chinese --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.5 --cuda 3

# roberta
# 0.7322/0.6529
# python train_k_fold_cross_val.py --model bert --dataset cn_fold_0 --pretrained_bert_name ./pretrain_models/roberta_wwm_ext_cn --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.3 --cuda 2


# * transdata
# 0.7308/0.6430
# python train_k_fold_cross_val.py --model bert_spc --transdara True --dataset cn_trans_fold_0 --max_length 128  --pretrained_bert_name ./pretrain_models/ERNIE_cn --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --cuda 1

# 0.7355/0.6489
# python train_k_fold_cross_val.py --model bert_spc_pos --transdara True --dataset cn_trans_fold_0 --max_length 128  --pretrained_bert_name ./pretrain_models/ERNIE_cn --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2

# 0.7363/0.6468
# python train_k_fold_cross_val.py --model bert_spc_att --transdara True --dataset cn_trans_fold_0 --max_length 128  --pretrained_bert_name ./pretrain_models/ERNIE_cn --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --cuda 3