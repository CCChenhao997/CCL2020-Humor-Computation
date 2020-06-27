#!/bin/bash

# bert_spc
python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_0 --seed 1000 --learning_rate 2e-5 --num_epoch 3  # bert_base
python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_1 --seed 1000 --learning_rate 2e-5 --num_epoch 3
python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_2 --seed 1000 --learning_rate 2e-5 --num_epoch 3
python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_3 --seed 1000 --learning_rate 2e-5 --num_epoch 3
python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_4 --seed 1000 --learning_rate 2e-5 --num_epoch 3

# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_0 --pretrained_bert_name ./pretrain_models/ERNIE_cn --seed 1000 --learning_rate 2e-5 --num_epoch 3  # bert_base
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_1 --pretrained_bert_name ./pretrain_models/ERNIE_cn --seed 1000 --learning_rate 2e-5 --num_epoch 3
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_2 --pretrained_bert_name ./pretrain_models/ERNIE_cn --seed 1000 --learning_rate 2e-5 --num_epoch 3
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_3 --pretrained_bert_name ./pretrain_models/ERNIE_cn --seed 1000 --learning_rate 2e-5 --num_epoch 3
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_4 --pretrained_bert_name ./pretrain_models/ERNIE_cn --seed 1000 --learning_rate 2e-5 --num_epoch 3