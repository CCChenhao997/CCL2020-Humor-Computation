#!/bin/bash

# * data

# bert_spc
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_0 --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.1 --cuda 2 # bert_base
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_1 --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.1 --cuda 2
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_2 --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.1 --cuda 2
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_3 --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.1 --cuda 2
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_4 --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.1 --cuda 2

# 0.7705/0.6439
# python train_k_fold_cross_val.py --model bert_att --dataset en_fold_0 --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.1 --cuda 3 --pretrained_bert_name bert-large-uncased --bert_dim 1024

# # -/0.66+
# python train_k_fold_cross_val.py --model bert_att --dataset en_fold_0 --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.1 --cuda 3

# * transdata

# 0.7859/0.6743
# python train_k_fold_cross_val.py --model bert_spc --transdara True --dataset en_trans_fold_0 --max_length 128 --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.1 --cuda 2

# 0.7886/0.6800
# python train_k_fold_cross_val.py --model bert_spc_pos --transdara True --dataset en_trans_fold_0 --max_length 128 --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.1 --cuda 3

# 0.7906/0.6772
python train_k_fold_cross_val.py --model bert_spc_att --transdara True --dataset en_trans_fold_0 --max_length 128 --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.1 --cuda 3