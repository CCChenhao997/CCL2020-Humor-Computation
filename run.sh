#!/bin/bash

# bert_att
# python train_k_fold_cross_val.py --model bert_att --dataset en --seed 1000 --learning_rate 2e-5 --num_epoch 3  # bert_base
python train_k_fold_cross_val.py --model bert_att --dataset cn --pretrained_bert_name ./pretrain_models/ERNIE_cn  --seed 1000 --learning_rate 2e-5 --num_epoch 3  # ERNIE