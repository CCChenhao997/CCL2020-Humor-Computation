#!/bin/bash

# python train.py --model bert_spc --dataset en --seed 1000 --learning_rate 2e-5 --num_epoch 3 --max_length 80 --pretrained_bert_name bert-large-uncased --bert_dim 1024
# python train.py --model bert_spc --dataset en --seed 1000 --learning_rate 2e-5 --num_epoch 3 --max_length 80

python train.py --model bert_spc --dataset cn --seed 1000 --learning_rate 2e-5 --num_epoch 3 --max_length 80 --batch_size 32