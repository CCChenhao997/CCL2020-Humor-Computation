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

# max_test_acc_overall:0.7337
# max_w_acc_overall:0.7317
# max_f1_overall:0.6507
# max_score_overall:1.3649
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_4 --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn

# max_test_acc_overall:0.7381
# max_w_acc_overall:0.7304
# max_f1_overall:0.6441
# max_score_overall:1.3515
# python train_k_fold_cross_val.py --model bert_att --dataset cn_fold_4 --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn

# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_0 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_1 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_2 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_3 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_4 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn

# 7.8 
# bert_spc_cap 实验效果不佳
# bert_spc + adv + batch_size = 1效果不佳
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_0 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --batch_size 1 --log_step 100 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_1 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --batch_size 1 --log_step 100 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_2 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --batch_size 1 --log_step 100 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_3 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --batch_size 1 --log_step 100 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_4 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --batch_size 1 --log_step 100 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn
# bert_spc + batch_size = 1 rubbish!
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_0 --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --batch_size 1 --log_step 100 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_1 --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --batch_size 1 --log_step 100 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_2 --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --batch_size 1 --log_step 100 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_3 --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --batch_size 1 --log_step 100 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_4 --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --batch_size 1 --log_step 100 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn

# 7.10
# under_data 效果不佳，1的数值下降严重
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_0 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_1 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_2 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_3 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn
# python train_k_fold_cross_val.py --model bert_spc --dataset cn_fold_4 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn

# 7.11 （调整模型选取后）
# drop_out 对比试验
# adv_uuu drop_out 0.5
# python train_cn.py --model bert_spc_lay --dataset cn_fold_0 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss
# python train_cn.py --model bert_spc_lay --dataset cn_fold_1 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss
# python train_cn.py --model bert_spc_lay --dataset cn_fold_2 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss
# python train_cn.py --model bert_spc_lay --dataset cn_fold_3 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss
# python train_cn.py --model bert_spc_lay --dataset cn_fold_4 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss
# uuu drop_out 0.3 似乎drop_out越小越好(其实差别不大？)
# 先选定0.3跑以下实验

# bert_spc + uuu + 0.3 no_adv

# 7.12 pseudo
# python train_cn.py --model bert_spc_lay --dataset cn_fold_0 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss
# python train_cn.py --model bert_spc_lay --dataset cn_fold_1 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss
# python train_cn.py --model bert_spc_lay --dataset cn_fold_2 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss
# python train_cn.py --model bert_spc_lay --dataset cn_fold_3 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss
# python train_cn.py --model bert_spc_lay --dataset cn_fold_4 --adv_type fgm --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss

# # python train_cn.py --model bert_spc_lay --dataset cn_fold_0_uuu --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss
# python train_cn.py --model bert_spc_lay --dataset cn_fold_1_uuu --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss
# python train_cn.py --model bert_spc_lay --dataset cn_fold_2_uuu --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss
# python train_cn.py --model bert_spc_lay --dataset cn_fold_3_uuu --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss
# python train_cn.py --model bert_spc_lay --dataset cn_fold_4_uuu --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss

# 7.13
# bert_spc_lay + adv + dropout 0 + rawdata(sensentence_pair_bert_indices_reversetence_pair_bert_indices)
# bert_spc_lay  + dropout 0 + rawdata(sentence_pair_bert_indices_reverse)

# python train_cn.py --model bert_spc_lay --dataset cn_fold_0  --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss
# python train_cn.py --model bert_spc_lay --dataset cn_fold_1  --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss
# python train_cn.py --model bert_spc_lay --dataset cn_fold_2  --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss
# python train_cn.py --model bert_spc_lay --dataset cn_fold_3  --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss
# python train_cn.py --model bert_spc_lay --dataset cn_fold_4  --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0 --cuda 3 --max_length 128 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss

# bert_spc_lay  + dropout 0.3 + rawdata(sentence_pair_bert_indices)
# bert_spc_lay  + dropout 0.3 + rawdata(sentence_pair_bert_indices) + super
# bert_spc_lay  + dropout 0.3 + rawdata(sentence_pair_bert_indices) + decay0.01
# bert_spc +dropout 0.3 + rawdata(sentence_pair_bert_indices) decay0.01
# bert_spc +dropout 0.3 + uuudata(sentence_pair_bert_indices) decay0.01 + lr 5e-5 : Too BAD!
# bert_spc +dropout 0.3 + uuudata(sentence_pair_bert_indices) decay0.01 + lr 1e-5 : a little BAD!
# decay 0.01 效果不理想

#0718
# ? 对比uuu和raw

# nn.linear > Linear 80>120 3e-5 > 2e-5 > 5e-5
# python train_cn.py --model bert_spc --dataset cn_fold_0 --seed 1000 --learning_rate 3e-5 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss  #--weight_decay 0.01
# python train_cn.py --model bert_spc --dataset cn_fold_1 --seed 1000 --learning_rate 3e-5 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss #--weight_decay 0.01
# python train_cn.py --model bert_spc --dataset cn_fold_2 --seed 1000 --learning_rate 3e-5 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss #--weight_decay 0.01
# python train_cn.py --model bert_spc --dataset cn_fold_3 --seed 1000 --learning_rate 3e-5 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss #--weight_decay 0.01
# python train_cn.py --model bert_spc --dataset cn_fold_4 --seed 1000 --learning_rate 3e-5 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss #--weight_decay 0.01


# * 0719 pseudo(reserved)
# python train_cn.py --model bert_spc_lay --dataset cn_fold_0 --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.5 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss  #--weight_decay 0.01
# python train_cn.py --model bert_spc_cnn --dataset cn_fold_1 --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss #--weight_decay 0.01
# python train_cn.py --model bert_spc_lay --dataset cn_fold_2 --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --weight_decay 0.01
# python train_cn.py --model bert_spc --dataset cn_fold_3 --seed 1000 --learning_rate 5e-5 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --weight_decay 0.01
# python train_cn.py --model bert_spc_lay --dataset cn_fold_4 --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.5 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss #--weight_decay 0.01

# 0719 lcf
# python train_cn.py --model lcf_bert --dataset cn_fold_0 --seed 1000 --learning_rate 3e-5 --num_epoch 4 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss  #--weight_decay 0.01
# python train_cn.py --model lcf_bert --dataset cn_fold_1 --seed 1000 --learning_rate 3e-5 --num_epoch 4 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss #--weight_decay 0.01
# python train_cn.py --model lcf_bert --dataset cn_fold_2 --seed 1000 --learning_rate 3e-5 --num_epoch 4 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss #--weight_decay 0.01
# python train_cn.py --model lcf_bert --dataset cn_fold_3 --seed 1000 --learning_rate 3e-5 --num_epoch 4 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss #--weight_decay 0.01
# python train_cn.py --model lcf_bert --dataset cn_fold_4 --seed 1000 --learning_rate 3e-5 --num_epoch 4 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss #--weight_decay 0.01

# # lcf + adv
# python train_cn.py --model lcf_bert --dataset cn_fold_0 --seed 1000 --learning_rate 3e-5 --num_epoch 4 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --adv_type fgm #--weight_decay 0.01
# python train_cn.py --model lcf_bert --dataset cn_fold_1 --seed 1000 --learning_rate 3e-5 --num_epoch 4 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --adv_type fgm #--weight_decay 0.01
# python train_cn.py --model lcf_bert --dataset cn_fold_2 --seed 1000 --learning_rate 3e-5 --num_epoch 4 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --adv_type fgm #--weight_decay 0.01
# python train_cn.py --model lcf_bert --dataset cn_fold_3 --seed 1000 --learning_rate 3e-5 --num_epoch 4 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --adv_type fgm # --weight_decay 0.01
# python train_cn.py --model lcf_bert --dataset cn_fold_4 --seed 1000 --learning_rate 3e-5 --num_epoch 4 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --adv_type fgm #--weight_decay 0.01

# aug_1st_try
# python train_cn.py --model bert_spc --dataset cn_fold_0_uuu_aug --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --log_step 50 --weight_decay 0.01 #--batch_size 8  --log_step 5 #--adv_type fgm #--weight_decay 0.01
# python train_cn.py --model bert_spc --dataset cn_fold_1_uuu_aug --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --log_step 5 --weight_decay 0.01 #--adv_type fgm #--weight_decay 0.01
# python train_cn.py --model bert_spc --dataset cn_fold_2_uuu_aug --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --log_step 5 --weight_decay 0.01 #--adv_type fgm #--weight_decay 0.01
# python train_cn.py --model bert_spc --dataset cn_fold_3_uuu_aug --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --log_step 5 --weight_decay 0.01 #--adv_type fgm #--weight_decay 0.01
# python train_cn.py --model bert_spc --dataset cn_fold_4_uuu_aug --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --log_step 5 --weight_decay 0.01 #--adv_type fgm #--weight_decay 0.01


# chinese dia try
# python train_cn.py --model bert_bag_cnn --dataset cn_fold_0_dia --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --batch_size 1 --log_step 50 --datatype diadata --rnntype LSTM --dia_maxlength 32
# python train_cn.py --model bert_bag_cnn --dataset cn_fold_1_dia --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --batch_size 1 --log_step 50 --datatype diadata --rnntype LSTM --dia_maxlength 32 
# python train_cn.py --model bert_bag_cnn --dataset cn_fold_2_dia --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --batch_size 1 --log_step 50 --datatype diadata --rnntype LSTM --dia_maxlength 32
# python train_cn.py --model bert_bag_cnn --dataset cn_fold_3_dia --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --batch_size 1 --log_step 50 --datatype diadata --rnntype LSTM --dia_maxlength 32
# python train_cn.py --model bert_bag_cnn --dataset cn_fold_4_dia --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --batch_size 1 --log_step 50 --datatype diadata --rnntype LSTM --dia_maxlength 32

# python train_cn.py --model bert_spc_lay --dataset cn_fold_0_uuu_aug --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --batch_size 8  # --datatype raw --log_step 5 #--adv_type fgm #--weight_decay 0.01
# python train_cn.py --model bert_spc_lay --dataset cn_fold_1_uuu_aug --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --batch_size 8 # --datatype raw # --log_step 50 #--adv_type fgm #--weight_decay 0.01
# python train_cn.py --model bert_spc_lay --dataset cn_fold_2_uuu_aug --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --batch_size 8 # --datatype raw # --log_step 50 #--adv_type fgm #--weight_decay 0.01
# python train_cn.py --model bert_spc_lay --dataset cn_fold_3_uuu_aug --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --batch_size 8 # --datatype raw # --log_step 50 #--adv_type fgm #--weight_decay 0.01
# python train_cn.py --model bert_spc_lay --dataset cn_fold_4_uuu_aug --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --batch_size 8 # --datatype raw  # --log_step 50 #--adv_type fgm #--weight_decay 0.01

# * 0727 batchNorm bad
# cn_fold_0_aug
# python train_cn.py --model bert_spc --dataset cn_fold_0 --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.5 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --log_step 50 --weight_decay 0.01 # --adv_type fgm #--batch_size 8  --log_step 5 # #--weight_decay 0.01
# # python train_cn.py --model bert_spc --dataset cn_fold_1 --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --log_step 50 --weight_decay 0.01 # --adv_type fgm #--weight_decay 0.01
# python train_cn.py --model bert_spc --dataset cn_fold_2 --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.5 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --log_step 50 --weight_decay 0.01 #--adv_type fgm #--weight_decay 0.01
# # python train_cn.py --model bert_spc --dataset cn_fold_3 --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --log_step 5 --weight_decay 0.01 --adv_type fgm #--weight_decay 0.01
# python train_cn.py --model bert_spc --dataset cn_fold_4 --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.5 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --log_step 50 --weight_decay 0.01 #--adv_type fgm #--weight_decay 0.01

# * 0728 filtered
# python train_cn.py --model bert_spc --dataset cn_fold_0_fil --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --log_step 5 --weight_decay 0.01 # --adv_type fgm #--batch_size 8  --log_step 5 # #--weight_decay 0.01
# python train_cn.py --model bert_spc --dataset cn_fold_1_fil --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --log_step 5 --weight_decay 0.01 # --adv_type fgm #--weight_decay 0.01
# python train_cn.py --model bert_spc --dataset cn_fold_2_fil --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --log_step 5 --weight_decay 0.01 #--adv_type fgm #--weight_decay 0.01
# python train_cn.py --model bert_spc --dataset cn_fold_3_fil --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --log_step 5 --weight_decay 0.01 #--adv_type fgm #--weight_decay 0.01
# python train_cn.py --model bert_spc --dataset cn_fold_4_fil --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --log_step 5 --weight_decay 0.01 #--adv_type fgm #--weight_decay 0.01

# * 0730 Ernie2
# python train_cn.py --model_name bert_spc --dataset cn_fold_0_aug --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 5 --weight_decay 0.01 #--adv_type fgm #--batch_size 8  --log_step 5 # #--weight_decay 0.01
# python train_cn.py --model_name bert_spc --dataset cn_fold_1_aug --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 5 --weight_decay 0.01 #--adv_type fgm #--weight_decay 0.01
# python train_cn.py --model_name bert_spc --dataset cn_fold_2_aug --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 5 --weight_decay 0.01 #--adv_type fgm #--weight_decay 0.01
# python train_cn.py --model_name bert_spc --dataset cn_fold_3_aug --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 5 --weight_decay 0.01 #--adv_type fgm #--weight_decay 0.01
# python train_cn.py --model_name bert_spc --dataset cn_fold_4_aug --seed 1000 --learning_rate 3e-5 --num_epoch 5 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 5 --weight_decay 0.01 #--adv_type fgm #--weight_decay 0.01

# python train_cn.py --model_name bert_spc --dataset cn_fold_1_pseudo --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.5 --cuda 2 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --log_step 25 --criterion focalloss 

# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_1_pseudo --seed 1000 --learning_rate 3e-5 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 5 --criterion crossentropy --weight_decay 0.01 --scheduler --notsavemodel

# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_0_aug --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 90 --pretrained_bert_name ./pretrain_models/ERNIE_0803_T_reverse --log_step 5 --criterion focalloss --weight_decay 0.01 --scheduler --adv_type fgm  --diff_lr
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_1_aug --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 90 --pretrained_bert_name ./pretrain_models/ERNIE_0803_T_reverse --log_step 5 --criterion focalloss --weight_decay 0.01 --scheduler --adv_type fgm  --diff_lr
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_2_aug --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 90 --pretrained_bert_name ./pretrain_models/ERNIE_0803_T_reverse --log_step 5 --criterion focalloss --weight_decay 0.01 --scheduler --adv_type fgm  --diff_lr # --weight_decay 0.01 
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_3_aug --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 90 --pretrained_bert_name ./pretrain_models/ERNIE_0803_T_reverse --log_step 5 --criterion focalloss --weight_decay 0.01 --scheduler --adv_type fgm  --diff_lr # --weight_decay 0.01
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_4_aug --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 90 --pretrained_bert_name ./pretrain_models/ERNIE_0803_T_reverse --log_step 5 --criterion focalloss --weight_decay 0.01 --scheduler --adv_type fgm  --diff_lr # --weight_decay 0.01 

# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_1_aug --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 90 --pretrained_bert_name ./pretrain_models/ERNIE_0803_T_reverse --log_step 5 --criterion focalloss --weight_decay 0.01 --scheduler --adv_type fgm  --diff_lr
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_2_aug_pseudo --seed 1000 --bert_lr 2e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.5 --cuda 3 --max_length 90 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 25 --criterion focalloss --weight_decay 0.01 --scheduler --diff_lr
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_3_aug_pseudo --seed 1000 --bert_lr 2e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.5 --cuda 3 --max_length 90 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 25 --criterion focalloss --weight_decay 0.01 --scheduler --diff_lr # --weight_decay 0.01
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_2_aug_pseudo --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.1 --cuda 3 --max_length 90 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 25 --criterion focalloss --weight_decay 0.01 --scheduler --diff_lr
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_3_aug_pseudo --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.1 --cuda 3 --max_length 90 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 25 --criterion focalloss --weight_decay 0.01 --scheduler --diff_lr
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_4_aug_pseudo --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 100 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 25 --criterion focalloss --weight_decay 0.01 --scheduler --diff_lr # --weight_decay 0.01

# python train_cn.py --model_name bert_spc --dataset cn_fold_0_aug --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.3 --cuda 1 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 25 --criterion crossentropy --weight_decay 0.01 --scheduler --diff_lr
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_1_aug_pseudo_enhanced_0 --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 90 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 25 --criterion focalloss --weight_decay 0.01 --scheduler --diff_lr
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_2_aug_pseudo --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.1 --cuda 3 --max_length 120 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 25 --criterion focalloss --weight_decay 0.01 --scheduler --diff_lr
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_3_aug_pseudo --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.1 --cuda 3 --max_length 120 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 25 --criterion focalloss --weight_decay 0.01 --scheduler --diff_lr # --weight_decay 0.01 
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_4_aug_pseudo --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.1 --cuda 3 --max_length 120 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 25 --criterion focalloss --weight_decay 0.01 --scheduler --diff_lr # --weight_decay 0.01
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_4_aug --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 90 --pretrained_bert_name ./pretrain_models/ERNIE_0803_T_reverse --log_step 5 --criterion crossentropy --weight_decay 0.01 --scheduler --diff_lr

# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_0_pseudo --seed 1000 --learning_rate 9e-5 --train_batch_size 32 --num_epoch 4 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_0803_T_reverse --log_step 5 --criterion focalloss --weight_decay 0.01 --scheduler --notsavemodel
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_1_pseudo --seed 1000 --learning_rate 9e-5 --train_batch_size 32 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_0803_T_reverse --log_step 5 --criterion focalloss --weight_decay 0.01 --scheduler --notsavemodel
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_2_pseudo --seed 1000 --learning_rate 9e-5 --train_batch_size 32 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_0803_T_reverse --log_step 5 --criterion focalloss --weight_decay 0.01 --scheduler --notsavemodel # --weight_decay 0.01 
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_3_pseudo --seed 1000 --learning_rate 9e-5 --train_batch_size 32 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_0803_T_reverse --log_step 5 --criterion focalloss --weight_decay 0.01 --scheduler --notsavemodel # --weight_decay 0.01
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_4_pseudo --seed 1000 --learning_rate 9e-5 --train_batch_size 32 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_0803_T_reverse --log_step 5 --criterion focalloss --weight_decay 0.01 --scheduler --notsavemodel # --weight_decay 0.01 

# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_0_pseudo --seed 1000 --learning_rate 3e-5 --num_epoch 4 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_0803_T_reverse --log_step 5 --criterion focalloss --adv_type fgm --weight_decay 0.01 --scheduler  --diff_lr #--notsavemodel
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_1_pseudo --seed 1000 --learning_rate 3e-5 --num_epoch 4 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_0803_T_reverse --log_step 5 --criterion focalloss --adv_type fgm --weight_decay 0.01 --scheduler   #--notsavemodel
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_2_pseudo --seed 1000 --learning_rate 3e-5 --num_epoch 4 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_0803_T_reverse --log_step 5 --criterion focalloss --adv_type fgm --weight_decay 0.01 --scheduler  #--notsavemodel
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_3_pseudo --seed 1000 --learning_rate 3e-5 --num_epoch 4 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_0803_T_reverse --log_step 5 --criterion focalloss --adv_type fgm --weight_decay 0.01 --scheduler  #--notsavemodel
# python train_cn.py --model_name bert_spc_rev --dataset cn_fold_4_pseudo --seed 1000 --learning_rate 3e-5 --num_epoch 4 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_0803_T_reverse --log_step 5 --criterion focalloss --adv_type fgm --weight_decay 0.01 --scheduler  #--notsavemodel


# python train_cn_0819.py --model_name bert_spc_rev --dataset cn_fold_0_pseudo --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 25 --criterion binaryfocalloss --weight_decay 0.01 --scheduler --notsavemodel # --polarities_dim 1 --threshold 0.40 #--diff_lr
# python train_cn_0819.py --model_name bert_spc_rev --dataset cn_fold_1_pseudo --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 25 --criterion binaryfocalloss --weight_decay 0.01 --scheduler --notsavemodel #--diff_lr
# python train_cn_0819.py --model_name bert_spc_rev --dataset cn_fold_2_pseudo --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 25 --criterion binaryfocalloss --weight_decay 0.01 --scheduler --notsavemodel #--diff_lr # --weight_decay 0.01 
# python train_cn_0819.py --model_name bert_spc_rev --dataset cn_fold_3_pseudo --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 25 --criterion binaryfocalloss --weight_decay 0.01 --scheduler --notsavemodel #--diff_lr # --weight_decay 0.01
# python train_cn_0819.py --model_name bert_spc_rev --dataset cn_fold_4_pseudo --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 25 --criterion binaryfocalloss --weight_decay 0.01 --scheduler --notsavemodel #--diff_lr # --weight_decay 0.01 

# test
python train_cn.py --model_name bert_spc_rev --dataset cn_fold_0_pseudo_test --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 5 --criterion focalloss --weight_decay 0.01 --scheduler #--adv_type fgm  --diff_lr
python train_cn.py --model_name bert_spc_rev --dataset cn_fold_1_pseudo_test --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 5 --criterion focalloss --weight_decay 0.01 --scheduler #--adv_type fgm  --diff_lr
python train_cn.py --model_name bert_spc_rev --dataset cn_fold_2_pseudo_test --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 5 --criterion focalloss --weight_decay 0.01 --scheduler #--adv_type fgm  --diff_lr # --weight_decay 0.01 
python train_cn.py --model_name bert_spc_rev --dataset cn_fold_3_pseudo_test --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --log_step 25 --criterion focalloss --weight_decay 0.01 --scheduler #--adv_type fgm  --diff_lr # --weight_decay 0.01
python train_cn.py --model_name bert_spc_rev --dataset cn_fold_4_pseudo_test --seed 1000 --bert_lr 3e-5 --train_batch_size 16 --num_epoch 3 --dropout 0.3 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0729_T --log_step 5 --criterion focalloss --weight_decay 0.01 --scheduler --adv_type fgm  #--diff_lr # --weight_decay 0.01 
