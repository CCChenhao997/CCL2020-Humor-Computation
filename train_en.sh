#!/bin/bash

# * data

# bert_spc
# 0.7806/0.6766
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_0 --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 2 # bert_base
# 0.7892/0.6653
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_1 --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 2
# 0.7878/0.6653
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_2 --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 2
# 0.7911/0.6753
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_3 --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 2
# 0.7938/0.6656
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_4 --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 2

# 0.7938/0.6656 num_epoch=4
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_4 --seed 1000 --learning_rate 1e-5 --num_epoch 4 --dropout 0.2 --cuda 3

# 除了bert之外的参数进行了初始化
# 0.7878/0.6650 结果竟然还下降了
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_4 --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 2

## * adv 对抗效果明显，但epoch数要增加
# 0.7958/0.6625  num_epoch=3  adv_type == 'fgm'
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_4 --adv_type fgm --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 2
# 0.7958/0.6814  num_epoch=4  adv_type == 'fgm'
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_4 --adv_type fgm --seed 1000 --learning_rate 1e-5 --num_epoch 4 --dropout 0.2 --cuda 2
# 0.7938/0.6613  num_epoch=4  0.7938/0.6683  num_epoch=6  adv_type == 'pgd'
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_4 --adv_type pgd --seed 1000 --learning_rate 1e-5 --num_epoch 6 --dropout 0.2 --cuda 2



## * 其他模型

# bert-large-uncased 0.7705/0.6439
# python train_k_fold_cross_val.py --model bert_att --dataset en_fold_0 --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.1 --cuda 3 --pretrained_bert_name bert-large-uncased --bert_dim 1024

# Albert
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_4 --seed 1000 --learning_rate 1e-5 --num_epoch 4 --dropout 0.2 --cuda 3 --pretrained_bert_name albert-xlarge-v2 --bert_dim 2048

# * transdata

# 0.7859/0.6743
# python train_k_fold_cross_val.py --model bert_spc --transdara True --dataset en_trans_fold_0 --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.1 --cuda 2

# 0.7886/0.6800
# python train_k_fold_cross_val.py --model bert_spc_pos --transdara True --dataset en_trans_fold_0 --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.1 --cuda 3

# 0.7906/0.6772
# python train_k_fold_cross_val.py --model bert_spc_att --transdara True --dataset en_trans_fold_0 --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.1 --cuda 3

# 0.7872/0.6615
# python train_k_fold_cross_val.py --model bert_spc_att --dataset en_trans_fold_0 --transdara True --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7866/0.6647
# python train_k_fold_cross_val.py --model bert_spc_att --dataset en_trans_fold_1 --transdara True --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7846/0.6525
# python train_k_fold_cross_val.py --model bert_spc_att --dataset en_trans_fold_2 --transdara True --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7764/0.6443
# python train_k_fold_cross_val.py --model bert_spc_att --dataset en_trans_fold_3 --transdara True --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7916/0.6881
# python train_k_fold_cross_val.py --model bert_spc_att --dataset en_trans_fold_4 --transdara True --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3

# 0.7892/0.6724
# python train_k_fold_cross_val.py --model bert_spc_pos --dataset en_trans_fold_0 --transdara True --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7826/0.6571
# python train_k_fold_cross_val.py --model bert_spc_pos --dataset en_trans_fold_1 --transdara True --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7826/0.6490
# python train_k_fold_cross_val.py --model bert_spc_pos --dataset en_trans_fold_2 --transdara True --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7797/0.6507
# python train_k_fold_cross_val.py --model bert_spc_pos --dataset en_trans_fold_3 --transdara True --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7890/0.6729
# python train_k_fold_cross_val.py --model bert_spc_pos --dataset en_trans_fold_4 --transdara True --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3

# 0.7933/0.6758
# python train_k_fold_cross_val.py --model bert_spc --dataset en_trans_fold_0 --transdara True --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7759/0.6414
# python train_k_fold_cross_val.py --model bert_spc --dataset en_trans_fold_1 --transdara True --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7772/0.6428
# python train_k_fold_cross_val.py --model bert_spc --dataset en_trans_fold_2 --transdara True --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7764/0.6258
# python train_k_fold_cross_val.py --model bert_spc --dataset en_trans_fold_3 --transdara True --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7910/0.6933
# python train_k_fold_cross_val.py --model bert_spc --dataset en_trans_fold_4 --transdara True --seed 1000 --learning_rate 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3


# python train_en.py --model bert_spc_cnn --dataset en_fold_0_uuu --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 --criterion focalloss #--transdata True
# python train_en.py --model bert_spc_cnn --dataset en_fold_1_uuu --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 --criterion focalloss #--transdata True
# python train_en.py --model bert_spc_cnn --dataset en_fold_2_uuu --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 --criterion focalloss #--transdata True
# python train_en.py --model bert_spc_cnn --dataset en_fold_3_uuu --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 --criterion focalloss #--transdata True
# python train_en.py --model bert_spc_cnn --dataset en_fold_4_uuu --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 --criterion focalloss #--transdata True

# python train_en.py --model bert_spc --dataset en_fold_0_aug --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 --criterion focalloss --log_step 5 --batch_size 8
# python train_en.py --model bert_spc --dataset en_fold_1_aug --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 --criterion focalloss --log_step 5 --batch_size 8
# python train_en.py --model bert_spc --dataset en_fold_2_aug --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 --criterion focalloss --log_step 5 --batch_size 8
# python train_en.py --model bert_spc --dataset en_fold_3_aug --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 --criterion focalloss --log_step 5 --batch_size 8
# python train_en.py --model bert_spc --dataset en_fold_4_aug --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 --criterion focalloss --log_step 5 --batch_size 8

# python train_en.py --model bert_spc --dataset en_fold_0_aug_pseudo --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 # --criterion focalloss
# python train_en.py --model bert_spc --dataset en_fold_1_aug_pseudo --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 # --criterion focalloss
# python train_en.py --model bert_spc --dataset en_fold_2_aug_pseudo --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 # --criterion focalloss
# python train_en.py --model bert_spc --dataset en_fold_3_aug_pseudo --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 # --criterion focalloss
# python train_en.py --model bert_spc --dataset en_fold_4_aug_pseudo --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 # --criterion focalloss


# * diadata
# python train_en.py --model bert_spc_rnn --dataset en_fold_0_dia --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.1 --cuda 2 --criterion focalloss --batch_size 1 --log_step 50 --datatype diadata --rnntype LSTM
# python train_en.py --model bert_spc_rnn --dataset en_fold_1_dia --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.1 --cuda 2 --criterion focalloss --batch_size 1 --log_step 50 --datatype diadata --rnntype LSTM
# python train_en.py --model bert_spc_rnn --dataset en_fold_2_dia --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.1 --cuda 2 --criterion focalloss --batch_size 1 --log_step 50 --datatype diadata --rnntype LSTM
# python train_en.py --model bert_spc_rnn --dataset en_fold_3_dia --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.1 --cuda 2 --criterion focalloss --batch_size 1 --log_step 50 --datatype diadata --rnntype LSTM
# python train_en.py --model bert_spc_rnn --dataset en_fold_4_dia --seed 1000 --learning_rate 2e-5 --num_epoch 5 --dropout 0.1 --cuda 2 --criterion focalloss --batch_size 1 --log_step 50 --datatype diadata --rnntype LSTM

# * aug
# bert_base_uncased_0728_T
python train_en.py --model_name bert_spc_rev --dataset en_fold_0_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler True
python train_en.py --model_name bert_spc_rev --dataset en_fold_1_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler True
python train_en.py --model_name bert_spc_rev --dataset en_fold_2_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler True
python train_en.py --model_name bert_spc_rev --dataset en_fold_3_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler True
python train_en.py --model_name bert_spc_rev --dataset en_fold_4_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler True

# adv
# python train_en.py --model_name bert_spc --dataset en_fold_0_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --adv_type fgm
# python train_en.py --model_name bert_spc --dataset en_fold_1_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --adv_type fgm
# python train_en.py --model_name bert_spc --dataset en_fold_2_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --adv_type fgm
# python train_en.py --model_name bert_spc --dataset en_fold_3_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --adv_type fgm
# python train_en.py --model_name bert_spc --dataset en_fold_4_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --adv_type fgm

# bert_base_uncased_0730_TD
python train_en.py --model_name bert_spc_rev --dataset en_fold_0_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler True
python train_en.py --model_name bert_spc_rev --dataset en_fold_1_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler True
python train_en.py --model_name bert_spc_rev --dataset en_fold_2_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler True
python train_en.py --model_name bert_spc_rev --dataset en_fold_3_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler True
python train_en.py --model_name bert_spc_rev --dataset en_fold_4_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler True

# * raw dia
# python train_en.py --model_name bert_spc_rev --dataset en_fold_0_dia_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --datatype raw --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T 
# python train_en.py --model_name bert_spc_rev --dataset en_fold_1_dia_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --datatype raw --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T 
# python train_en.py --model_name bert_spc_rev --dataset en_fold_2_dia_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --datatype raw --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T 
# python train_en.py --model_name bert_spc_rev --dataset en_fold_3_dia_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --datatype raw --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T 
# python train_en.py --model_name bert_spc_rev --dataset en_fold_4_dia_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --datatype raw --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T 

# * cn
# python train_en.py --model_name bert_spc --dataset cn_fold_0_pseudo --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.5 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --log_step 5 --criterion focalloss # --weight_decay 0.01
# python train_en.py --model_name bert_spc_rev --dataset cn_fold_1_pseudo --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.5 --cuda 2 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --log_step 25 --criterion focalloss # --weight_decay 0.01 
# python train_en.py --model_name bert_spc --dataset cn_fold_2_pseudo --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.5 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --log_step 5 --criterion focalloss # --weight_decay 0.01 
# python train_en.py --model_name bert_spc --dataset cn_fold_3_pseudo --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.5 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --log_step 5 --criterion focalloss # --weight_decay 0.01
# python train_en.py --model_name bert_spc --dataset cn_fold_4_pseudo --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.5 --cuda 3 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --log_step 5 --criterion focalloss # --weight_decay 0.01 

# python train_en.py --model_name bert_spc --dataset en_fold_0_dia_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --datatype raw --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T
# python train_en.py --model_name bert_spc --dataset en_fold_1_dia_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --datatype raw --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T
# python train_en.py --model_name bert_spc --dataset en_fold_2_dia_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --datatype raw --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T
# python train_en.py --model_name bert_spc --dataset en_fold_3_dia_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --datatype raw --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T
# python train_en.py --model_name bert_spc --dataset en_fold_4_dia_aug --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --datatype raw --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T


# python train_cn.py --model bert_spc --dataset cn_fold_0_aug --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0728 --criterion focalloss --log_step 5 --weight_decay 0.01
# python train_cn.py --model bert_spc --dataset cn_fold_1_aug --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0728 --criterion focalloss --log_step 5 --weight_decay 0.01
# python train_cn.py --model bert_spc --dataset cn_fold_2_aug --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0728 --criterion focalloss --log_step 5 --weight_decay 0.01
# python train_cn.py --model bert_spc --dataset cn_fold_3_aug --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0728 --criterion focalloss --log_step 5 --weight_decay 0.01
# python train_cn.py --model bert_spc --dataset cn_fold_4_aug --seed 1000 --learning_rate 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn_0728 --criterion focalloss --log_step 5 --weight_decay 0.01

# python train_cn.py --model bert_spc --dataset cn_fold_0 --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --log_step 5 --weight_decay 0.01 --adv_type fgm
# python train_cn.py --model bert_spc --dataset cn_fold_1 --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --log_step 5 --weight_decay 0.01 --adv_type fgm
# python train_cn.py --model bert_spc --dataset cn_fold_2 --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --log_step 5 --weight_decay 0.01 --adv_type fgm
# python train_cn.py --model bert_spc --dataset cn_fold_3 --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --log_step 5 --weight_decay 0.01 --adv_type fgm
# python train_cn.py --model bert_spc --dataset cn_fold_4 --seed 1000 --learning_rate 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --max_length 80 --pretrained_bert_name ./pretrain_models/ERNIE_cn --criterion focalloss --log_step 5 --weight_decay 0.01 --adv_type fgm