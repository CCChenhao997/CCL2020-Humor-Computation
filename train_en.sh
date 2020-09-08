#!/bin/bash

# * data

# bert_spc
# 0.7806/0.6766
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_0 --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 2 # bert_base
# 0.7892/0.6653
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_1 --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 2
# 0.7878/0.6653
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_2 --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 2
# 0.7911/0.6753
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_3 --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 2
# 0.7938/0.6656
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_4 --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 2

# 0.7938/0.6656 num_epoch=4
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_4 --seed 1000 --bert_lr 1e-5 --num_epoch 4 --dropout 0.2 --cuda 3

# 除了bert之外的参数进行了初始化
# 0.7878/0.6650 结果竟然还下降了
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_4 --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 2

## * adv 对抗效果明显，但epoch数要增加
# 0.7958/0.6625  num_epoch=3  adv_type == 'fgm'
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_4 --adv_type fgm --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 2
# 0.7958/0.6814  num_epoch=4  adv_type == 'fgm'
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_4 --adv_type fgm --seed 1000 --bert_lr 1e-5 --num_epoch 4 --dropout 0.2 --cuda 2
# 0.7938/0.6613  num_epoch=4  0.7938/0.6683  num_epoch=6  adv_type == 'pgd'
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_4 --adv_type pgd --seed 1000 --bert_lr 1e-5 --num_epoch 6 --dropout 0.2 --cuda 2



## * 其他模型

# bert-large-uncased 0.7705/0.6439
# python train_k_fold_cross_val.py --model bert_att --dataset en_fold_0 --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.1 --cuda 3 --pretrained_bert_name bert-large-uncased --bert_dim 1024

# Albert
# python train_k_fold_cross_val.py --model bert_spc --dataset en_fold_4 --seed 1000 --bert_lr 1e-5 --num_epoch 4 --dropout 0.2 --cuda 3 --pretrained_bert_name albert-xlarge-v2 --bert_dim 2048

# * transdata

# 0.7859/0.6743
# python train_k_fold_cross_val.py --model bert_spc --transdara True --dataset en_trans_fold_0 --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.1 --cuda 2

# 0.7886/0.6800
# python train_k_fold_cross_val.py --model bert_spc_pos --transdara True --dataset en_trans_fold_0 --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.1 --cuda 3

# 0.7906/0.6772
# python train_k_fold_cross_val.py --model bert_spc_att --transdara True --dataset en_trans_fold_0 --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.1 --cuda 3

# 0.7872/0.6615
# python train_k_fold_cross_val.py --model bert_spc_att --dataset en_trans_fold_0 --transdara True --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7866/0.6647
# python train_k_fold_cross_val.py --model bert_spc_att --dataset en_trans_fold_1 --transdara True --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7846/0.6525
# python train_k_fold_cross_val.py --model bert_spc_att --dataset en_trans_fold_2 --transdara True --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7764/0.6443
# python train_k_fold_cross_val.py --model bert_spc_att --dataset en_trans_fold_3 --transdara True --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7916/0.6881
# python train_k_fold_cross_val.py --model bert_spc_att --dataset en_trans_fold_4 --transdara True --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3

# 0.7892/0.6724
# python train_k_fold_cross_val.py --model bert_spc_pos --dataset en_trans_fold_0 --transdara True --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7826/0.6571
# python train_k_fold_cross_val.py --model bert_spc_pos --dataset en_trans_fold_1 --transdara True --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7826/0.6490
# python train_k_fold_cross_val.py --model bert_spc_pos --dataset en_trans_fold_2 --transdara True --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7797/0.6507
# python train_k_fold_cross_val.py --model bert_spc_pos --dataset en_trans_fold_3 --transdara True --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7890/0.6729
# python train_k_fold_cross_val.py --model bert_spc_pos --dataset en_trans_fold_4 --transdara True --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3

# 0.7933/0.6758
# python train_k_fold_cross_val.py --model bert_spc --dataset en_trans_fold_0 --transdara True --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7759/0.6414
# python train_k_fold_cross_val.py --model bert_spc --dataset en_trans_fold_1 --transdara True --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7772/0.6428
# python train_k_fold_cross_val.py --model bert_spc --dataset en_trans_fold_2 --transdara True --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7764/0.6258
# python train_k_fold_cross_val.py --model bert_spc --dataset en_trans_fold_3 --transdara True --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3
# 0.7910/0.6933
# python train_k_fold_cross_val.py --model bert_spc --dataset en_trans_fold_4 --transdara True --seed 1000 --bert_lr 1e-5 --num_epoch 3 --dropout 0.2 --cuda 3


# python train_en.py --model bert_spc_cnn --dataset en_fold_0_uuu --seed 1000 --bert_lr 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 --criterion focalloss #--transdata True
# python train_en.py --model bert_spc_cnn --dataset en_fold_1_uuu --seed 1000 --bert_lr 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 --criterion focalloss #--transdata True
# python train_en.py --model bert_spc_cnn --dataset en_fold_2_uuu --seed 1000 --bert_lr 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 --criterion focalloss #--transdata True
# python train_en.py --model bert_spc_cnn --dataset en_fold_3_uuu --seed 1000 --bert_lr 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 --criterion focalloss #--transdata True
# python train_en.py --model bert_spc_cnn --dataset en_fold_4_uuu --seed 1000 --bert_lr 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 --criterion focalloss #--transdata True

# python train_en.py --model bert_spc --dataset en_fold_0_aug --seed 1000 --bert_lr 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 --criterion focalloss --log_step 5 --train_batch_size_size 8
# python train_en.py --model bert_spc --dataset en_fold_1_aug --seed 1000 --bert_lr 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 --criterion focalloss --log_step 5 --train_batch_size_size 8
# python train_en.py --model bert_spc --dataset en_fold_2_aug --seed 1000 --bert_lr 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 --criterion focalloss --log_step 5 --train_batch_size_size 8
# python train_en.py --model bert_spc --dataset en_fold_3_aug --seed 1000 --bert_lr 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 --criterion focalloss --log_step 5 --train_batch_size_size 8
# python train_en.py --model bert_spc --dataset en_fold_4_aug --seed 1000 --bert_lr 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 --criterion focalloss --log_step 5 --train_batch_size_size 8

# python train_en.py --model bert_spc --dataset en_fold_0_aug_pseudo --seed 1000 --bert_lr 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 # --criterion focalloss
# python train_en.py --model bert_spc --dataset en_fold_1_aug_pseudo --seed 1000 --bert_lr 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 # --criterion focalloss
# python train_en.py --model bert_spc --dataset en_fold_2_aug_pseudo --seed 1000 --bert_lr 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 # --criterion focalloss
# python train_en.py --model bert_spc --dataset en_fold_3_aug_pseudo --seed 1000 --bert_lr 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 # --criterion focalloss
# python train_en.py --model bert_spc --dataset en_fold_4_aug_pseudo --seed 1000 --bert_lr 2e-5 --num_epoch 5 --dropout 0.5 --cuda 2 # --criterion focalloss


# * diadata
# python train_en.py --model bert_spc_rnn --dataset en_fold_0_dia --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --eval_batch_size 16 --log_step 5 --datatype diadata --rnntype LSTM --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel
# python train_en.py --model bert_spc_rnn --dataset en_fold_1_dia --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 1 --eval_batch_size 1 --log_step 5 --datatype diadata --rnntype LSTM --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel
# python train_en.py --model bert_spc_rnn --dataset en_fold_2_dia --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 1 --eval_batch_size 1 --log_step 5 --datatype diadata --rnntype LSTM --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel
# python train_en.py --model bert_spc_rnn --dataset en_fold_3_dia --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 1 --eval_batch_size 1 --log_step 5 --datatype diadata --rnntype LSTM --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel
# python train_en.py --model bert_spc_rnn --dataset en_fold_4_dia --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 1 --eval_batch_size 1 --log_step 5 --datatype diadata --rnntype LSTM --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel

# * aug
# bert_base_uncased_0728_T
# python train_en.py --model_name bert_spc_rnn --dataset en_fold_0_dia --datatype diadata --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 3 --criterion focalloss --train_batch_size 1 --eval_batch_size 1 --diff_lr --dia_maxlength 64 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel # --adv_type fgm
# python train_en.py --model_name bert_spc_rnn --dataset en_fold_1_dia --datatype diadata --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 3 --criterion focalloss --train_batch_size 1 --eval_batch_size 1 --diff_lr --dia_maxlength 64 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel # --adv_type fgm
# python train_en.py --model_name bert_spc_rnn --dataset en_fold_2_dia --datatype diadata --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 3 --criterion focalloss --train_batch_size 1 --eval_batch_size 1 --diff_lr --dia_maxlength 64 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel # --adv_type fgm
# python train_en.py --model_name bert_spc_rnn --dataset en_fold_3_dia --datatype diadata --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 3 --criterion focalloss --train_batch_size 1 --eval_batch_size 1 --diff_lr --dia_maxlength 64 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel # --adv_type fgm
# python train_en.py --model_name bert_spc_rnn --dataset en_fold_4_dia --datatype diadata --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 3 --criterion focalloss --train_batch_size 1 --eval_batch_size 1 --diff_lr --log_step 50 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel # --adv_type fgm

# python train_en.py --model_name bert_spc --dataset en_fold_0_dia_aug --datatype raw --seed 3000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel # --adv_type fgm
# python train_en.py --model_name bert_spc_rev --dataset en_fold_1_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion crossentropy --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel # --adv_type fgm
# python train_en.py --model_name bert_spc_rev --dataset en_fold_2_dia_aug --datatype raw --seed 3000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion crossentropy --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel # --adv_type fgm
# python train_en.py --model_name bert_spc --dataset en_fold_3_dia_aug --datatype raw --seed 3000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel # --adv_type fgm
# python train_en.py --model_name bert_spc --dataset en_fold_4_dia_aug --datatype raw --seed 3000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel # --adv_type fgm

# ? weight_score dia_aug
# * en_fold_0 f1_0.7128_f1_0_0.8680_f1_1_0.5576_acc_0.7967_score_1.3542
# python train_en.py --model_name bert_spc --dataset en_fold_0_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler # --notsavemodel
# * en_fold_1 f1_0.6958_f1_0_0.8252_f1_1_0.5664_acc_0.7508_score_1.3173
# python train_en.py --model_name bert_spc_rev --dataset en_fold_1_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion crossentropy --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler # --notsavemodel
# * en_fold_2 f1_0.6703_f1_0_0.8477_f1_1_0.4929_acc_0.7657_score_1.2586
# python train_en.py --model_name bert_spc_rev --dataset en_fold_2_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler # --notsavemodel
# * en_fold_3 f1_0.6718_f1_0_0.8388_f1_1_0.5049_acc_0.7568_score_1.2617
# python train_en.py --model_name bert_spc_rev --dataset en_fold_3_dia_aug --datatype raw --seed 3000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler # --notsavemodel
# * en_fold_4 f1_0.7189_f1_0_0.8817_f1_1_0.5561_acc_0.8132_score_1.3693
# python train_en.py --model_name bert_spc_lay --dataset en_fold_4_dia_aug --datatype raw --seed 3000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler # --notsavemodel
# * en_fold_4 f1_0.7189_f1_0_0.8543_f1_1_0.5835_acc_0.7841_score_1.3676
# python train_en.py --model_name bert_spc --dataset en_fold_4_dia_aug --datatype raw --seed 3000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler # --notsavemodel

# ? f1 dia_aug
# * en_fold_0 f1_0.7128_f1_0_0.8680_f1_1_0.5576_acc_0.7967_score_1.3432
# python train_en.py --model_name bert_spc --dataset en_fold_0_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel
# * en_fold_1 f1_0.6978_f1_0_0.8518_f1_1_0.5437_acc_0.7763_score_1.3211
# python train_en.py --model_name bert_spc_rev --dataset en_fold_1_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion crossentropy --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel
# * en_fold_2 f1_0.6706_f1_0_0.8553_f1_1_0.4858_acc_0.7742_score_1.2667.txt
# python train_en.py --model_name bert_spc_rev --dataset en_fold_2_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel
# * f1_0.6718_f1_0_0.8388_f1_1_0.5049_acc_0.7568_score_1.2635.txt
# python train_en.py --model_name bert_spc_rev --dataset en_fold_3_dia_aug --datatype raw --seed 3000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel
# * f1_0.7189_f1_0_0.8817_f1_1_0.5561_acc_0.8132_score_1.3663.txt
# python train_en.py --model_name bert_spc_lay --dataset en_fold_4_dia_aug --datatype raw --seed 3000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel
# * f1_0.7189_f1_0_0.8543_f1_1_0.5835_acc_0.7841_score_1.3685.txt
# python train_en.py --model_name bert_spc --dataset en_fold_4_dia_aug --datatype raw --seed 3000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel


# python train_en.py --model_name bert_spc --dataset en_fold_0_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel
# python train_en.py --model_name bert_spc_rev --dataset en_fold_1_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion crossentropy --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel
# python train_en.py --model_name bert_spc_rev --dataset en_fold_2_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel
# python train_en.py --model_name bert_spc_rev --dataset en_fold_3_dia_aug --datatype raw --seed 3000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel
# python train_en.py --model_name bert_spc_lay --dataset en_fold_4_dia_aug --datatype raw --seed 3000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel
# python train_en.py --model_name bert_spc --dataset en_fold_4_dia_aug --datatype raw --seed 3000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel

# python train_en.py --model_name bert_spc_rev --dataset en_fold_2_dia_aug --datatype raw --seed 2000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler --notsavemodel
# python train_en.py --model_name bert_spc_rev --dataset en_fold_3_dia_aug --datatype raw --seed 2000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler --notsavemodel

# python train_en.py --model_name bert_spc_rev --dataset en_fold_2_dia_aug --datatype raw --seed 2000 --bert_lr 3e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler --notsavemodel
# python train_en.py --model_name bert_spc_rev --dataset en_fold_3_dia_aug --datatype raw --seed 2000 --bert_lr 3e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler --notsavemodel

# python train_en.py --model_name bert_spc_rev --dataset en_fold_2_dia_aug --datatype raw --seed 7777 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler --notsavemodel
# python train_en.py --model_name bert_spc_rev --dataset en_fold_3_dia_aug --datatype raw --seed 7777 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler --notsavemodel

# * f1_0.6770_f1_0_0.8250_f1_1_0.5290_acc_0.7448_wacc_0.7468_score_1.2737
python train_en.py --model_name bert_spc_rev --dataset en_fold_2_dia_aug --datatype raw --seed 7777 --bert_lr 3e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler # --notsavemodel


# python train_en_0819.py --model_name bert_spc --dataset en_fold_0_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion binaryfocalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel --polarities_dim 1 --threshold 0.40
# python train_en_0819.py --model_name bert_spc_rev --dataset en_fold_2_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion binaryfocalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel --polarities_dim 1 --threshold 0.40

# ? weight_score dia_aug_pseudo
# * f1_0.7105_f1_0_0.8639_f1_1_0.5570_acc_0.7918_score_1.3488
# python train_en.py --model_name bert_spc --dataset en_fold_0_dia_aug_pseudo --datatype raw --seed 1000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler
# * f1_0.6948_f1_0_0.8232_f1_1_0.5665_acc_0.7488_score_1.3153
# python train_en.py --model_name bert_spc_rev --dataset en_fold_1_dia_aug_pseudo --datatype raw --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion crossentropy --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler
# * f1_0.6668_f1_0_0.8225_f1_1_0.5111_acc_0.7395_score_1.2506
# python train_en.py --model_name bert_spc_rev --dataset en_fold_2_dia_aug_pseudo --datatype raw --seed 1000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler
# * f1_0.6747_f1_0_0.8480_f1_1_0.5015_acc_0.7670_score_1.2685
# python train_en.py --model_name bert_spc_rev --dataset en_fold_3_dia_aug_pseudo --datatype raw --seed 3000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler
# * f1_0.7181_f1_0_0.8552_f1_1_0.5811_acc_0.7847_score_1.3659
# python train_en.py --model_name bert_spc_lay --dataset en_fold_4_dia_aug_pseudo --datatype raw --seed 3000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler
# * f1_0.7212_f1_0_0.8703_f1_1_0.5722_acc_0.8009_score_1.3731
# python train_en.py --model_name bert_spc --dataset en_fold_4_dia_aug_pseudo --datatype raw --seed 3000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler


# python train_en.py --model_name bert_spc_lay --dataset en_fold_0_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel # --adv_type fgm
# python train_en.py --model_name bert_spc_lay --dataset en_fold_1_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel # --adv_type fgm
# python train_en.py --model_name bert_spc_lay --dataset en_fold_2_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel # --adv_type fgm
# python train_en.py --model_name bert_spc_lay --dataset en_fold_3_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel # --adv_type fgm
# python train_en.py --model_name bert_spc_lay --dataset en_fold_4_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --diff_lr --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel # --adv_type fgm

# python train_en.py --model_name bert_spc_lay --dataset en_fold_0_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler # --notsavemodel # --adv_type fgm
# python train_en.py --model_name bert_spc_lay --dataset en_fold_1_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler # --notsavemodel # --adv_type fgm
# python train_en.py --model_name bert_spc_lay --dataset en_fold_2_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name bert-base-uncased --scheduler # --notsavemodel # --adv_type fgm
# python train_en.py --model_name bert_spc_lay --dataset en_fold_3_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name bert-base-uncased --scheduler # --notsavemodel # --adv_type fgm
# python train_en.py --model_name bert_spc_lay --dataset en_fold_4_dia_aug --datatype raw --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler # --notsavemodel # --adv_type fgm


# python train_en.py --model_name bert_spc --dataset en_fold_0_aug --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler --notsavemodel --adv_type fgm
# python train_en.py --model_name bert_spc --dataset en_fold_1_aug --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler --notsavemodel --adv_type fgm
# python train_en.py --model_name bert_spc --dataset en_fold_2_aug --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler --notsavemodel --adv_type fgm
# python train_en.py --model_name bert_spc --dataset en_fold_3_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler --notsavemodel --adv_type fgm
# python train_en.py --model_name bert_spc --dataset en_fold_4_aug --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler --notsavemodel --adv_type fgm

# python train_en.py --model_name bert_spc --dataset en_fold_0_aug --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler --notsavemodel
# python train_en.py --model_name bert_spc --dataset en_fold_1_aug --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler --notsavemodel
# python train_en.py --model_name bert_spc --dataset en_fold_2_aug --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler --notsavemodel
# python train_en.py --model_name bert_spc --dataset en_fold_3_aug --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler --notsavemodel
# python train_en.py --model_name bert_spc --dataset en_fold_4_aug --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler --notsavemodel

# adv
# python train_en.py --model_name bert_spc --dataset en_fold_0_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --adv_type fgm
# python train_en.py --model_name bert_spc --dataset en_fold_1_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --adv_type fgm
# python train_en.py --model_name bert_spc --dataset en_fold_2_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --adv_type fgm
# python train_en.py --model_name bert_spc --dataset en_fold_3_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --adv_type fgm
# python train_en.py --model_name bert_spc --dataset en_fold_4_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --adv_type fgm

# bert_base_uncased_0730_TD
# python train_en.py --model_name bert_spc_rev --dataset en_fold_0_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler --adv_type fgm
# python train_en.py --model_name bert_spc_rev --dataset en_fold_1_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler --adv_type fgm
# python train_en.py --model_name bert_spc_rev --dataset en_fold_2_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler --adv_type fgm
# python train_en.py --model_name bert_spc_rev --dataset en_fold_3_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler --adv_type fgm
# python train_en.py --model_name bert_spc_rev --dataset en_fold_4_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0730_TD --scheduler --adv_type fgm

# * raw dia
# python train_en.py --model_name bert_spc --dataset en_fold_0_aug_pseudo --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --datatype raw --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel
# python train_en.py --model_name bert_spc --dataset en_fold_1_aug_pseudo --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --datatype raw --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel
# python train_en.py --model_name bert_spc --dataset en_fold_2_aug_pseudo --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --datatype raw --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel
# python train_en.py --model_name bert_spc --dataset en_fold_3_aug_pseudo --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --datatype raw --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel
# python train_en.py --model_name bert_spc --dataset en_fold_4_aug_pseudo --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --datatype raw --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --notsavemodel

# * en pseudo
# python train_en.py --model_name bert_spc --dataset en_fold_0_pseudo --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.5 --cuda 2 --max_length 80 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --log_step 25 --criterion focalloss # --weight_decay 0.01
# python train_en.py --model_name bert_spc --dataset en_fold_1_pseudo --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.5 --cuda 2 --max_length 80 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --log_step 25 --criterion focalloss # --weight_decay 0.01 
# python train_en.py --model_name bert_spc --dataset en_fold_2_pseudo --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.5 --cuda 2 --max_length 80 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --log_step 25 --criterion focalloss # --weight_decay 0.01 
# python train_en.py --model_name bert_spc --dataset en_fold_3_pseudo --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.5 --cuda 2 --max_length 80 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --log_step 25 --criterion focalloss # --weight_decay 0.01
# python train_en.py --model_name bert_spc --dataset en_fold_4_pseudo --seed 1000 --bert_lr 2e-5 --num_epoch 3 --dropout 0.5 --cuda 2 --max_length 80 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --log_step 25 --criterion focalloss # --weight_decay 0.01 

# python train_en.py --model_name bert_spc --dataset en_fold_0_dia_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --datatype raw --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T
# python train_en.py --model_name bert_spc --dataset en_fold_1_dia_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --datatype raw --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T
# python train_en.py --model_name bert_spc --dataset en_fold_2_dia_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --datatype raw --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T
# python train_en.py --model_name bert_spc --dataset en_fold_3_dia_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --datatype raw --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T
# python train_en.py --model_name bert_spc --dataset en_fold_4_dia_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --datatype raw --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T


# python train_en.py --model_name bert_spc --dataset en_fold_0_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion crossentropy --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler
# python train_en.py --model_name bert_spc --dataset en_fold_1_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion crossentropy --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler
# python train_en.py --model_name bert_spc --dataset en_fold_2_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion crossentropy --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler
# python train_en.py --model_name bert_spc --dataset en_fold_3_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion crossentropy --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler
# python train_en.py --model_name bert_spc --dataset en_fold_4_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion crossentropy --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler

# python train_en.py --model_name bert_spc --dataset en_fold_0_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.5 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler
# python train_en.py --model_name bert_spc --dataset en_fold_1_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.5 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler
# python train_en.py --model_name bert_spc --dataset en_fold_2_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.5 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler
# python train_en.py --model_name bert_spc --dataset en_fold_3_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.5 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler
# python train_en.py --model_name bert_spc --dataset en_fold_4_aug --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.5 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 5 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler

# python train_en.py --model_name bert_spc --dataset en_fold_0 --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 25 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --smooth 0.1 --gamma 2 --notsavemodel
# python train_en.py --model_name bert_spc --dataset en_fold_1 --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 25 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --smooth 0.1 --gamma 2 --notsavemodel
# python train_en.py --model_name bert_spc --dataset en_fold_2 --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 25 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --smooth 0.1 --gamma 2 --notsavemodel
# python train_en.py --model_name bert_spc --dataset en_fold_3 --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 25 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --smooth 0.1 --gamma 2 --notsavemodel
# python train_en.py --model_name bert_spc --dataset en_fold_4 --seed 1000 --bert_lr 2e-5 --num_epoch 4 --dropout 0.1 --cuda 2 --criterion focalloss --train_batch_size 16 --log_step 25 --weight_decay 0.01 --pretrained_bert_name ./pretrain_models/bert_base_uncased_0728_T --scheduler --smooth 0.1 --gamma 2 --notsavemodel