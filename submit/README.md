### 提交记录

#### 第一次提交

日期：2020/06/27 （第二周）

排名：Rank 1 / 8只队伍提交结果

英文模型：bert_spc

输入：Speaker + Sentence

模型参数：

- seed = 1000 
- learning rate = 2e-5
- epoch = 3
- k-fold = 5
- dropout = 0.5
- max_length = 80
- weight_decay = 0.0
- FocalLoss(num_class=2, alpha=0.25, gamma=2, smooth=0.2)
- adv_type = None
- cuda = unkown



中文模型：bert_spc

输入：Speaker + Sentence

模型参数：

- seed = 1000 
- learning rate = 2e-5
- epoch = 3
- k-fold = 5
- dropout = 0.5
- max_length = 80
- weight_decay = 0.0
- FocalLoss(num_class=2, alpha=0.25, gamma=2, smooth=0.2)
- adv_type = None
- cuda = unkown

**线下指标**

| EN/CN | K-fold | F1     | Pseudo_F1 |
| ----- | ------ | ------ | --------- |
| EN    | 0      | 0.6552 | 0.6870    |
| EN    | 1      | 0.6627 | 0.6807    |
| EN    | 2      | 0.6703 | 0.7068    |
| EN    | 3      | 0.6753 | 0.6881    |
| EN    | 4      | 0.6711 | 0.7092    |
| CN    | 0      | 0.6540 | 0.6710    |
| CN    | 1      | 0.6440 | 0.6826    |
| CN    | 2      | 0.6376 | 0.6632    |
| CN    | 3      | 0.6434 | 0.6790    |
| CN    | 4      | 0.6493 | 0.6713    |

**线上指标**

| 参赛队名    | 中文f1值 | 中文acc | 英文f1值 | 英文acc | 总分  |
| ----------- | -------- | ------- | -------- | ------- | ----- |
| Walking Dad | 0.507    | 0.739   | 0.489    | 0.755   | 2.490 |

--------------------

#### 第二次提交

日期：2020/07/05 （第三周）

排名：

英文模型：bert_spc

输入：Speaker + Sentence

模型参数：

- seed = 1000 
- learning rate = 2e-5
- epoch = 5
- k-fold = 5
- dropout = 0.1
- max_length = 80
- weight_decay = 0.0
- FocalLoss(num_class=2, alpha=0.25, gamma=2, smooth=0.2)
- adv_type = fgm
- cuda = 2



中文模型：bert_spc

输入：Speaker + Sentence

模型参数：

- seed = 1000 
- learning rate = 2e-5
- epoch = 3
- k-fold = 5
- dropout = 0.1
- max_length = 128
- weight_decay = 0.0
- FocalLoss(num_class=2, alpha=0.25, gamma=2, smooth=0.2)
- adv_type = fgm
- cuda = 2 and 3

**线下指标**

score = weighted_acc + f1

| EN/CN | K-fold | score  | Pseudo score |
| ----- | ------ | ------ | ------------ |
| EN    | 0      | 1.4303 | 1.4734       |
| EN    | 1      | 1.4434 | 1.4553       |
| EN    | 2      | 1.4481 | 1.4593       |
| EN    | 3      | 1.4569 | 1.4274       |
| EN    | 4      | 1.4566 | 1.4384       |
| CN    | 0      | 1.3849 | 1.3920       |
| CN    | 1      | 1.3520 | 1.4059       |
| CN    | 2      | 1.3619 | 1.4094       |
| CN    | 3      | 1.3773 | 1.3736       |
| CN    | 4      | 1.3710 | 1.4310       |

