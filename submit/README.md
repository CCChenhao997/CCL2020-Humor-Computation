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

| 参赛队名     | 中文f1值 | 中文acc | 英文f1值 | 英文acc | 总分  |
| ------------ | -------- | ------- | -------- | ------- | ----- |
| Walking Dad  | 0.507    | 0.739   | 0.489    | 0.755   | 2.490 |
| BERT 4EVER   | 0.517    | 0.719   | 0.463    | 0.744   | 2.443 |
| 哈哈哈       | 0.444    | 0.701   | 0.472    | 0.770   | 2.387 |
| mAI@pumc     | 0.462    | 0.736   | 0.410    | 0.745   | 2.345 |
| 我们都队     | 0.458    | 0.710   | 0.447    | 0.718   | 2.331 |
| LastDance    | 0.472    | 0.691   | 0.502    | 0.621   | 2.286 |
| will         | 0.000    | 0.749   | 0.324    | 0.728   | 1.802 |
| 好未来_AI_ML | 0.477    | 0.726   | \        | \       | \     |

--------------------

#### 第二次提交

日期：2020/07/05 （第三周）

排名：Rank 7 / 8只队伍提交结果

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

**线上指标**

| 参赛队名     | 中文f1值 | 中文acc | 英文f1值 | 英文acc | 总分  |
| ------------ | -------- | ------- | -------- | ------- | ----- |
| LastDance    | 0.489    | 0.717   | 0.571    | 0.784   | 2.560 |
| 好未来_AI_ML | 0.484    | 0.739   | 0.500    | 0.745   | 2.468 |
| 哈哈哈       | 0.511    | 0.711   | 0.463    | 0.751   | 2.437 |
| BERT 4EVER   | 0.515    | 0.721   | 0.451    | 0.738   | 2.425 |
| 我们都队     | 0.489    | 0.697   | 0.465    | 0.737   | 2.389 |
| will         | 0.397    | 0.731   | 0.301    | 0.734   | 2.164 |
| Walking Dad  | 0.484    | 0.759   | 0.224    | 0.739   | 2.206 |
| 第一次打比赛 | 0.091    | 0.755   | 0.399    | 0.299   | 1.544 |

