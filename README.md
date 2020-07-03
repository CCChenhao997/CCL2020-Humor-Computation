# CCL2020-Humor-Computation
CCL2020 第二届“小牛杯”幽默计算——情景喜剧笑点识别

---------------

## 竞赛提分尝试

### 模型部分

- [x] 交叉验证（五折或十折）
- [x] 模型融合 + 伪标签
- [x] Focalloss + 标签平滑
- [x] 对抗训练
- [x] 加权Accuracy + F1选择模型
- [] bert + 胶囊网络
- [] 加载最优模型，学习率*0.1，再跑几个epoch
- [] 动态衰减
- [] F1优化
- [] EMA 指数滑动平均 [[参考]](https://zhuanlan.zhihu.com/p/51672655?utm_source=wechat_session&utm_medium=social&utm_oi=602621868809916416)
- [] 预训练的 Word Embedding 与 Bert 结合

### 数据部分 

- [x] Speaker + Sentence
- [x] Pre-sentence + Post-sentence
- [] 数据增强 [[参考1]](https://zhuanlan.zhihu.com/p/145521255?utm_source=wechat_session&utm_medium=social&utm_oi=602621868809916416) [[参考2]](https://github.com/tongchangD/text_data_enhancement_with_LaserTagger?utm_source=wechat_session&utm_medium=social&utm_oi=602621868809916416) [[参考3]](https://github.com/QData/TextAttack)

---------------------------

## 比赛记录

### 提交记录

#### 第一次提交

日期：2020/06/27 （第二周）

排名：Rank 1 / 8只队伍提交结果

英文模型：bert_spc

输入：Speaker + Sentence

模型参数：

- cn_seed = 1000 
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

- cn_seed = 1000 
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

线上指标

| 参赛队名    | 中文f1值 | 中文acc | 英文f1值 | 英文acc | 总分  |
| ----------- | -------- | ------- | -------- | ------- | ----- |
| Walking Dad | 0.507    | 0.739   | 0.489    | 0.755   | 2.490 |

--------------------

#### 第二次提交

日期：2020/07/05 （第三周）

