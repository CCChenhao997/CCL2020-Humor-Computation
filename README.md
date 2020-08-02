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
- [x] bert + 胶囊网络
- [ ] bert + 图神经网络
- [ ] 加载最优模型，学习率*0.1，再跑几个epoch
- [x] 动态衰减
- [ ] F1优化
- [ ] EMA 指数滑动平均 [[参考]](https://zhuanlan.zhihu.com/p/51672655?utm_source=wechat_session&utm_medium=social&utm_oi=602621868809916416)
- [ ] 预训练的 Word Embedding 与 Bert 结合
- [x] BERT Post-Training [[参考]](https://github.com/howardhsu/BERT-for-RRC-ABSA)
- [ ] GHM-C loss
- [x] dialogue level作为输入（一个dialogue作为一个batch）

### 数据部分 

- [x] Speaker + Sentence
- [x] Pre-sentence + Cur-sentence
- [x] Pre-sentence + Cur-sentence + Post-sentence
- [x] Speaker_Pre-sentence + Speaker_Post-sentence
- [x] 欠采样
- [x] 过采样
- [x] 数据增强EDA
- [ ] 数据增强UDA
- [x] 数据清洗



### 训练加速

- [ ] 多卡并行训练
- [x] 混合精度训练

