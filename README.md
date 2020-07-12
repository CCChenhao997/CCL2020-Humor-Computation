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
- [ ] BERT Post-Training [[参考]](https://github.com/howardhsu/BERT-for-RRC-ABSA)

### 数据部分 

- [x] Speaker + Sentence
- [x] Pre-sentence + Cur-sentence
- [ ] Pre-sentence + Cur-sentence + Post-sentence
- [ ] Speaker_Pre-sentence + Speaker_Post-sentence
- [x] 欠采样
- [x] 过采样
- [x] 数据增强 [[参考1]](https://zhuanlan.zhihu.com/p/145521255?utm_source=wechat_session&utm_medium=social&utm_oi=602621868809916416) [[参考2]](https://github.com/tongchangD/text_data_enhancement_with_LaserTagger?utm_source=wechat_session&utm_medium=social&utm_oi=602621868809916416) [[参考3]](https://github.com/QData/TextAttack) [[参考4]](https://github.com/flyingwaters/EDA-Easier-Data-Augment-for-chinese) [[参考5]](https://github.com/zhanlaoban/eda_nlp_for_Chinese)



### 训练加速

- [ ] 多卡并行训练
- [x] 双精度训练

