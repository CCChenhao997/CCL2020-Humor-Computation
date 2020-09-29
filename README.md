# CCL2020-Humor-Computation
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
CCL2020 第二届“小牛杯”幽默计算——情景喜剧笑点识别

---------------

## 竞赛提分尝试

### 模型部分

- [x] 交叉验证（五折或十折）
- [x] 模型融合 + 伪标签
- [x] Focalloss + 标签平滑
- [x] 对抗训练（FGM）
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
- [ ] Flooding [[参考]](https://mp.weixin.qq.com/s/GpwEu8jTv46LiF6dQeJMQQ)

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
- [x] dialogue level作为输入（一个dialogue作为一个batch）



### 训练加速

- [ ] 多卡并行训练
- [x] 混合精度训练


## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/airflier"><img src="https://avatars0.githubusercontent.com/u/51824426?v=4" width="100px;" alt=""/><br /><sub><b>airflier</b></sub></a><br /><a href="https://github.com/CCChenhao997/CCL2020-Humor-Computation/commits?author=airflier" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!