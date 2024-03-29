# NOTES

<!-- TOC -->
- [NOTES](#notes)    
  - [文本数据增强的方式](#文本数据增强的方式)        
    - [文本本身](#文本本身)        
    - [外部帮助](#外部帮助)    
  - [Contrastive Loss和Triplet Loss的改进](#contrastive-loss和triplet-loss的改进)        
    - [度量学习](#度量学习)
<!-- /TOC -->

## 文本数据增强的方式

### 文本本身

- 随机删除一个词
- 随机选择一个词，用它的同义词替换
- 随机选择两个词，然后交换它们的位置
- 随机选择一个词，然后随机选择一个它的近义词，然后随机插入句子的任意位置

### 外部帮助

- 翻译为另一种语言再翻译回原语言

## Contrastive Loss和Triplet Loss的改进

### 度量学习

Metric Learning, 模型是一个Encoder，将输入编码为一个特征向量，属于同一类别的样本距离近，不同类别样本距离远，通常使用pair-based loss，当然也可以使用[分类方法](<https://spaces.ac.cn/archives/5743>)看待这个问题

