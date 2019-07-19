---
layout:     post
title:     DeepLearning.ai-Week1-深度学习概括
subtitle:   2019-04-23 深度学习课程
date:       2019-04-23
author:     koko
header-img: img/post-bg-universe.jpg
catalog: true
catalog: true
tags:
- 深度学习
- deepLearning.ai
---

# 什么是神经网络？

- 房价预测和单个神经元的预测
	- 房子面积（输入值）与价格（目标值）的关系
	- 该模型是在模拟RELU函数
	- ![](https://ws4.sinaimg.cn/large/006tNc79ly1g2ccjh1m5gj31d60r0ad5.jpg) 

- 房价预测和多个神经元的预测
	- 房子面积，卧室数量，邮编，小区富裕程度（四个输入）与价格（一个目标值）的关系
	- 每个神经元都可能在模拟RELU函数
	- 我们无需考虑每个神经元代表的特征值处理（如家庭规模，超市距离，学区质量）
	- 我们只需提供足够多的数据（输入值和目标值），模型自动筛选创造特征值
	- ![](https://ws1.sinaimg.cn/large/006tNc79ly1g2ccoe6t2oj31820naq66.jpg)
	- 三个神经元通过调节四个weights的大小，来自动创建3个特征，至于特征是什么安全由神经元控制的参数weights决定 

# Supervised Learning with Neural Networks

- 监督学习是众多深度学习中应用最广和赚钱效应最大的类别

  - 特点：特征值+目标值 = 必须的数据组合

- 监督学习的主要应用

  - ![](https://ws3.sinaimg.cn/large/006tNc79ly1g2cdicp18dj31260gyac7.jpg)
  - 监督学习的主要模型种类
    - 标准模型
    - CNN
    - RNN
    - 定制混合模型
  - 三种监督模型结构
    - 标准，CNN，RNN
    - ![](https://ws4.sinaimg.cn/large/006tNc79ly1g2cdkswybzj31nq0li432.jpg)
  - 结构化数据，非结构化数据
    - ![](https://ws1.sinaimg.cn/large/006tNc79ly1g2cdle7kdnj31n40re0z1.jpg)

  	- 结构化数据
  	  - 可以放进数据库或者Excel
  	- 非结构化数据
  	  - 不能直接放进数据库或者Excel
  	  - 音频、图片、文字等
  	- 人类天生擅长处理不规则数据，近2年深度学习帮助机器胜任不规则数据的处理

# Why is Deep Learning taking off?

- 数据、计算能力、算法之间的关系
  - ![](https://ws3.sinaimg.cn/large/006tNc79ly1g2ce35ju8jj31ks0towjc.jpg)
  - 当数据量比较小的时候，传统机器学习模型如果恰当使用手工特征值处理，效果会跟深度学习效果进行持平，有时还会更好。
  - 当数据量巨大，神经网络的层数多，神经元多的模型，表现效果越好，远优于传统机器学习模型
  - 神经网络的大小会受制于计算能力的大小
- 算法的用途
  - ![](https://ws1.sinaimg.cn/large/006tNc79ly1g2ce8uwdu3j31jm0nw77h.jpg)
  - sigmoid vs RELU
    - RELU让SGD算法提升运算能力，大幅缩短运算时间
    - sigmoid会导致梯度爆炸和梯度消失，让深度学习的参数更新，运行很慢
- 强大计算能力的必要性
  - 工作循环：从想法到验证，从完善想法到再验证
  - 强大的计算能力，可以缩短工作循环时间