---
layout:     post
title:     	DeepLearning.ai-0401-卷积神经网络
subtitle:   2019-08-10 笔记
date:       2019-08-10
author:     koko
header-img: img/post-bg-universe.jpg
catalog: true
catalog: true
tags:

- 深度学习
- DeepLearning.ai
---

[TOC]

## 什么是图像分类，目标检测，大致的介绍

使用传统神经网络处理机器视觉的一个主要问题是输入层维度很大。例如一张64x64x3的图片，神经网络输入层的维度为12288。如果图片尺寸较大，例如一张1000x1000x3的图片，神经网络输入层的维度将达到3百万，使得网络权重W非常庞大。这样会造成两个后果，一是神经网络结构复杂，数据量相对不够，容易出现过拟合；二是所需内存、计算量较大。解决这一问题的方法就是使用卷积神经网络（CNN）。

## 垂直边缘检测

![image-20190806141041405](http://blogpicturekoko.oss-cn-beijing.aliyuncs.com/blog/2019-08-14-064612.jpg)



## 其他边缘检测器

- 图片边缘有两种渐变方式，一种是由明变暗，另一种是由暗变明。

以垂直边缘检测为例，下图展示了两种方式的区别：

![image-20190806141216093](http://blogpicturekoko.oss-cn-beijing.aliyuncs.com/blog/2019-08-14-064600.jpg)



实际应用中，这两种渐变方式并不影响边缘检测结果，可以对输出图片取绝对值操作，得到同样的结果。

- 垂直边缘检测算子与水平边缘检测算子

![image-20190806141324335](http://blogpicturekoko.oss-cn-beijing.aliyuncs.com/blog/2019-08-14-064606.jpg)

- 水平边缘检测的例子

![image-20190806141415655](http://blogpicturekoko.oss-cn-beijing.aliyuncs.com/blog/2019-08-14-64603.jpg)

**传统的边缘检测算子还有很多其他的，稍后可以总结。**

在深度学习中，如果我们想检测图片的各种边缘特征，而不仅限于垂直边缘和水平边缘，那么filter的数值一般需要**通过模型训练**得到，类似于标准神经网络中的权重W一样由梯度下降算法反复迭代求得。CNN的主要目的就是计算出这些filter的数值。确定得到了这些filter后，CNN浅层网络也就实现了对图片所有边缘特征的检测。



## padding

按照我们上面讲的图片卷积，如果原始图片尺寸为n x n，filter尺寸为f x f，则卷积后的图片尺寸为(n-f+1) x (n-f+1)，注意f一般为奇数。这样会带来两个问题：

**卷积运算后，输出图片尺寸缩小**

- **卷积运算后，输出图片尺寸缩小**做不了几次卷积，图像就会变得非常小（如果是很深层次的网络的话，比如100层，这样图像就会变得非常小）

- **原始图片边缘信息对输出贡献得少，输出图片丢失边缘信息。**图像角落的像素只会被使用一次，这会使图像丢失很多边界信息。

为了解决图片缩小的问题，可以使用padding方法，即把原始图片尺寸进行扩展，扩展区域补零，用p来表示每个方向扩展的宽度。

![image-20190806141757771](http://blogpicturekoko.oss-cn-beijing.aliyuncs.com/blog/2019-08-14-064610.jpg)

经过padding之后，原始图片尺寸为(n+2p) x (n+2p)，filter尺寸为f x f，则卷积后的图片尺寸为(n+2p-f+1) x (n+2p-f+1)。若要保证卷积前后图片尺寸不变，则p应满足：
$$
p=\frac{f-1}{2}
$$
没有padding操作，p=0，我们称之为“Valid convolutions”；

有padding操作，$$p=\frac{f-1}{2}$$，我们称之为“Same convolutions”。

如果滤波器的边界为奇数，可以通过选择填充边界大小，使得输出图像等于输入图像。

**滤波器大小基本为奇数的原因：**

1、如果为偶数，需要一些不对等的填充。只有f为奇数的时候，same卷积才会产生。

2、奇数的滤波器会有一个中心点。有中心像素点，就方便描述滤波器经过的位置。



## strided convolution

![image-20190806142148260](http://blogpicturekoko.oss-cn-beijing.aliyuncs.com/blog/2019-08-14-064607.jpg)

我们用s表示stride长度，p表示padding长度，如果原始图片尺寸为n x n，filter尺寸为f x f，则卷积后的图片尺寸为：
$$
\left\lfloor\frac{n+2 p-f}{s}+1\right\rfloor X\left\lfloor\frac{n+2 p-f}{s}+1\right\rfloor
$$
如果是非整数，就要进行**向下取整**，它表示最接近该数并小于该数的整数。

**计算举例：**

![image-20190806143915867](http://blogpicturekoko.oss-cn-beijing.aliyuncs.com/blog/2019-08-14-064608.jpg)

值得一提的是，相关系数（cross-correlations）与卷积（convolutions）之间是有区别的。实际上，真正的卷积运算会先将filter绕其中心旋转180度，然后再将旋转后的filter在原始图片上进行滑动计算。filter旋转如下所示：

![image-20190806142325542](http://blogpicturekoko.oss-cn-beijing.aliyuncs.com/blog/2019-08-14-64611.jpg)



比较而言，相关系数的计算过程则不会对filter进行旋转，而是直接在原始图片上进行滑动计算。

其实，**目前为止我们介绍的CNN卷积实际上计算的是相关系数**，而不是数学意义上的卷积。但是，为了简化计算，我们一般把CNN中的这种“相关系数”就称作卷积运算。之所以可以这么等效，是因为滤波器算子一般是水平或垂直对称的，180度旋转影响不大；而且最终滤波器算子需要通过CNN网络梯度下降算法计算得到，旋转部分可以看作是包含在CNN模型算法中。总的来说，忽略旋转运算可以大大提高CNN网络运算速度，而且不影响模型性能。

卷积运算服从结合律：
$$
(A * B) * C=A *(B * C)
$$
在机器学习的卷积操作过程中，通常忽略掉翻转的操作。这样的卷积实质上是交叉相关。只是大部分机器学习文献都叫它卷积操作。



## convolutions on RGB images

3通道图片的卷积运算与单通道图片的卷积运算基本一致。过程是将每个单通道（R，G，B）与对应的filter进行卷积运算求和，然后再将3通道的和相加，得到输出图片的一个像素值：

![image-20190806142723053](http://blogpicturekoko.oss-cn-beijing.aliyuncs.com/blog/2019-08-14-064605.jpg)

**不同通道的滤波算子可以不相同。**例如R通道filter实现垂直边缘检测，G和B通道不进行边缘检测，全部置零，或者将R，G，B三通道filter全部设置为水平边缘检测。

为了进行多个卷积运算，实现更多边缘检测，**可以增加更多的滤波器组**。例如设置第一个滤波器组实现垂直边缘检测，第二个滤波器组实现水平边缘检测。这样，不同滤波器组卷积得到不同的输出，个数由滤波器组决定。

![image-20190806142818100](http://blogpicturekoko.oss-cn-beijing.aliyuncs.com/blog/2019-08-14-64608.jpg)

## One Layer of a Convolutional Network

![image-20190806144126979](http://blogpicturekoko.oss-cn-beijing.aliyuncs.com/blog/2019-08-14-064602.jpg)

相比之前的卷积过程，CNN的单层结构多了激活函数ReLU和偏移量b。整个过程与标准的神经网络单层结构非常类似：
$$
\begin{aligned} Z^{[l]} &=W^{[l]} A^{[l-1]}+b \\ A^{[l]} &=g^{[l]}\left(Z^{[l]}\right) \end{aligned}
$$
卷积运算对应着上式中的乘积运算，滤波器组数值对应着权重$$W^{[l]}$$，所选的激活函数为ReLU。

我们来计算一下上图中参数的数目：每个滤波器组有3x3x3=27个参数，还有1个偏移量b，则每个滤波器组有27+1=28个参数，两个滤波器组总共包含28×2=56个参数。我们发现，选定滤波器组后，参数数目与输入图片尺寸无关。所以，就不存在由于图片尺寸过大，造成参数过多的情况。例如一张1000x1000x3的图片，标准神经网络输入层的维度将达到3百万，而在CNN中，参数数目只由滤波器组决定，数目相对来说要少得多，这是CNN的优势之一。

**参数数目只由滤波器组决定的原因：**

论输入图片有多大，1000×1000也好，5000×5000也好，参数始终都是280个。用这10个过滤器来提取特征，如垂直边缘，水平边缘和其它特征。即使这些图片很大，参数却很少，这就是卷积神经网络的一个特征，叫作“**避免过拟合**”。

最后，我们总结一下CNN单层结构的所有标记符号，设层数为$$l$$。
$$
\begin{array}{l}{f^{[l]}=\text { filter size }} \\ {p^{[l]}=\text { padding }} \\ {\boldsymbol{s}^{[l]}=\text { stride }} \\ {\boldsymbol{n}_{c}^{[l]}=\text { number of filters }}\end{array}
$$
![image-20190806144342411](http://blogpicturekoko.oss-cn-beijing.aliyuncs.com/blog/2019-08-14-064604.jpg)

其中：
$$
\begin{aligned} \boldsymbol{n}_{H}^{[l]} &=\left\lfloor\frac{n_{H}^{[l-1]}+2 p^{[l]}-f^{[l]}}{s^{[l]}}+1\right\rfloor \\ \boldsymbol{n}_{W}^{[l]} &=\left\lfloor\frac{n_{W}^{[l-1]}+2 p^{[l]}-f^{[l]}}{s^{l} ]}+1\right\rfloor \end{aligned}
$$
如果有m个样本，进行向量化运算，相应的输出维度为：$$m \times n_{H}^{[l]} \times n_{W}^{[l]} \times n_{c}^{[l]}$$。

## Simple Convolutional Network Example

**简单CNN模型：**

![image-20190806144910166](http://blogpicturekoko.oss-cn-beijing.aliyuncs.com/blog/2019-08-14-064611.jpg)

该CNN模型各层结构如上图所示。需要注意的是，$$a^{[3]}$$的维度是7 x 7 x 40，将$$a^{[3]}$$排列成1列，维度为1960 x 1，然后连接最后一级输出层。输出层可以是一个神经元，即二元分类（logistic）；也可以是多个神经元，即多元分类（softmax）。最后得到预测输出$$\hat y$$。

值得一提的是，随着CNN层数增加，$$n_H^{[l]}$$和$$n_W^{[l]}$$一般逐渐减小，而$$n_c^{[l]}$$一般逐渐增大。

CNN有三种类型的layer：

- Convolution层（CONV）最常见也最重要
- Pooling层（POOL）
- Fully connected层（FC）

## Pooling Layers

Pooling layers是CNN中用来减小尺寸，提高运算速度的，同样能减小噪声影响，让各特征更具有健壮性。

Pooling layers的做法比convolution layers简单许多，没有卷积运算，仅仅是在滤波器算子滑动区域内取最大值，即max pooling，这是最常用的做法。注意，超参数p很少在pooling layers中使用。

![image-20190806145422719](http://blogpicturekoko.oss-cn-beijing.aliyuncs.com/blog/2019-08-14-064609.jpg)

Max pooling的好处是只保留区域内的最大值（特征），忽略其它值，降低noise影响，提高模型健壮性。而且，max pooling需要的**超参数仅为滤波器尺寸f和滤波器步进长度s**，**没有其他参数需要模型训练得到**，计算量很小。

如果是多个通道，那么就每个通道单独进行max pooling操作。

除了max pooling之外，还有一种做法：average pooling。顾名思义，average pooling就是在滤波器算子滑动区域计算平均值。

![image-20190806145457759](http://blogpicturekoko.oss-cn-beijing.aliyuncs.com/blog/2019-08-14-064601.jpg)

实际应用中，max pooling比average pooling更为常用。

## CNN Examples

简单的数字识别CNN的例子：

![image-20190806151920441](http://blogpicturekoko.oss-cn-beijing.aliyuncs.com/blog/2019-08-14-064603.jpg)

图中，CON层后面紧接一个POOL层，CONV1和POOL1构成第一层，CONV2和POOL2构成第二层。特别注意的是FC3和FC4为全连接层FC，它跟标准的神经网络结构一致。最后的输出层（softmax）由10个神经元构成。

整个网络各层的尺寸和参数如下表格所示：

![image-20190806152224856](http://blogpicturekoko.oss-cn-beijing.aliyuncs.com/blog/2019-08-14-64605.jpg)

## Why Convolutions

相比标准神经网络，CNN的优势之一就是参数数目要少得多。参数数目少的原因有两个：

**todo:具体原因还要查阅下资料**

- 参数共享：一个特征检测器（例如垂直边缘检测）对图片某块区域有用，同时也可能作用在图片其它区域。

- 连接的稀疏性：因为滤波器算子尺寸限制，每一层的每个输出只与输入部分区域内有关。

除此之外，由于CNN参数数目较小，所需的训练样本就相对较少，从而一定程度上不容易发生过拟合现象。而且，CNN比较擅长捕捉区域位置偏移。也就是说CNN进行物体检测时，不太受物体所处图片位置的影响，增加检测的准确性和系统的健壮性。