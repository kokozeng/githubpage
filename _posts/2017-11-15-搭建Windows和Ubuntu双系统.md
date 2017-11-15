---
layout:     post
title:      搭建Windows和Ubuntu双系统
subtitle:   Windows7和Ubuntu双系统
date:       2017-11-15
author:     koko
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - Ubuntu
    - Windows
---

# 搭建Window和Ubuntu双系统

最近很多小伙伴问我怎么搭建双系统，回答的多了觉得还是写个详细的教程解答比较方便一点，也节约时间。这篇教程本意是写给小白看的，我会尽量的详细一些。
先说几个问题。

>问：Ubuntu系统是不是必须的？

答：Ubuntu不是必须的，但是作为一名程序员还是很建议使用这个系统。很多开源框架是基于Linux，很利于学习。用Ubuntu进行软件开发也是一件很优雅的事情。

>问：装了双系统之后电脑会不会卡呀？我Windows是32位的能不能装64位的Ubuntu呢？

答：一般来说，你的PC能够运行Windows不卡顿的话，那么Ubuntu比Windows更加轻量级，更加不会卡的。至于32位的能不能装...这不是在Windows上安装一个软件，这两个系统分别在你硬盘的不同位置，它们是相对独立的关系。所以...基本上跟你的Windows没有任何关系的。

现在开始安装双系统之旅啦~其实很简单的，安装系统真的一点都不难。好啦，废话不多说，进入正题。

## 一、磁盘分区
如果是从Windows安装Ubuntu，我们需要在Windows下给Ubuntu腾出一小块地方---给Ubuntu分配一些硬盘空间。可以进入这个链接查看[完整教程](https://jingyan.baidu.com/article/4b07be3c79863648b380f314.html)。

1.点击我的电脑，点击鼠标右键，选择管理项。

2.点击磁盘管理



![](http://img.blog.csdn.net/20171114210819603?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva29rb3plbmcxOTk1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

3.接下来，选择你需要分割的盘。点击鼠标右键，选择压缩卷。（这一步是为了腾出可分割空间用的）

4.输入需要腾出给Ubuntu系统的空间大小，可进行换算。(1G=1024M)例：如果你要增添一个10G的盘，在压缩这一步输入10240即可。点击压缩。

5.压缩完毕后，你会发现多了一块绿盘，这一部分空间就是新盘区域了。到这里其实就可以了，可以不用做新建简单卷的操作。


## 二、下载Ubuntu镜像文件

去[Ubuntu官方下载地址](http://cn.ubuntu.com/download/)下载你需要安装的Ubuntu的版本。Ubuntu是一个开源系统，可以免费下载、使用并分享。

## 三、制作U盘启动器

你需要一个格式化的U盘作为一个启动器，将下载的镜像文件写入U盘。我们还需要一个写入文件的软件，一般使用软碟通UltralISO[点击这里下载](http://cn.ezbsystems.com/ultraiso/)。
参考[使用软碟通制作U盘启动器教程](https://jingyan.baidu.com/article/a378c960630e61b329283045.html)，将下好的软件用软碟通写入，此时U盘启动器就做好了。

## 四、进入BIOS

进入BIOS，用U盘启动电脑。一般PC都是在开机时按F2键或者F12键。


![BIOS](http://img.blog.csdn.net/20161209205409537?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQnlDaGVuNjIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

## 五、安装Ubuntu

### 首先选择语言

强烈建议使用英文。


![这里写图片描述](http://img.blog.csdn.net/20161209205420506?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQnlDaGVuNjIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

可以选择第二项也可以不选，看需求。

![这里写图片描述](http://img.blog.csdn.net/20161209205430714?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQnlDaGVuNjIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

### 这里注意选择others


![这里写图片描述](http://img.blog.csdn.net/20161209205441308?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQnlDaGVuNjIz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

接下来进行Ubuntu系统分区。

### 第一次分区：

“空闲”处点“+”，进行如下设置：
挂载点：“/”
大小：至少5G,建议100G以下分配一半空间给主分区,100G以上至少给50G.
新分区的类型：主分区
新分区的位置：空间起始位置
用于：EXT4日志文件系统

### 第二次分区：

“空闲”处，继续点“+”，如下设置，
挂载点：（不设置）
大小：2048MB
新分区的类型：逻辑分区
新分区的位置：空间起始位置
用于：交换空间

### 第三次分区：

“空闲”处，继续点“+”，如下设置，
挂载点：/boot  （网上有的说不需要设置这项，但双系统引导时需要，先不要去理解这些）
大小：200MB
新分区的类型：逻辑分区
新分区的位置：空间起始位置
用于：EXT4日志文件系统

### 第四次分区：

“空闲”处，继续点“+”，如下设置，
挂载点：/home
大小：剩余全部空间，剩下显示多少，就多少
新分区的类型：逻辑分区
新分区的位置：空间起始位置
用于：EXT4日志文件系统

### 引导位置
如果需要从Windows引导，那么就选择/boot分区所在的盘符。但是这样装好Ubuntu之后，还需要在Windows下载EasyBCD将Ubuntu系统调出来。
直接从Ubuntu引导，选择`/dev/sda` 盘符。那么重新开机后，会出现一个选项，选择是进入Windows系统还是Ubuntu系统。

至此，Windows和Ubuntu双系统就安装完成了。希望这篇教程对大家有用~



