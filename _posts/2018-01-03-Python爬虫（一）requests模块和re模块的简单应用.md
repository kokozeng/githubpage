---
layout:     post
title:      Python爬虫（一）requests模块和re模块的简单应用
subtitle:   requests模块和re模块
date:       2017-01-03
author:     koko
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - 爬虫
---


# Python爬虫（一）requests模块和re模块的简单应用

## Requests模块

最近运用爬虫做了一些事情，写一些简单的博客做个简要的记录。这篇博客主要是介绍requests模块的运用和如何使用简易的正则表达式。

对于一般的HTML网页来说，requests就基本满足需求了。听说还有什么urlib，但是我没用过。做第一个爬虫的时候，lowlow的我并没有用什么高大上的框架，或者是特别酷炫的库。基本用到的就是这两个模块。

本文讲的两个模块的安装方法就不讲了，直接pip install就好了～网上教程也是一抓一大把的。


```python
import requests

def get_page_html(url):
    headers = {"Accept": "text/html,application/xhtml+xml,application/xml;",
               "Accept-Encoding": "gzip",
               "Accept-Language": "zh-CN,zh;q=0.8",
               "Referer": "http://www.example.com/",
               "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36"}
    r = requests.get(url, headers=headers)
    r.encoding = 'UTF-8'
    result = r.text
    return result
```



如上所示代码是写了一个获取网页的函数。headers是模拟浏览器进入，一些网页可能会有一些反爬机制。直接爬取会报403的错误，加上headers可以防止它的发生。

r.encoding是将网页按照指定格式编码。requests自带的函数内部，有编写一定的编码机制。但也存在一些时候不能够正确的识别所在网页的编码格式。自己check一下网页的编码格式，然后指定编码格式会减少出现乱码的情况。

介绍一种查看网页编码格式的方法，浏览器进入开发者模式，选择Console,在console窗口输入“document.charset”或“document.characterSet”查看。然后把该窗口出现的编码格式替换上述代码的UTF-8。



```python
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
```


在python中，编码解码其实是不同编码系统间的转换，默认情况下，转换目标是Unicode，即编码unicode→str，解码str→unicode，其中str指的是字节流，而str.decode是将字节流str按给定的解码方式解码，并转换成utf-8形式，u.encode是将unicode类按给定的编码方式转换成字节流str。注意调用encode方法的是unicode对象，生成的是字节流；调用decode方法的是str对象（字节流），生成的是unicode对象。若str对象调用encode会默认先按系统默认编码方式decode成unicode对象再encode，忽视了中间默认的decode往往导致报错。

因为大多数网页都是使用utf-8的编码方式，在开头加上这一段，可以防止出现将抓取下来的网页保存成txt时的报错。

## re模块

request模块很简单已经介绍完毕。简单情况下这样使用就足够了。如果你试了试上述代码，你会发现爬下来的HTML非常复杂，我们如何从中提取到我们需要的信息呢。

就以我的博客为例，教大家如何爬爬我的blog，哈哈。



```python
import re
url = 'https://kokozeng.github.io/'
page = get_page_html(url)
pattern = re.compile('<h2 class="post-title">(.*?)</h2>', re.S)
titles = re.findall(pattern, page)
for title in titles:
    print title
```

跑完以上代码，就可以出来我博客的所有标题了。是不是简单又神奇～上述的page是之前用requests模块获取的网页文件。
pattern是用正则表达式做一个匹配模板。

这里的正则表达式是用的.*?可以匹配任意字段。

re.findall是找寻网页文件中符合pattern的字段，网页中能够与之匹配的不只有一个title,也就是说我不只是只写了一篇博客（捂脸..　其实目前只有三篇（不禁鄙视一下自己的自律...
然后我们再用一个遍历，将匹配到的titles数组进行遍历输出。

学到这里你就可以去爬一些简单的网页了。：）有空的话，我也会补充一些实例。

