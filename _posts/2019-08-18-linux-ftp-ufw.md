---
layout:     post
title:     	「linux」 ftp 、ufw
subtitle:   2019-08-18 笔记
date:       2019-08-18
author:     koko
header-img: img/post-bg-universe.jpg
catalog: true
catalog: true
tags:

- Linux
---


# 「linux」 ftp 、ufw

## 阿里云服务器搭建ftp

### 准备工作

阿里云需要配置安全组：

![image-20190818175108889](http://blogpicturekoko.oss-cn-beijing.aliyuncs.com/blog/2019-08-18-101309.jpg)

```python
apt-get update    # 更新软件
apt-get install vsftpd     # 安装vsftpd
service vsftpd status # 查看vsftpd的状态
```

![image-20190818174235665](http://blogpicturekoko.oss-cn-beijing.aliyuncs.com/blog/2019-08-18-101308.jpg)

如果出问题，我们可以检查端口：

```python
netstat -anp
```

### 目录及用户

然后创建想访问的目录：

```python
mkdir /home/uftp     # 创建目录
chmod 775 /home      # 赋予同组用户读写权限
```

接着指定用户以及密码：

```python
useradd -m -d /home/uftp -s /bin/sh -g root uftp		# 添加用户
passwd uftp		# 设置密码
```

### 修改vsftpd配置文件

```python
vim /etc/vsftpd.conf             # 修改配置文件
anonymous_enable=NO/YES          # 是否允许匿名登录FTP服务器
local_enable=NO/YES              # 是否允许本地用户登录FTP服务器
listen=NO/YES                    # 设置vsftpd服务器是否以standalone模式运行
write_enable=YES/NO              # 是否允许登录用户有写的权限

## 如果限制用户只能访问主目录
chroot_local_user=YES
chroot_list_enable=YES
chroot_list_file=/etc/vsftpd.chroot_list		# 在该文件中列出的用户可以跳出主目录

service vsftpd restart		# 重启服务
```

### 登陆ftp

![image-20190818174924316](http://blogpicturekoko.oss-cn-beijing.aliyuncs.com/blog/2019-08-18-101306.jpg)

### 删除操作

```python
userdel uftp          # 删除用户
rm -rf /home/uftp     # 删除目录
```

## 防火墙

### 基本功能

```python
sudo apt-get install ufw         # 安装
sudo ufw enable/disable          # 启动/关闭 启动即关闭所有外部对本机的访问
sudo ufw default deny            # 系统启动时自动开启
sudo ufw status									 # 查看ufw状态
sudo ufw logging on|off          #	转换状态日志
```

### 使用举例

```python
sudo ufw allow smtp		# 允许所有的外部IP访问本机的25/tcp (smtp)端口
sudo ufw allow 22/tcp		# 允许所有的外部IP访问本机的22/tcp (ssh)端口
sudo ufw allow 53		# 允许外部访问53端口(tcp/udp)
sudo ufw allow from 192.168.1.100		# 允许此IP访问所有的本机端口
sudo ufw allow proto udp 192.168.0.1 port 53 to 192.168.0.2 port 53
sudo ufw deny smtp		# 禁止外部访问smtp服务
sudo ufw delete allow smtp		# 删除上面建立的某条规则
```

参考：

1、https://blog.csdn.net/hohaizx/article/details/78484540

2、https://www.cnblogs.com/sweet521/p/5733466.html

