# caffe实现图片分类

# 数据介绍
下载地址：https://pan.baidu.com/s/1D8gW8dnLK-Nm5VOjLsKAeg

数据集共500张，分为5类，分别是：大巴，恐龙，大象，花朵，马

训练集：400，测试集：100

原始数据在data/image/目录下，分为train和test；3**.jpg表示大巴，4**.jpg表示恐龙，5**.jpg表示大象，6**.jpg表示花朵，7**.jpg表示马，标签分别为0,1,2,3,4

# 目录说明
data/： 存放图片数据image/，数据标签label/，train_lmdb文件，test_lmdb文件

caffenet/：存放prototxt文件及模型

preprocess/：存放数据处理脚本及测试代码

# 数据处理
## 1.执行 sh preprocess/create_filelist.sh 
 
 作用是在data/label/目录下生成训练集和测试集的图片和标签列表

 需注意的是修改脚本里自己的DATA路径

## 2.执行 sh preprocess/create_lmdb.sh

 在data/下生成train_lmdb、test_lmdb，注意修改自己的DATA、CAFFE_DIR路径

## 3.生成均值文件

在data/下 执行 compute_image_mean ./train_lmdb ./mean.binaryproto

## 4.训练 
 在caffenet/路径下执行：

 caffe train -solver solver.prototxt 

 注意train_val.prototxt中data_param下数据路径修改为自己的

## 5.测试

 Python preprocess/pycaffe_test.py

 注意修改自己的caffe根目录，模型路径，测试图片路径
