#encoding:utf-8
import sys,os
caffe_root = '/home2/caffe/'
sys.path.append(caffe_root + 'python')
import numpy as np
import caffe
import cv2
import time
caffe.set_device(0)
caffe.set_mode_gpu()
time_begin = time.time()
# 设置当前的工作环境在caffe下, 根据自己实际情况更改
# 我们也把caffe/python也添加到当前环境
os.chdir(caffe_root)#更换工作目录

# 设置网络结构
net_file='/home2/workplce/github/caffe_img_cls/caffenet/deploy.prototxt'
# 添加训练之后的参数
caffe_model='/home2/workplce/github/caffe_img_cls/caffenet/models/caffenet_train_iter_500.caffemodel'
'''
#这是一个由mean.binaryproto文件生成mean.npy文件的函数
def convert_mean(binMean,npyMean):
    blob = caffe.proto.caffe_pb2.BlobProto()
    bin_mean = open(binMean, 'rb' ).read()
    blob.ParseFromString(bin_mean)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    npy_mean = arr[0]
    np.save(npyMean, npy_mean )
binMean='../data/mean.binaryproto' #修改成你的mean.binaryproto文件的路径
# 均值文件
mean_file='../data/mean.npy' #你想把生成的mean.npy文件放在哪个路径下
convert_mean(binMean,mean_file)
'''
# 这里对任何一个程序都是通用的，就是处理图片
# 把上面添加的两个变量都作为参数构造一个Net
net = caffe.Net(net_file,caffe_model,caffe.TEST)
# 得到data的形状，这里的图片是默认matplotlib底层加载的
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# matplotlib加载的image是像素[0-1],图片的数据格式[h,w,c]，RGB
# caffe加载的图片需要的是[0-255]像素，数据格式[c,h,w],BGR，那么就需要转换

# channel 放到前面
transformer.set_transpose('data', (2,0,1))
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1)) #如果你在训练模型的时候没有对输入做mean操作，那么这边也不需要
# 图片像素放大到[0-255]
transformer.set_raw_scale('data', 255)
# RGB-->BGR 转换
transformer.set_channel_swap('data', (2,1,0))

# 加载一张测试图片
image_file = '/home2/workplce/github/caffe_img_cls/data/image/test/711.jpg'
im=caffe.io.load_image(image_file)
# 用上面的transformer.preprocess来处理刚刚加载图片
net.blobs['data'].data[...] = transformer.preprocess('data',im)
#网络开始向前传播啦
output = net.forward()
# 最终的结果: 当前这个图片的属于哪个物体的概率(列表表示)
#print('output:',len(output['prob'][0]))
output_prob = output['prob'][0]
# 找出最大的那个概率
chars = ["大巴", "恐龙", "大象", "花朵", "马"]

label = chars[output_prob.argmax()]
score = round(np.max(output_prob),3)

print '\n测试图片: {}'.format(image_file)
print '\n预测结果:\n类别 {}； 得分 {}'.format(label, score)
print '\ntime used: ', round(time.time()-time_begin, 4), 's'
img = cv2.imread(image_file)
cv2.imwrite('./result.jpg',img)
#cv2.imshow('img',img)
cv2.putText(img, 'label', (30,30), 1, 2, (0,255,255)) 
#cv2.waitKey(0)
