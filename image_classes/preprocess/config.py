#encoding:utf -8

# resize
resize_w = 227
resize_h = 227
# 网络结构路径
#net_file='/home2/workplce/github/image_classes/caffenet/deploy.prototxt'
net_file='/home2/workplce/github/image_classes/vggnet16/deploy.prototxt'

# caffemodel路径
caffe_model='/home2/workplce/github/image_classes/vggnet16/models/caffenet_train_iter_16000.caffemodel'
#caffe_model='/home2/workplce/github/image_classes/caffenet/models/caffenet_train_iter_4000.caffemodel'

# 二进制格式均值文件
binMean='/home2/workplce/github/image_classes/data/mean.binaryproto'

# Python格式均值文件（测试用）
mean_file='/home2/workplce/github/image_classes/data/mean.npy' #自定义路径，运行代码新生成的文件

# 测试图片路径
test_img_path = '/home2/workplce/github/image_classes/caffenet/test_img/'

# 测试图片保存路径
img_save_path = '/home2/workplce/github/image_classes/preprocess/'

