#!/usr/bin/env python
#encoding:utf-8
import numpy as np
import sys, caffe

if len(sys.argv)!=3:
    print("Usage: python convert_mean.py mean.binaryproto mean.npy")
    sys.exit()

blob=caffe.proto.caffe_pb2.BlobProto()
bin_mean=open(sys.argv[1], 'rb').read()
blob.ParseFromString(bin_mean)
arr=np.array(caffe.io.blobproto_to_array(blob))
npy_mean=arr[0]
np.save(sys.argv[2], npy_mean)
len_shape = npy_mean.shape[0]
print('image shape: {}\n'.format(npy_mean.shape))
for i in range(len_shape):
    print('channle {} mean: {}\n'.format(i+1, np.mean(npy_mean[i])))
