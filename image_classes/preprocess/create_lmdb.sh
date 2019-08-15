#!/usr/bin/env sh

DATA=/home2/workplce/github/image_classes/data
CAFFE_DIR=/home2/caffe

echo "Create train lmdb..."
rm -rf $DATA/train_lmdb
$CAFFE_DIR/build/tools/convert_imageset \
--shuffle \
--resize_height=256 \
--resize_width=256 \
/ \
$DATA/label/train.txt \
$DATA/train_lmdb

echo "Create test lmdb..."
rm -rf $DATA/test_lmdb
$CAFFE_DIR/build/tools/convert_imageset \
--shuffle \
--resize_width=256 \
--resize_height=256 \
/ \
$DATA/label/test.txt \
$DATA/test_lmdb

echo "All Done..."

