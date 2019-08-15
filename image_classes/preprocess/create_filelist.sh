#!/usr/bin/env sh

DATA=/home2/workplce/github/caffe_img_cls/data
IMAGE=$DATA/image
LABEL=$DATA/label

echo "Create train.txt..."
rm -rf $LABEL/train.txt
for i in 3 4 5 6 7
do
j=`expr $i - 3`
find $IMAGE/train -name $i*.jpg | sed "s/$/ $j/">>$LABEL/train.txt
done

echo "Create test.txt..."
rm -rf $LABEL/test.txt
for i in 3 4 5 6 7
do 
j=`expr $i - 3`
find $IMAGE/test -name $i*.jpg | sed "s/$/ $j/">>$LABEL/test.txt
done

echo "All done"
