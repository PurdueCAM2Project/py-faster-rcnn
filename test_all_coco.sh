#!/bin/bash

#601,602,603

def_path="./models/coco/VGG16/faster_rcnn_end2end/test_corg.prototxt"
model_path="./output/faster_rcnn_end2end/coco_2014_train/coco_700k_train_iter_10000.caffemodel"

./test_faster_rcnn.sh 0 VGG16 imagenet $model_path coco
wait $!
mv results_imagenet.txt validation_coco_on_imagenet.txt

./test_faster_rcnn.sh 0 VGG16 coco $model_path
wait $!
mv results_coco.txt validation_coco_on_coco.txt

# -=-=-=-=-
sed '603s/^/# /' $def_path
sed '602s/#//' $def_path
# -=-=-=-=-
./test_faster_rcnn.sh 0 VGG16 pascal_voc_2012 $model_path coco
wait $!
mv results_voc.txt validation_coco_on_voc.txt

# -=-=-=-=-
sed '602s/^/# /' $def_path
sed '601s/#//' $def_path
# -=-=-=-=-
./test_faster_rcnn.sh 0 VGG16 sun $model_path coco
wait $!
mv results_sun.txt validation_coco_on_sun.txt
./test_faster_rcnn.sh 0 VGG16 caltech $model_path coco
wait $!
mv results_caltech.txt validation_coco_on_caltech.txt
./test_faster_rcnn.sh 0 VGG16 cam2 $model_path coco
wait $!
mv results_cam2.txt validation_coco_on_cam2.txt
./test_faster_rcnn.sh 0 VGG16 inria $model_path coco
wait $!

echo "likely death soon..."

mv results_inria.txt validation_coco_on_inria.txt
./test_faster_rcnn.sh 0 VGG16 kitti $model_path coco
wait $!
mv results_kitti.txt validation_coco_on_kitti.txt

exit 0
