#!/bin/bash

# ON KENT's PC
# ./experiments/scripts/test_faster_rcnn.sh 0 VGG16 pascal_voc ~/Documents/ML/models/py-faster-rcnn/output/downloaded_pascal/VGG16_faster_rcnn_final.caffemodel

# CAM2 2017
#./experiments/scripts/test_faster_rcnn.sh 0 VGG16 pascal_voc ~/Documents/CAM2/image_team/faster_rcnn/frcnn_caffe/py-faster-rcnn/output/gpu0_end2end_vgg16_12_25_2016/voc_2007_trainval/vgg16_faster_rcnn_iter_70000.caffemodel

#./tools/reval.py --imdb cam2_2017_trainval output/faster_rcnn_end2end/cam2_2017_trainval/vgg16_faster_rcnn_iter_70000/

#./experiments/scripts/test_faster_rcnn.sh 0 VGG16 cam2 ~/Documents/CAM2/image_team/faster_rcnn/frcnn_caffe/py-faster-rcnn/output/gpu0_end2end_vgg16_12_25_2016/voc_2007_trainval/vgg16_faster_rcnn_iter_70000.caffemodel

# PASCAL VOC 2007
#./experiments/scripts/test_faster_rcnn.sh 0 VGG16 pascal_voc ~/Documents/CAM2/image_team/faster_rcnn/frcnn_caffe/py-faster-rcnn/output/gpu0_end2end_vgg16_12_25_2016/voc_2007_trainval/vgg16_faster_rcnn_iter_70000.caffemodel


set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3
NET_FINAL=$4
CORG_DIR=$5
VIS_DIR=$6

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:6:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}


case $DATASET in
    imagenet)
	TRAIN_IMDB="imagenet_train"
	#TEST_IMDB="imagenet_test"
	#TEST_IMDB="imagenet_very_short_train"
	TEST_IMDB="imagenet_val1"
	PT_DIR="imagenet"
	ITERS=100000
	;;
    pascal_voc)
	TRAIN_IMDB="voc_2007_trainval"
	TEST_IMDB="voc_2007_test"
	PT_DIR="pascal_voc"
	ITERS=70000
	;;
    pascal_voc_2012)
	TRAIN_IMDB="voc_2012_trainval"
	TEST_IMDB="voc_2012_val"
	PT_DIR="pascal_voc"
	ITERS=70000
	;;
    coco)
	# This is a very long and slow training schedule
	# You can probably use fewer iterations and reduce the
	# time to the LR drop (set in the solver to 350,000 iterations).
	TRAIN_IMDB="coco_2014_train"
	TEST_IMDB="coco_2014_val"
	PT_DIR="coco"
	ITERS=490000
	;;
    cam2)
	# this is cam2 data :-)
	TRAIN_IMDB="cam2_2017_trainval"
	#TEST_IMDB="cam2_2017_test" #"cam2_2017_trainval"
	TEST_IMDB="cam2_2017_all" #"cam2_2017_trainval"
	PT_DIR="cam2"
	ITERS=10000
	;;
    sun)
	TRAIN_IMDB="sun_2012_train"
	#TEST_IMDB="sun_2012_taste"
	TEST_IMDB="sun_2012_test"
	PT_DIR="sun"
	ITERS=10000
	;;
    caltech)
	TRAIN_IMDB="caltech_2009_train"
	TEST_IMDB="caltech_2009_test"
	PT_DIR="caltech"
	ITERS=10000
	;;
    kitti)
	TRAIN_IMDB="kitti_2013_train"
	TEST_IMDB="kitti_2013_val"
	PT_DIR="kitti"
	ITERS=70000
	;;
    *)
	echo "No dataset given"
	exit
	;;
esac

# original command -- set here for testing with associated prototxt to caffemodel
# ./tools/test_net.py --gpu ${GPU_ID} \
# 		    --vis ${VIS_DIR} \
#   --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
#   --net ${NET_FINAL} \
#   --imdb ${TEST_IMDB} \
#   --cfg experiments/cfgs/faster_rcnn_end2end.yml \
#   ${EXTRA_ARGS}

# CORG says: "The original model is trained on the loaded corg file."

if [ "${CORG_DIR}" != "" ]; then
    echo "corg"
    DEF=models/${CORG_DIR}/${NET}/faster_rcnn_end2end/test_corg.prototxt
else
    echo "NOT corg"
    if [ "${DATASET}" == "pascal_voc_2012" ]; then
	DATASET="pascal_voc"
    fi
    DEF=models/${DATASET}/${NET}/faster_rcnn_end2end/test.prototxt
fi

./tools/test_net.py --gpu ${GPU_ID} \
		    --vis ${VIS_DIR} \
  --def ${DEF} \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}
  #--def models/imagenet/${NET}/faster_rcnn_end2end/test.prototxt \

