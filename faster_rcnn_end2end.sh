#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3
PRETRAINED=$4
SS=$5

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
    imagenet)
	TRAIN_IMDB="imagenet_train"
	TEST_IMDB="imagenet_test"
	PT_DIR="imagenet"
	ITERS=600000
	;;
    pascal_voc)
	TRAIN_IMDB="voc_2007_trainval"
	TEST_IMDB="voc_2007_test"
	PT_DIR="pascal_voc"
	ITERS=50000
	;;
    pascal_voc_2012)
	TRAIN_IMDB="voc_2012_trainval"
	TEST_IMDB="voc_2012_test"
	PT_DIR="pascal_voc"
	ITERS=70000
	;;
    coco)
	# This is a very long and slow training schedule
	# You can probably use fewer iterations and reduce the
	# time to the LR drop (set in the solver to 350,000 iterations).
	TRAIN_IMDB="coco_2014_train"
	TEST_IMDB="coco_2015_test"
	PT_DIR="coco"
	ITERS=350000
	;;
    cam2)
	# this is cam2 data :-)
	TRAIN_IMDB="cam2_2017_trainval"
	TEST_IMDB="cam2_2017_test"
	PT_DIR="cam2"
	ITERS=10000
	;;
    sun)
	TRAIN_IMDB="sun_2012_train"
	#TEST_IMDB="sun_2012_taste"
	TEST_IMDB="sun_2012_test"
	PT_DIR="sun"
	ITERS=50000
	;;
    caltech)
	TRAIN_IMDB="caltech_2009_train"
	TEST_IMDB="caltech_2009_test"
	PT_DIR="caltech"
	ITERS=350000
	;;
    kitti)
	TRAIN_IMDB="kitti_2013_train"
	TEST_IMDB="kitti_2013_val"
	PT_DIR="kitti"
	#ITERS=150000
	ITERS=200000
	;;
    *)
	echo "No dataset given"
	exit
	;;
esac

LOG="experiments/logs/faster_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
echo ${EXTRA_ARGS}

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/faster_rcnn_end2end/solver.prototxt \
  --weights ${PRETRAINED} \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  # --solver_state ${SS} \
  # ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}
