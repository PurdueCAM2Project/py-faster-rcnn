#!/bin/bash


files=($(ls output/faster_rcnn_end2end/kitti_2013_train/kitti3_*caffemodel))
for file in "${files[@]}"
do
    fn=$(echo $file | rev | cut -d'/' -f1 | cut -d'.' -f2 | rev)
    fn=$fn"_results.txt"
    echo ${file:0}
    ./test_faster_rcnn.sh 0 VGG16 kitti $file
    wait $!
    mv "results_kitti.txt" $fn
done
	    

exit 0
