#!/bin/bash


files=($(ls output/faster_rcnn_end2end/imagenet_train/imagenet_train_*))
for file in "${files[@]}"
do
    fn=$(echo $file | rev | cut -d'/' -f1 | cut -d'.' -f2 | rev)
    fn=$fn"_results.txt"
    echo ${file:0}
    ./test_faster_rcnn.sh 0 VGG16 caltech $file
    wait $!
    mv "results_imagenet.txt" $fn
done
	    

exit 0
