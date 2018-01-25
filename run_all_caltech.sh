#!/bin/bash


files=($(ls output/faster_rcnn_end2end/caltech_2009_train/caltech_*))
for file in "${files[@]}"
do
    fn=$(echo $file | rev | cut -d'/' -f1 | cut -d'.' -f2 | rev)
    fn=$fn"_results.txt"
    echo ${file:0}
    ./test_faster_rcnn.sh 0 VGG16 caltech $file
    wait $!
    mv "results_caltech.txt" $fn
done
	    

exit 0
