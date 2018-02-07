#!/bin/bash

# coco_2014_val
files=($(ls output/faster_rcnn_end2end/coco_2014_train/coco* ))
#completed_files=($(ls ./coco* ))
counter=0
for file in "${files[@]}"
do
    fn=$(echo $file | rev | cut -d'/' -f1 | cut -d'.' -f2 | rev)

    # for cfile in "${completed_files[@]}"
    # do
	
    # done

    if [ $counter -gt 71 ]
    then
	fn=$fn"_results.txt"
	echo $fn
	./test_faster_rcnn.sh 0 VGG16 coco $file
	wait $!
	mv "results_coco.txt" $fn
    fi   
    counter=$(( $counter + 1 ))

done
	    

exit 0
