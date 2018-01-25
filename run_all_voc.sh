#!/bin/bash

# voc_2007_trainval and voc_2012_trainval
files=($(ls output/faster_rcnn_end2end/voc_2007_trainval/*caffemodel ) $(ls output/faster_rcnn_end2end/voc_2012_trainval/*caffemodel ) )
counter=0
for file in "${files[@]}"
do
    fn=$(echo $file | rev | cut -d'/' -f1 | cut -d'.' -f2 | rev)


    if [ $counter -gt -1 ]
    then
	fn=$fn"_results.txt"
	echo $fn
	./test_faster_rcnn.sh 0 VGG16 pascal_voc $file
	wait $!
	mv "results_voc.txt" $fn
    fi   
    counter=$(( $counter + 1 ))

done
	    

exit 0
