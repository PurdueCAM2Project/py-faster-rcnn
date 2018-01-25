#!/bin/bash

# sun_2012_train
files=($(ls output/faster_rcnn_end2end/sun_2012_train/*caffemodel ))
counter=0
for file in "${files[@]}"
do
    fn=$(echo $file | rev | cut -d'/' -f1 | cut -d'.' -f2 | rev)


    if [ $counter -gt -1 ]
    then
	fn=$fn"_results.txt"
	echo $fn
	./test_faster_rcnn.sh 0 VGG16 sun $file
	wait $!
	mv "results_sun.txt" $fn
    fi   
    counter=$(( $counter + 1 ))

done
	    

exit 0
