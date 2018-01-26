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
	    
# WRITTEN AND USED [01/25/2018]
# run caltech dataset "reval.py" since the evaluation of detections failed on the intial running of the code. see git commit eb8c5f for caltech fix
# for the moment change to manage the non-running issue...
# files=($(ls output/faster_rcnn_end2end/caltech_2009_test/))
# for file in "${files[@]}"
# do
#     fn=$(echo $file | rev | cut -d'/' -f1 | cut -d'.' -f2 | rev)
#     fn=$fn"_results.txt"
#     echo ${file:0}
#     ./tools/reval.py --imdb caltech_2009_test "output/faster_rcnn_end2end/caltech_2009_test/"$file
#     wait $!
#     mv "results_caltech.txt" $fn
# done


exit 0
