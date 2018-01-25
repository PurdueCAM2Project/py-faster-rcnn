#!/bin/bash

prefix="./validation/coco_results/coco_2014_val/"
#prefix="./"
fn="coco_ap_plot.dat"

files=($(ls $prefix | grep coco | grep results ))
echo "Number of files: ${#files[@]}"

declare -a person_ap_p
declare -a iters

echo "iters" "ap" > $fn

for i in `seq 1 "${#files[@]}"`
do

    file=${files[$i]}
    if [ "$file" == "" ]
    then
	break
    fi
    
    person_ap_p[$i]=$(head -n2 $prefix$file | tail -n1)


    iter=$( echo $prefix$file | rev | cut -d'_' -f2 | rev )
    is_350=$( echo $file | grep 350k)
    is_700=$( echo $file | grep 700k)

    if [ "$is_350" != "" ]
    then
	iter=$(( iter + 350000 ))
    fi

    if [ "$is_700" != "" ]
    then
	iter=$(( iter + 700000 ))
    fi

    iters[$i]=$(( iter ))	

    echo "${iters[$i]}" "${person_ap_p[$i]}" >> $fn
done
	    
#echo "Plotting with coco_ap_plot.p"

#gnuplot "plot_coco.p"

exit 0


