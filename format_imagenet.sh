#!/bin/bash

fn="imagenet_ap_plot.dat"

files=($(ls ./ | grep imagenet | grep results | grep -v "10k"))
echo "Number of files: ${#files[@]}"

declare -a person_ap
declare -a iters

echo "iters" "ap@50" "ap@75" "ap@95" > $fn

for i in `seq 1 "${#files[@]}"`
do

    file=${files[$i]}
    if [ "$file" == "" ]
    then
	break
    fi
    
    person_ap_p50[$i]=$(cat $file | grep person | cut -d':' -f2 | cut -d'	' -f2 | tr -d ' ')
    person_ap_p75[$i]=$(cat $file | grep person | cut -d':' -f2 | cut -d'	' -f3 | tr -d ' ')
    person_ap_p95[$i]=$(cat $file | grep person | cut -d':' -f2 | cut -d'	' -f4 | tr -d ' ')

    add_10k=$(echo $file | grep 10k)
    iter=$( echo $file | rev | cut -d'_' -f2 | rev )
    if [ "$add_10k" != "" ]
    then
	iters[$i]=$(( 100000 + iter ))
    else	
	iters[$i]=$(( iter ))	
    fi
    echo ${iters[$i]} "${person_ap_p50[$i]}" "${person_ap_p75[$i]}" "${person_ap_p95[$i]}" >> $fn
done
	    
#echo "Plotting with imagenet_ap_plot.p"

#gnuplot "plot_imagenet.p"

exit 0


