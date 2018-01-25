#!/usr/bin/gnuplot

# Gnuplot script file for plotting map data in file "coco_ap_plot.dat"
#set term png
#set output 'coco_ap_plot.png'
set   autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "Average Precision for People Class v.s. Iterations (Higher is better)"
set xlabel "Iterations"
set ylabel "log(AP for People)"
set key at 100000,0.01
#set xr [0.0:0.022]
#set yr [0:325]
#set logscale y
plot    '< sort -n -k1 "coco_ap_plot.dat"' using 1:2 title "ap" with linespoints

