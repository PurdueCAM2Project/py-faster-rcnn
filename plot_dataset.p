#!/usr/bin/gnuplot --persist

# Gnuplot script file for plotting map data in "filename" argument
#set term png
FILENAME=ARG1
OUTPUT=ARG2
print FILENAME
print OUTPUT
set output OUTPUT
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
plot    '< sort -n -k1 '.FILENAME using 1:2 title "ap@50" with linespoints,\
   '< sort -n -k1 '.FILENAME using 1:3 title "ap@75" with linespoints, \
      '< sort -n -k1 '.FILENAME using 1:4 title "ap@95" with linespoints
      


