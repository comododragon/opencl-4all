#!/bin/bash

PROJECTS=(
	"hotspot"
	"kmeans"
	"lavamd"
	"nn"
	"nw1"
	"nw2"
	"pathfinder"
	"srad"
	"backprop1"
	"backprop2"
	"lud1"
	"lud2"
	"lud3"
	"leukocyte1"
	"leukocyte2"
	"hybridsort1"
	"hybridsort2"
	"hybridsort3"
	"hotspot3D"
	"cfd"
	"bptree"
	"particlefilter1"
	"particlefilter2"
	"streamcluster"
	"bfs"
	"fft"
	"gemm"
	"md"
	"md5hash"
	"reduction"
	"spmv"
	"stencil2d"
	"scan"
	"ndrsd1"
	"ndrsd2"
	"ndrsd3"
	"ndrsd4"
	"ndrsdfull"
)

EXECTIMES=10
TIMEGPU="$(pwd)/gpu.csv"

echo "Initialising csv files..."
echo -n "kernel" > $TIMEGPU
for i in `seq 1 $EXECTIMES`; do
	echo -n ",total$(($i-1))" >> $TIMEGPU
	NASTRING="$NASTRING,---"
done
for i in `seq 1 $EXECTIMES`; do
	echo -n ",periter$(($i-1))" >> $TIMEGPU
	NASTRING="$NASTRING,---"
done
echo "" >> $TIMEGPU

echo "Running GPU projects..."
for i in ${PROJECTS[@]}; do
	if [ -d $i ]; then
		echo -e "\tRunning: $i"
		if [ -a $i/gpu/execute ]; then
			cd $i
			cd gpu
			./execute &> out.log
			echo -e "\t\tIteration: 0"
			FULLTIMES="$(grep "Elapsed time" out.log | sed "s/Elapsed time spent on kernels: \\(.\\+\\) us;.*/\\1/g")"
			ITERTIMES="$(grep "Elapsed time" out.log | sed "s/.*Average time per iteration: \\(.\\+\\) us./\\1/g")"
			for j in `seq 2 $EXECTIMES`; do
				echo -e "\t\tIteration: $(($j-1))"
				./execute &> out.log
				FULLTIMES="$FULLTIMES,$(grep "Elapsed time" out.log | sed "s/Elapsed time spent on kernels: \\(.\\+\\) us;.*/\\1/g")"
				ITERTIMES="$ITERTIMES,$(grep "Elapsed time" out.log | sed "s/.*Average time per iteration: \\(.\\+\\) us./\\1/g")"
			done
			cd ..
			cd ..
			echo "$i,$FULLTIMES,$ITERTIMES" >> $TIMEGPU
		else
			echo -e "\t\tProject is not compiled"
			echo "$i$NASTRING" >> $TIMEGPU
		fi
	else
		echo -e "\tMissing: $i"
		echo "$i$NASTRING" >> $TIMEGPU
	fi
done
