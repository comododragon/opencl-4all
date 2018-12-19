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

for i in ${PROJECTS[@]}; do
	if [ -d $i ]; then
		echo -n "$i: "
		if [ -a $i/fpga/bin/program/acl_quartus_report.txt ]; then
			grep "Kernel fmax:" $i/fpga/bin/program/acl_quartus_report.txt | sed "s/Kernel fmax: //g"
		else
			echo "---"
		fi
	else
		echo -e "$i: missing folder"
	fi
done
