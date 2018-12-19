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
	if [ -a "$i/fpga/bin/program.aocx" ]; then
		sha1sum $i/fpga/bin/program.aocx
	else
		echo "---"
	fi
done