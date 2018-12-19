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

PROJECTTYPES=(
	"Unop_sv"
)

# Description
echo "=Description="
echo "1 - Unop sv"
echo "2 - gpu"
echo "============="

# Check for program.aocx
echo "=program.aocx="
echo "| 1 | 2 |"
for i in ${PROJECTS[@]}; do
	if [ -a "Unop_sv/$i/fpga/bin/program.aocx" ]; then
		echo -n "|[x]"
	else
		echo -n "|[ ]"
	fi
	echo -n "|[-]"
	echo "|$i"
done
echo "=============="

# Check for execute
echo "===execute==="
echo "| 1 | 2 |"
for i in ${PROJECTS[@]}; do
	if [ -a "Unop_sv/$i/fpga/bin/execute" ]; then
		echo -n "|[x]"
	else
		echo -n "|[ ]"
	fi
	if [ -a "Unop_sv/$i/gpu/execute" ]; then
		echo -n "|[x]"
	else
		echo -n "|[ ]"
	fi
	echo "|$i"
done
echo "============="

# Check for project
echo "===top.qpf==="
echo "| 1 | 2 |"
for i in ${PROJECTS[@]}; do
	if [ -a "Unop_sv/$i/fpga/bin/program/top.qpf" ]; then
		echo -n "|[x]"
	else
		echo -n "|[ ]"
	fi
	echo -n "|[-]"
	echo "|$i"
done
echo "============="
