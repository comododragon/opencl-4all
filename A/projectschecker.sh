#!/bin/bash

# Copyright (c) 2018 Andre Bannwart Perina
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
