#!/bin/bash

PROJECTS=(
	"nw"
	"hotspot"
	"hotspot3D"
	"pathfinder"
)

PROJECTTYPES=(
	"Opts_sv/3_FullTask"
)

# Description
echo "=Description="
echo "1 - Opts sv: full task"
echo "2 - gpu"
echo "============="

# Check for program.aocx
echo "=program.aocx="
echo "| 1 | 2 |"
for i in ${PROJECTS[@]}; do
	if [ -a "Opts_sv/3_FullTask/$i/fpga/bin/program.aocx" ]; then
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
	if [ -a "Opts_sv/3_FullTask/$i/fpga/bin/execute" ]; then
		echo -n "|[x]"
	else
		echo -n "|[ ]"
	fi
	if [ -a "Opts_sv/3_FullTask/$i/gpu/execute" ]; then
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
	if [ -a "Opts_sv/3_FullTask/$i/fpga/bin/program/top.qpf" ]; then
		echo -n "|[x]"
	else
		echo -n "|[ ]"
	fi
	echo -n "|[-]"
	echo "|$i"
done
echo "============="
