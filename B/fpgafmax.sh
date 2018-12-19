#!/bin/bash

PROJECTS=(
	"poisson1"
	"poisson2"
	"bitonic1"
	"bitonic2"
	"bitonic3"
	"nw1"
	"nw2"
	"hotspot"
	"hotspot3D"
	"pathfinder"
	"srad"
	"lud1"
	"lud2"
	"lud3"
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
