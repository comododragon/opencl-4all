#!/bin/bash

PROJECTS=(
	"nw"
	"hotspot"
	"hotspot3D"
	"pathfinder"
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
