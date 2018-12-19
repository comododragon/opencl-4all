#!/bin/bash

PROJECTS=(
	"nw"
	"hotspot"
	"hotspot3D"
	"pathfinder"
)

for i in ${PROJECTS[@]}; do
	if [ -a "$i/fpga/bin/program.aocx" ]; then
		sha1sum $i/fpga/bin/program.aocx
	else
		echo "---"
	fi
done