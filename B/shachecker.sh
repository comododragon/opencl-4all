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
	if [ -a "$i/fpga/bin/program.aocx" ]; then
		sha1sum $i/fpga/bin/program.aocx
	else
		echo "---"
	fi
done