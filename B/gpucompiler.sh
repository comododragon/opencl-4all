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

echo "Compiling GPU projects..."
for i in ${PROJECTS[@]}; do
	echo -e "\tCompiling: $i"
	cd $i
	make gpu/execute
	cd ..
done
