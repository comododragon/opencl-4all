#!/bin/bash

PROJECTS=(
	"nw"
	"hotspot"
	"hotspot3D"
	"pathfinder"
)

echo "Compiling GPU projects..."
for i in ${PROJECTS[@]}; do
	echo -e "\tCompiling: $i"
	cd $i
	make gpu/execute
	cd ..
done
