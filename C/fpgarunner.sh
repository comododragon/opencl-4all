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
	"nw"
	"hotspot"
	"hotspot3D"
	"pathfinder"
)

EXECTIMES=10
TIMEFPGA="$(pwd)/fpga.csv"

echo "Initialising csv files..."
echo -n "kernel" > $TIMEFPGA
for i in `seq 1 $EXECTIMES`; do
	echo -n ",total$(($i-1))" >> $TIMEFPGA
	NASTRING="$NASTRING,---"
done
for i in `seq 1 $EXECTIMES`; do
	echo -n ",periter$(($i-1))" >> $TIMEFPGA
	NASTRING="$NASTRING,---"
done
echo "" >> $TIMEFPGA

echo "Running FPGA projects..."
for i in ${PROJECTS[@]}; do
	if [ -d $i ]; then
		echo -e "\tRunning: $i"
		if [ -a $i/fpga/bin/execute ]; then
			cd $i
			cd fpga/bin
			./execute &> out.log
			echo -e "\t\tIteration: 0"
			FULLTIMES="$(grep "Elapsed time" out.log | sed "s/Elapsed time spent on kernels: \\(.\\+\\) us;.*/\\1/g")"
			ITERTIMES="$(grep "Elapsed time" out.log | sed "s/.*Average time per iteration: \\(.\\+\\) us./\\1/g")"
			for j in `seq 2 $EXECTIMES`; do
				echo -e "\t\tIteration: $(($j-1))"
				./execute &> out.log
				FULLTIMES="$FULLTIMES,$(grep "Elapsed time" out.log | sed "s/Elapsed time spent on kernels: \\(.\\+\\) us;.*/\\1/g")"
				ITERTIMES="$ITERTIMES,$(grep "Elapsed time" out.log | sed "s/.*Average time per iteration: \\(.\\+\\) us./\\1/g")"
			done
			cd ../..
			cd ..
			echo "$i,$FULLTIMES,$ITERTIMES" >> $TIMEFPGA
		else
			echo -e "\t\tProject is not compiled"
			echo "$i$NASTRING" >> $TIMEFPGA
		fi
	else
		echo -e "\tMissing: $i"
		echo "$i$NASTRING" >> $TIMEFPGA
	fi
done
