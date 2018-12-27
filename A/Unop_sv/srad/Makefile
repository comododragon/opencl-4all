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

GENERALFLAGS=-fPIC -DCOMMON_COLOURED_PRINTS -Iinclude -lm
AOCLFLAGS=`aocl compile-config` `aocl link-config`
GPUFLAGS=-L/usr/lib64/nvidia -lOpenCL

fpga/emu/emulate: src/host.fpga.c include/common.h include/prepostambles.h fpga/emu/program.aocx
	cd fpga/emu; ln -sf ../../aux/inI .
	cd fpga/emu; ln -sf ../../aux/inQ0sqr .
	$(CC) src/host.fpga.c -g -o fpga/emu/emulate $(GENERALFLAGS) $(AOCLFLAGS)

fpga/emu/program.aocx: src/kern.cl
	mkdir -p fpga/emu
	aoc -v -march=emulator -g --board s5phq_a7 src/kern.cl -o fpga/emu/program.aocx

fpga/bin/execute: src/host.fpga.c include/common.h include/prepostambles.h fpga/bin/program.aocx
	cd fpga/bin; ln -sf ../../aux/inI .
	cd fpga/bin; ln -sf ../../aux/inQ0sqr .
	$(CC) src/host.fpga.c -o fpga/bin/execute $(GENERALFLAGS) $(AOCLFLAGS)

fpga/bin/program.aocx: src/kern.cl
	mkdir -p fpga/bin
	aoc -v --board s5phq_a7 src/kern.cl -o fpga/bin/program.aocx

gpu/execute: src/host.gpu.c include/common.h include/prepostambles.h src/kern.cl
	mkdir -p gpu
	cd gpu; ln -sf ../aux/inI .
	cd gpu; ln -sf ../aux/inQ0sqr .
	cd gpu; ln -sf ../src/kern.cl
	$(CC) src/host.gpu.c -o gpu/execute $(GENERALFLAGS) $(GPUFLAGS)

.PHONY: clean
clean:
	rm -rf fpga gpu
