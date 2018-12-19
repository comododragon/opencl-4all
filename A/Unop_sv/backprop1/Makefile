GENERALFLAGS=-fPIC -DCOMMON_COLOURED_PRINTS -Iinclude -lm
AOCLFLAGS=`aocl compile-config` `aocl link-config`
GPUFLAGS=-L/usr/lib64/nvidia -lOpenCL

fpga/emu/emulate: src/host.fpga.c include/common.h include/prepostambles.h fpga/emu/program.aocx include/bpnn.h src/bpnn.c
	cd fpga/emu; ln -sf ../../aux/outputHiddenPartialSum
	$(CC) src/host.fpga.c src/bpnn.c -g -o fpga/emu/emulate $(GENERALFLAGS) $(AOCLFLAGS)

fpga/emu/program.aocx: src/kern.cl
	mkdir -p fpga/emu
	aoc -v -march=emulator -Iinclude -g --board s5phq_a7 src/kern.cl -o fpga/emu/program.aocx

fpga/bin/execute: src/host.fpga.c include/common.h include/prepostambles.h fpga/bin/program.aocx include/bpnn.h src/bpnn.c
	cd fpga/bin; ln -sf ../../aux/outputHiddenPartialSum
	$(CC) src/host.fpga.c src/bpnn.c -o fpga/bin/execute $(GENERALFLAGS) $(AOCLFLAGS)

fpga/bin/program.aocx: src/kern.cl
	mkdir -p fpga/bin
	aoc -v -Iinclude --board s5phq_a7 src/kern.cl -o fpga/bin/program.aocx

gpu/execute: src/host.gpu.c include/common.h include/prepostambles.h src/kern.cl include/bpnn.h src/bpnn.c
	mkdir -p gpu
	cd gpu; ln -sf ../aux/outputHiddenPartialSum
	cd gpu; ln -sf ../src/kern.cl
	cd gpu; ln -sf ../include/bpnn.h
	cd gpu; ln -sf ../src/bpnn.c
	$(CC) src/host.gpu.c src/bpnn.c -o gpu/execute $(GENERALFLAGS) $(GPUFLAGS)

.PHONY: clean
clean:
	rm -rf fpga gpu
