GENERALFLAGS=-fPIC -DCOMMON_COLOURED_PRINTS -Iinclude -lm
AOCLFLAGS=`aocl compile-config` `aocl link-config`
GPUFLAGS=-L/usr/lib64/nvidia -lOpenCL

fpga/emu/emulate: src/host.fpga.c include/common.h include/prepostambles.h fpga/emu/program.aocx
	cd fpga/emu; ln -sf ../../aux/inputS
	cd fpga/emu; ln -sf ../../aux/outputC
	cd fpga/emu; ln -sf ../../aux/outputW
	$(CC) src/host.fpga.c -g -o fpga/emu/emulate $(GENERALFLAGS) $(AOCLFLAGS)

fpga/emu/program.aocx: src/kern.cl
	mkdir -p fpga/emu
	aoc -v -march=emulator -Iinclude -g --board s5phq_a7 src/kern.cl -o fpga/emu/program.aocx

fpga/bin/execute: src/host.fpga.c include/common.h include/prepostambles.h fpga/bin/program.aocx
	cd fpga/bin; ln -sf ../../aux/inputS
	cd fpga/bin; ln -sf ../../aux/outputC
	cd fpga/bin; ln -sf ../../aux/outputW
	$(CC) src/host.fpga.c -o fpga/bin/execute $(GENERALFLAGS) $(AOCLFLAGS)

fpga/bin/program.aocx: src/kern.cl
	mkdir -p fpga/bin
	aoc -v -Iinclude --board s5phq_a7 src/kern.cl -o fpga/bin/program.aocx

gpu/execute: src/host.gpu.c include/common.h include/prepostambles.h src/kern.cl include/constants.h include/gfa.h
	mkdir -p gpu
	cd gpu; ln -sf ../aux/inputS
	cd gpu; ln -sf ../aux/outputC
	cd gpu; ln -sf ../aux/outputW
	cd gpu; ln -sf ../src/kern.cl
	cd gpu; ln -sf ../include/constants.h
	cd gpu; ln -sf ../include/gfa.h
	$(CC) src/host.gpu.c -o gpu/execute $(GENERALFLAGS) $(GPUFLAGS)

.PHONY: clean
clean:
	rm -rf fpga gpu
