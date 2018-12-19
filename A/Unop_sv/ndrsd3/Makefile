GENERALFLAGS=-fPIC -DCOMMON_COLOURED_PRINTS -Iinclude -lm
AOCLFLAGS=`aocl compile-config` `aocl link-config`
GPUFLAGS=-L/usr/lib64/nvidia -lOpenCL

fpga/emu/emulate: src/host.fpga.c include/common.h include/prepostambles.h fpga/emu/program.aocx
	cd fpga/emu; ln -sf ../../aux/inputLambda
	cd fpga/emu; ln -sf ../../aux/inputOmega
	cd fpga/emu; ln -sf ../../aux/inputErrCnt
	cd fpga/emu; ln -sf ../../aux/inputErrLoc
	cd fpga/emu; ln -sf ../../aux/inputAlphaInv
	cd fpga/emu; ln -sf ../../aux/outputErrOut
	$(CC) src/host.fpga.c -g -o fpga/emu/emulate $(GENERALFLAGS) $(AOCLFLAGS)

fpga/emu/program.aocx: src/kern.cl
	mkdir -p fpga/emu
	aoc -v -march=emulator -Iinclude -g --board s5phq_a7 src/kern.cl -o fpga/emu/program.aocx

fpga/bin/execute: src/host.fpga.c include/common.h include/prepostambles.h fpga/bin/program.aocx
	cd fpga/bin; ln -sf ../../aux/inputLambda
	cd fpga/bin; ln -sf ../../aux/inputOmega
	cd fpga/bin; ln -sf ../../aux/inputErrCnt
	cd fpga/bin; ln -sf ../../aux/inputErrLoc
	cd fpga/bin; ln -sf ../../aux/inputAlphaInv
	cd fpga/bin; ln -sf ../../aux/outputErrOut
	$(CC) src/host.fpga.c -o fpga/bin/execute $(GENERALFLAGS) $(AOCLFLAGS)

fpga/bin/program.aocx: src/kern.cl
	mkdir -p fpga/bin
	aoc -v -Iinclude --board s5phq_a7 src/kern.cl -o fpga/bin/program.aocx

gpu/execute: src/host.gpu.c include/common.h include/prepostambles.h src/kern.cl include/constants.h include/gfa.h
	mkdir -p gpu
	cd gpu; ln -sf ../aux/inputLambda
	cd gpu; ln -sf ../aux/inputOmega
	cd gpu; ln -sf ../aux/inputErrCnt
	cd gpu; ln -sf ../aux/inputErrLoc
	cd gpu; ln -sf ../aux/inputAlphaInv
	cd gpu; ln -sf ../aux/outputErrOut
	cd gpu; ln -sf ../src/kern.cl
	cd gpu; ln -sf ../include/constants.h
	cd gpu; ln -sf ../include/gfa.h
	$(CC) src/host.gpu.c -o gpu/execute $(GENERALFLAGS) $(GPUFLAGS)

.PHONY: clean
clean:
	rm -rf fpga gpu
