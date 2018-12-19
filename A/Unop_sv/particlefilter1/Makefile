GENERALFLAGS=-fPIC -DCOMMON_COLOURED_PRINTS -Iinclude -lm
AOCLFLAGS=`aocl compile-config` `aocl link-config`
GPUFLAGS=-L/usr/lib64/nvidia -lOpenCL

fpga/emu/emulate: src/host.fpga.c include/common.h include/prepostambles.h fpga/emu/program.aocx
	cd fpga/emu; ln -sf ../../aux/outputArrayX
	cd fpga/emu; ln -sf ../../aux/outputArrayY
	cd fpga/emu; ln -sf ../../aux/inputXj
	cd fpga/emu; ln -sf ../../aux/inputYj
	cd fpga/emu; ln -sf ../../aux/outputInd
	cd fpga/emu; ln -sf ../../aux/inputObjxy
	cd fpga/emu; ln -sf ../../aux/outputLikelihood
	cd fpga/emu; ln -sf ../../aux/inputI
	cd fpga/emu; ln -sf ../../aux/outputWeights
	cd fpga/emu; ln -sf ../../aux/inputSeed
	cd fpga/emu; ln -sf ../../aux/outputPartialSums
	$(CC) src/host.fpga.c -g -o fpga/emu/emulate $(GENERALFLAGS) $(AOCLFLAGS)

fpga/emu/program.aocx: src/kern.cl
	mkdir -p fpga/emu
	aoc -v -march=emulator -Iinclude -g --board s5phq_a7 src/kern.cl -o fpga/emu/program.aocx

fpga/bin/execute: src/host.fpga.c include/common.h include/prepostambles.h fpga/bin/program.aocx
	cd fpga/bin; ln -sf ../../aux/outputArrayX
	cd fpga/bin; ln -sf ../../aux/outputArrayY
	cd fpga/bin; ln -sf ../../aux/inputXj
	cd fpga/bin; ln -sf ../../aux/inputYj
	cd fpga/bin; ln -sf ../../aux/outputInd
	cd fpga/bin; ln -sf ../../aux/inputObjxy
	cd fpga/bin; ln -sf ../../aux/outputLikelihood
	cd fpga/bin; ln -sf ../../aux/inputI
	cd fpga/bin; ln -sf ../../aux/outputWeights
	cd fpga/bin; ln -sf ../../aux/inputSeed
	cd fpga/bin; ln -sf ../../aux/outputPartialSums
	$(CC) src/host.fpga.c -o fpga/bin/execute $(GENERALFLAGS) $(AOCLFLAGS)

fpga/bin/program.aocx: src/kern.cl
	mkdir -p fpga/bin
	aoc -v -Iinclude --board s5phq_a7 src/kern.cl -o fpga/bin/program.aocx

gpu/execute: src/host.gpu.c include/common.h include/prepostambles.h src/kern.cl
	mkdir -p gpu
	cd gpu; ln -sf ../aux/outputArrayX
	cd gpu; ln -sf ../aux/outputArrayY
	cd gpu; ln -sf ../aux/inputXj
	cd gpu; ln -sf ../aux/inputYj
	cd gpu; ln -sf ../aux/outputInd
	cd gpu; ln -sf ../aux/inputObjxy
	cd gpu; ln -sf ../aux/outputLikelihood
	cd gpu; ln -sf ../aux/inputI
	cd gpu; ln -sf ../aux/outputWeights
	cd gpu; ln -sf ../aux/inputSeed
	cd gpu; ln -sf ../aux/outputPartialSums
	cd gpu; ln -sf ../src/kern.cl
	$(CC) src/host.gpu.c -o gpu/execute $(GENERALFLAGS) $(GPUFLAGS)

.PHONY: clean
clean:
	rm -rf fpga gpu
