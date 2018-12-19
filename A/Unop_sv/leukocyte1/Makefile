GENERALFLAGS=-fPIC -DCOMMON_COLOURED_PRINTS -Iinclude -lm
AOCLFLAGS=`aocl compile-config` `aocl link-config`
GPUFLAGS=-L/usr/lib64/nvidia -lOpenCL

fpga/emu/emulate: src/host.fpga.c include/common.h include/prepostambles.h fpga/emu/program.aocx
	cd fpga/emu; ln -sf ../../aux/inputGradX
	cd fpga/emu; ln -sf ../../aux/inputGradY
	cd fpga/emu; ln -sf ../../aux/inputCSinAngle
	cd fpga/emu; ln -sf ../../aux/inputCCosAngle
	cd fpga/emu; ln -sf ../../aux/inputCTX
	cd fpga/emu; ln -sf ../../aux/inputCTY
	cd fpga/emu; ln -sf ../../aux/outputGicov
	$(CC) src/host.fpga.c -g -o fpga/emu/emulate $(GENERALFLAGS) $(AOCLFLAGS)

fpga/emu/program.aocx: src/kern.cl
	mkdir -p fpga/emu
	aoc -v -march=emulator -Iinclude -g --board s5phq_a7 src/kern.cl -o fpga/emu/program.aocx

fpga/bin/execute: src/host.fpga.c include/common.h include/prepostambles.h fpga/bin/program.aocx
	cd fpga/bin; ln -sf ../../aux/inputGradX
	cd fpga/bin; ln -sf ../../aux/inputGradY
	cd fpga/bin; ln -sf ../../aux/inputCSinAngle
	cd fpga/bin; ln -sf ../../aux/inputCCosAngle
	cd fpga/bin; ln -sf ../../aux/inputCTX
	cd fpga/bin; ln -sf ../../aux/inputCTY
	cd fpga/bin; ln -sf ../../aux/outputGicov
	$(CC) src/host.fpga.c -o fpga/bin/execute $(GENERALFLAGS) $(AOCLFLAGS)

fpga/bin/program.aocx: src/kern.cl
	mkdir -p fpga/bin
	aoc -v -Iinclude --board s5phq_a7 src/kern.cl -o fpga/bin/program.aocx

gpu/execute: src/host.gpu.c include/common.h include/prepostambles.h src/kern.cl
	mkdir -p gpu
	cd gpu; ln -sf ../aux/inputGradX
	cd gpu; ln -sf ../aux/inputGradY
	cd gpu; ln -sf ../aux/inputCSinAngle
	cd gpu; ln -sf ../aux/inputCCosAngle
	cd gpu; ln -sf ../aux/inputCTX
	cd gpu; ln -sf ../aux/inputCTY
	cd gpu; ln -sf ../aux/outputGicov
	cd gpu; ln -sf ../src/kern.cl
	$(CC) src/host.gpu.c -o gpu/execute $(GENERALFLAGS) $(GPUFLAGS)

.PHONY: clean
clean:
	rm -rf fpga gpu
