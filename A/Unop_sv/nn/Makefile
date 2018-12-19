GENERALFLAGS=-fPIC -DCOMMON_COLOURED_PRINTS -Iinclude -lm
AOCLFLAGS=`aocl compile-config` `aocl link-config`
GPUFLAGS=-L/usr/lib64/nvidia -lOpenCL

fpga/emu/emulate: src/host.fpga.c src/list.c include/list.h include/common.h include/prepostambles.h fpga/emu/program.aocx
	cd fpga/emu; ln -sf ../../aux/filelist
	cd fpga/emu; ln -sf ../../aux/cane4_0.db
	cd fpga/emu; ln -sf ../../aux/cane4_1.db
	cd fpga/emu; ln -sf ../../aux/cane4_2.db
	cd fpga/emu; ln -sf ../../aux/cane4_3.db
	$(CC) src/host.fpga.c src/list.c -g -o fpga/emu/emulate $(GENERALFLAGS) $(AOCLFLAGS)

fpga/emu/program.aocx: src/kern.cl
	mkdir -p fpga/emu
	aoc -v -march=emulator -Iinclude -g --board s5phq_a7 src/kern.cl -o fpga/emu/program.aocx

fpga/bin/execute: src/host.fpga.c src/list.c include/list.h include/common.h include/prepostambles.h fpga/bin/program.aocx
	cd fpga/bin; ln -sf ../../aux/filelist
	cd fpga/bin; ln -sf ../../aux/cane4_0.db
	cd fpga/bin; ln -sf ../../aux/cane4_1.db
	cd fpga/bin; ln -sf ../../aux/cane4_2.db
	cd fpga/bin; ln -sf ../../aux/cane4_3.db
	$(CC) src/host.fpga.c src/list.c -o fpga/bin/execute $(GENERALFLAGS) $(AOCLFLAGS)

fpga/bin/program.aocx: src/kern.cl
	mkdir -p fpga/bin
	aoc -v -Iinclude --board s5phq_a7 src/kern.cl -o fpga/bin/program.aocx

gpu/execute: src/host.gpu.c src/list.c include/list.h include/common.h include/prepostambles.h src/kern.cl
	mkdir -p gpu
	cd gpu; ln -sf ../aux/filelist
	cd gpu; ln -sf ../aux/cane4_0.db
	cd gpu; ln -sf ../aux/cane4_1.db
	cd gpu; ln -sf ../aux/cane4_2.db
	cd gpu; ln -sf ../aux/cane4_3.db
	cd gpu; ln -sf ../src/kern.cl
	$(CC) src/host.gpu.c src/list.c -o gpu/execute $(GENERALFLAGS) $(GPUFLAGS)

.PHONY: clean
clean:
	rm -rf fpga gpu
