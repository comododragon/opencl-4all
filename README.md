# OpenCL-forró (opencl-4all)

Collection of OpenCL kernels for GPU/FPGA execution

## Author

André Bannwart Perina

## Introduction

This is an extensive set of OpenCL applications organised for execution and profiling on GPUs and FPGAs. Focused on analysing the FPGA performance and energy consumption when compared against GPUs, this repository is comprised of 3 experiments, each having a different level of OpenCL FPGA-biased optimisations. Comparison results are also available.

## Repository structure

This repository contains three root folders, one for each experiment:

* Experiment A: OpenCL kernels with no FPGA optimisations (NDRange);
* Experiment B: OpenCL kernels with FPGA optimisations but without change in execution model (NDRange);
* Experiment C: OpenCL kernels with FPGA optimisations, changing also execution model (task).

### Experiment structure

Each experiment has a description spreadsheet, several helper scripts, result files and OpenCL projects prepared for GPU compilation and FPGA synthesis.

#### Description spreadsheet

The experiment description spreadsheet (`description.ods`) contains information about the OpenCL applications including: application original source, data set, modifications and compilation report.

#### Helper scripts

There are several BASH helper scripts, useful for acquiring information or performing tasks automatically for all projects:

* Check which projects are compiled (including host executable and kernel object file);
* Collect operational frequency of all kernels (FPGA);
* Calculate checksum of all synthesised kernels (FPGA);
* Compile host executables (GPU);
* Run kernels.

To run the first script:
```
cd path/to/experiment
./projectschecker.sh
```

To run all other scripts:
```
cd path/to/experiment/OPTLEVEL
./script.sh
```

OPTLEVEL differs in each experiment:
* Experiment A: `Unop_sv`
* Experiment B: `Opts_sv/2_Full`
* Experiment C: `Opts_sv/3_FullTask`

## Licence

See LICENSE file.
