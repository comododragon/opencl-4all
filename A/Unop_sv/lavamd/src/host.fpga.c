/* ********************************************************************************************* */
/* * C Template for Kernel Execution                                                           * */
/* * Author: André Bannwart Perina                                                             * */
/* ********************************************************************************************* */
/* * Copyright (c) 2017 André B. Perina                                                        * */
/* *                                                                                           * */
/* * Permission is hereby granted, free of charge, to any person obtaining a copy of this      * */
/* * software and associated documentation files (the "Software"), to deal in the Software     * */
/* * without restriction, including without limitation the rights to use, copy, modify,        * */
/* * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to        * */
/* * permit persons to whom the Software is furnished to do so, subject to the following       * */
/* * conditions:                                                                               * */
/* *                                                                                           * */
/* * The above copyright notice and this permission notice shall be included in all copies     * */
/* * or substantial portions of the Software.                                                  * */
/* *                                                                                           * */
/* * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,       * */
/* * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR  * */
/* * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE * */
/* * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      * */
/* * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER    * */
/* * DEALINGS IN THE SOFTWARE.                                                                 * */
/* ********************************************************************************************* */

#include <CL/opencl.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include "common.h"

/**
 * @brief Header where pre/postamble macro functions should be located.
 *        Function headers:
 *            PREAMBLE(d_par_gpu_alpha, d_dim_gpu_number_boxes, d_box_gpu_offset, d_box_gpu_offsetSz, d_box_gpu_nn, d_box_gpu_nnSz, d_box_gpu_nei_number, d_box_gpu_nei_numberSz, d_rv_gpu, d_rv_gpuSz, d_qv_gpu, d_qv_gpuSz, d_fv_gpu, d_fv_gpuSz);
 *            POSTAMBLE(d_par_gpu_alpha, d_dim_gpu_number_boxes, d_box_gpu_offset, d_box_gpu_offsetSz, d_box_gpu_nn, d_box_gpu_nnSz, d_box_gpu_nei_number, d_box_gpu_nei_numberSz, d_rv_gpu, d_rv_gpuSz, d_qv_gpu, d_qv_gpuSz, d_fv_gpu, d_fv_gpuSz);
 *            LOOPPREAMBLE(d_par_gpu_alpha, d_dim_gpu_number_boxes, d_box_gpu_offset, d_box_gpu_offsetSz, d_box_gpu_nn, d_box_gpu_nnSz, d_box_gpu_nei_number, d_box_gpu_nei_numberSz, d_rv_gpu, d_rv_gpuSz, d_qv_gpu, d_qv_gpuSz, d_fv_gpu, d_fv_gpuSz, loopFlag);
 *            LOOPPOSTAMBLE(d_par_gpu_alpha, d_dim_gpu_number_boxes, d_box_gpu_offset, d_box_gpu_offsetSz, d_box_gpu_nn, d_box_gpu_nnSz, d_box_gpu_nei_number, d_box_gpu_nei_numberSz, d_rv_gpu, d_rv_gpuSz, d_qv_gpu, d_qv_gpuSz, d_fv_gpu, d_fv_gpuSz, loopFlag);
 *            CLEANUP(d_par_gpu_alpha, d_dim_gpu_number_boxes, d_box_gpu_offset, d_box_gpu_offsetSz, d_box_gpu_nn, d_box_gpu_nnSz, d_box_gpu_nei_number, d_box_gpu_nei_numberSz, d_rv_gpu, d_rv_gpuSz, d_qv_gpu, d_qv_gpuSz, d_fv_gpu, d_fv_gpuSz);
 *        where:
 *            d_par_gpu_alpha: variable (float);
 *            d_dim_gpu_number_boxes: variable (long);
 *            d_box_gpu_offset: variable (long *);
 *            d_box_gpu_offsetSz: number of members in variable (unsigned int);
 *            d_box_gpu_nn: variable (int *);
 *            d_box_gpu_nnSz: number of members in variable (unsigned int);
 *            d_box_gpu_nei_number: variable (int *);
 *            d_box_gpu_nei_numberSz: number of members in variable (unsigned int);
 *            d_rv_gpu: variable (cl_float4 *);
 *            d_rv_gpuSz: number of members in variable (unsigned int);
 *            d_qv_gpu: variable (float *);
 *            d_qv_gpuSz: number of members in variable (unsigned int);
 *            d_fv_gpu: variable (cl_float4 *);
 *            d_fv_gpuSz: number of members in variable (unsigned int);
 *            loopFlag: loop condition variable (bool).
 */
#include "prepostambles.h"

/**
 * @brief Test if two operands are outside an epsilon range.
 *
 * @param a First operand.
 * @param b Second operand.
 * @param e Epsilon value.
 */
#define TEST_EPSILON(a, b, e) (((a > b) && (a - b > e)) || ((b >= a) && (b - a > e)))

/**
 * @brief Standard statements for function error handling and printing.
 *
 * @param funcName Function name that failed.
 */
#define FUNCTION_ERROR_STATEMENTS(funcName) {\
	rv = EXIT_FAILURE;\
	PRINT_FAIL();\
	fprintf(stderr, "Error: %s failed with return code %d.\n", funcName, fRet);\
}

/**
 * @brief Standard statements for POSIX error handling and printing.
 *
 * @param arg Arbitrary string to the printed at the end of error string.
 */
#define POSIX_ERROR_STATEMENTS(arg) {\
	rv = EXIT_FAILURE;\
	PRINT_FAIL();\
	fprintf(stderr, "Error: %s: %s\n", strerror(errno), arg);\
}

int main(void) {
	/* Return variable */
	int rv = EXIT_SUCCESS;

	/* OpenCL and aux variables */
	int i = 0, j = 0;
	cl_int platformsLen, devicesLen, fRet;
	cl_platform_id *platforms = NULL;
	cl_device_id *devices = NULL;
	cl_context context = NULL;
	cl_command_queue queueKernel_Gpu_Opencl = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelKernel_Gpu_Opencl = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimKernel_Gpu_Opencl = 1;
	size_t globalSizeKernel_Gpu_Opencl[1] = {
		128000
	};
	size_t localSizeKernel_Gpu_Opencl[1] = {
		128
	};

	/* Input/output variables */
	float d_par_gpu_alpha;
	long d_dim_gpu_number_boxes = 1000;
	long *d_box_gpu_offset = malloc(1000 * sizeof(long));
	cl_mem d_box_gpu_offsetK = NULL;
	int *d_box_gpu_nn = malloc(1000 * sizeof(int));
	cl_mem d_box_gpu_nnK = NULL;
	int *d_box_gpu_nei_number = malloc(26000 * sizeof(int));
	cl_mem d_box_gpu_nei_numberK = NULL;
	cl_float4 *d_rv_gpu = malloc(100000 * sizeof(cl_float4));
	cl_mem d_rv_gpuK = NULL;
	float *d_qv_gpu = malloc(100000 * sizeof(float));
	cl_mem d_qv_gpuK = NULL;
	cl_float4 *d_fv_gpu = malloc(100000 * sizeof(cl_float4));
	cl_mem d_fv_gpuK = NULL;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(d_par_gpu_alpha, d_dim_gpu_number_boxes, d_box_gpu_offset, 1000, d_box_gpu_nn, 1000, d_box_gpu_nei_number, 26000, d_rv_gpu, 100000, d_qv_gpu, 100000, d_fv_gpu, 100000);
	PRINT_SUCCESS();

	/* Get platforms IDs */
	PRINT_STEP("Getting platforms IDs...");
	fRet = clGetPlatformIDs(0, NULL, &platformsLen);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clGetPlatformIDs"));
	platforms = malloc(platformsLen * sizeof(cl_platform_id));
	fRet = clGetPlatformIDs(platformsLen, platforms, NULL);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clGetPlatformIDs"));
	PRINT_SUCCESS();

	/* Get devices IDs for first platform availble */
	PRINT_STEP("Getting devices IDs for first platform...");
	fRet = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &devicesLen);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clGetDevicesIDs"));
	devices = malloc(devicesLen * sizeof(cl_device_id));
	fRet = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, devicesLen, devices, NULL);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clGetDevicesIDs"));
	PRINT_SUCCESS();

	/* Create context for first available device */
	PRINT_STEP("Creating context...");
	context = clCreateContext(NULL, 1, devices, NULL, NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateContext"));
	PRINT_SUCCESS();

	/* Create command queue for kernel_gpu_opencl kernel */
	PRINT_STEP("Creating command queue for \"kernel_gpu_opencl\"...");
	queueKernel_Gpu_Opencl = clCreateCommandQueue(context, devices[0], 0, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateCommandQueue"));
	PRINT_SUCCESS();

	/* Open binary file */
	PRINT_STEP("Opening program binary...");
	programFile = fopen("program.aocx", "rb");
	ASSERT_CALL(programFile, POSIX_ERROR_STATEMENTS("program.aocx"));
	PRINT_SUCCESS();

	/* Get size and read file */
	PRINT_STEP("Reading program binary...");
	fseek(programFile, 0, SEEK_END);
	programSz = ftell(programFile);
	fseek(programFile, 0, SEEK_SET);
	programContent = malloc(programSz);
	fread(programContent, programSz, 1, programFile);
	fclose(programFile);
	programFile = NULL;
	PRINT_SUCCESS();

	/* Create program from binary file */
	PRINT_STEP("Creating program from binary...");
	program = clCreateProgramWithBinary(context, 1, devices, &programSz, (const unsigned char **) &programContent, &programRet, &fRet);
	ASSERT_CALL(CL_SUCCESS == programRet, FUNCTION_ERROR_STATEMENTS("clCreateProgramWithBinary (when loading binary)"));
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateProgramWithBinary"));
	PRINT_SUCCESS();

	/* Build program */
	PRINT_STEP("Building program...");
	fRet = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clBuildProgram"));
	PRINT_SUCCESS();

	/* Create kernel_gpu_opencl kernel */
	PRINT_STEP("Creating kernel \"kernel_gpu_opencl\" from program...");
	kernelKernel_Gpu_Opencl = clCreateKernel(program, "kernel_gpu_opencl", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	d_box_gpu_offsetK = clCreateBuffer(context, CL_MEM_READ_ONLY, 1000 * sizeof(long), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (d_box_gpu_offsetK)"));
	d_box_gpu_nnK = clCreateBuffer(context, CL_MEM_READ_ONLY, 1000 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (d_box_gpu_nnK)"));
	d_box_gpu_nei_numberK = clCreateBuffer(context, CL_MEM_READ_ONLY, 26000 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (d_box_gpu_nei_numberK)"));
	d_rv_gpuK = clCreateBuffer(context, CL_MEM_READ_ONLY, 100000 * sizeof(cl_float4), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (d_rv_gpuK)"));
	d_qv_gpuK = clCreateBuffer(context, CL_MEM_READ_ONLY, 100000 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (d_qv_gpuK)"));
	d_fv_gpuK = clCreateBuffer(context, CL_MEM_READ_WRITE, 100000 * sizeof(cl_float4), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (d_fv_gpuK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for kernel_gpu_opencl */
	PRINT_STEP("Setting kernel arguments for \"kernel_gpu_opencl\"...");
	fRet = clSetKernelArg(kernelKernel_Gpu_Opencl, 0, sizeof(float), &d_par_gpu_alpha);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_par_gpu_alpha)"));
	fRet = clSetKernelArg(kernelKernel_Gpu_Opencl, 1, sizeof(long), &d_dim_gpu_number_boxes);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_dim_gpu_number_boxes)"));
	fRet = clSetKernelArg(kernelKernel_Gpu_Opencl, 2, sizeof(cl_mem), &d_box_gpu_offsetK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_box_gpu_offsetK)"));
	fRet = clSetKernelArg(kernelKernel_Gpu_Opencl, 3, sizeof(cl_mem), &d_box_gpu_nnK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_box_gpu_nnK)"));
	fRet = clSetKernelArg(kernelKernel_Gpu_Opencl, 4, sizeof(cl_mem), &d_box_gpu_nei_numberK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_box_gpu_nei_numberK)"));
	fRet = clSetKernelArg(kernelKernel_Gpu_Opencl, 5, sizeof(cl_mem), &d_rv_gpuK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_rv_gpuK)"));
	fRet = clSetKernelArg(kernelKernel_Gpu_Opencl, 6, sizeof(cl_mem), &d_qv_gpuK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_qv_gpuK)"));
	fRet = clSetKernelArg(kernelKernel_Gpu_Opencl, 7, sizeof(cl_mem), &d_fv_gpuK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_fv_gpuK)"));
	PRINT_SUCCESS();

	do {
		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clSetKernelArg(kernelKernel_Gpu_Opencl, 0, sizeof(float), &d_par_gpu_alpha);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_par_gpu_alpha)"));
		fRet = clSetKernelArg(kernelKernel_Gpu_Opencl, 1, sizeof(long), &d_dim_gpu_number_boxes);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_dim_gpu_number_boxes)"));
		fRet = clEnqueueWriteBuffer(queueKernel_Gpu_Opencl, d_box_gpu_offsetK, CL_TRUE, 0, 1000 * sizeof(long), d_box_gpu_offset, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (d_box_gpu_offsetK)"));
		fRet = clEnqueueWriteBuffer(queueKernel_Gpu_Opencl, d_box_gpu_nnK, CL_TRUE, 0, 1000 * sizeof(int), d_box_gpu_nn, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (d_box_gpu_nnK)"));
		fRet = clEnqueueWriteBuffer(queueKernel_Gpu_Opencl, d_box_gpu_nei_numberK, CL_TRUE, 0, 26000 * sizeof(int), d_box_gpu_nei_number, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (d_box_gpu_nei_numberK)"));
		fRet = clEnqueueWriteBuffer(queueKernel_Gpu_Opencl, d_rv_gpuK, CL_TRUE, 0, 100000 * sizeof(cl_float4), d_rv_gpu, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (d_rv_gpuK)"));
		fRet = clEnqueueWriteBuffer(queueKernel_Gpu_Opencl, d_qv_gpuK, CL_TRUE, 0, 100000 * sizeof(float), d_qv_gpu, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (d_qv_gpuK)"));
		fRet = clEnqueueWriteBuffer(queueKernel_Gpu_Opencl, d_fv_gpuK, CL_TRUE, 0, 100000 * sizeof(cl_float4), d_fv_gpu, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (d_fv_gpuK)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueKernel_Gpu_Opencl, kernelKernel_Gpu_Opencl, workDimKernel_Gpu_Opencl, NULL, globalSizeKernel_Gpu_Opencl, localSizeKernel_Gpu_Opencl, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueKernel_Gpu_Opencl);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueKernel_Gpu_Opencl, d_fv_gpuK, CL_TRUE, 0, 100000 * sizeof(cl_float4), d_fv_gpu, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueReadBuffer"));
		PRINT_SUCCESS();

		timersub(&tNow, &tThen, &tDelta);
		timeradd(&tExecTime, &tDelta, &tExecTime);
		i++;
	} while(loopFlag);

	/* Calling postamble function */
	PRINT_STEP("Calling postamble function...");
	POSTAMBLE(d_par_gpu_alpha, d_dim_gpu_number_boxes, d_box_gpu_offset, 1000, d_box_gpu_nn, 1000, d_box_gpu_nei_number, 26000, d_rv_gpu, 100000, d_qv_gpu, 100000, d_fv_gpu, 100000);
	PRINT_SUCCESS();

	/* Print profiling results */
	long totalTime = (1000000 * tExecTime.tv_sec) + tExecTime.tv_usec;
	printf("Elapsed time spent on kernels: %ld us; Average time per iteration: %lf us.\n", totalTime, totalTime / (double) i);

	/* Validate received data */
	PRINT_STEP("Validating received data...");
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(d_box_gpu_offsetK)
		clReleaseMemObject(d_box_gpu_offsetK);
	if(d_box_gpu_nnK)
		clReleaseMemObject(d_box_gpu_nnK);
	if(d_box_gpu_nei_numberK)
		clReleaseMemObject(d_box_gpu_nei_numberK);
	if(d_rv_gpuK)
		clReleaseMemObject(d_rv_gpuK);
	if(d_qv_gpuK)
		clReleaseMemObject(d_qv_gpuK);
	if(d_fv_gpuK)
		clReleaseMemObject(d_fv_gpuK);

	/* Dealloc variables */
	free(d_box_gpu_offset);
	free(d_box_gpu_nn);
	free(d_box_gpu_nei_number);
	free(d_rv_gpu);
	free(d_qv_gpu);
	free(d_fv_gpu);

	/* Dealloc kernels */
	if(kernelKernel_Gpu_Opencl)
		clReleaseKernel(kernelKernel_Gpu_Opencl);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueKernel_Gpu_Opencl)
		clReleaseCommandQueue(queueKernel_Gpu_Opencl);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);


	return rv;
}
