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
 *            PREAMBLE(reference, referenceSz, data, dataSz, dataC, dataCSz, input_v, input_vSz, dim, penalty, loop_exit, block_offset);
 *            POSTAMBLE(reference, referenceSz, data, dataSz, dataC, dataCSz, input_v, input_vSz, dim, penalty, loop_exit, block_offset);
 *            LOOPPREAMBLE(reference, referenceSz, data, dataSz, dataC, dataCSz, input_v, input_vSz, dim, penalty, loop_exit, block_offset, loopFlag);
 *            LOOPPOSTAMBLE(reference, referenceSz, data, dataSz, dataC, dataCSz, input_v, input_vSz, dim, penalty, loop_exit, block_offset, loopFlag);
 *            CLEANUP(reference, referenceSz, data, dataSz, dataC, dataCSz, input_v, input_vSz, dim, penalty, loop_exit, block_offset);
 *        where:
 *            reference: variable (int *);
 *            referenceSz: number of members in variable (unsigned int);
 *            data: variable (int *);
 *            dataSz: number of members in variable (unsigned int);
 *            dataC: variable (int *);
 *            dataCSz: number of members in variable (unsigned int);
 *            input_v: variable (int *);
 *            input_vSz: number of members in variable (unsigned int);
 *            dim: variable (int);
 *            penalty: variable (int);
 *            loop_exit: variable (int);
 *            block_offset: variable (int);
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
	cl_command_queue queueNw_Kernel1 = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelNw_Kernel1 = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	long totalTime;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimNw_Kernel1 = 1;
	size_t globalSizeNw_Kernel1[1] = {
		1
	};
	size_t localSizeNw_Kernel1[1] = {
		1
	};

	/* Input/output variables */
	int *reference = malloc(4196352 * sizeof(int));
	cl_mem referenceK = NULL;
	int *data = malloc(4196352 * sizeof(int));
	int *dataC = malloc(4196352 * sizeof(int));
	cl_mem dataK = NULL;
	int *input_v = malloc(2049 * sizeof(int));
	cl_mem input_vK = NULL;
	int dim = 2048;
	int penalty = 10;
	int loop_exit;
	int block_offset;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(reference, 4196352, data, 4196352, dataC, 4196352, input_v, 2049, dim, penalty, loop_exit, block_offset);
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

	/* Create command queue for nw_kernel1 kernel */
	PRINT_STEP("Creating command queue for \"nw_kernel1\"...");
	queueNw_Kernel1 = clCreateCommandQueue(context, devices[0], 0, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateCommandQueue"));
	PRINT_SUCCESS();

	/* Open binary file */
	PRINT_STEP("Opening program binary...");
	programFile = fopen("kern.cl", "rb");
	ASSERT_CALL(programFile, POSIX_ERROR_STATEMENTS("kern.cl"));
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

	/* Create program from source file */
	PRINT_STEP("Creating program from source...");
	program = clCreateProgramWithSource(context, 1, (const char **) &programContent, &programSz, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateProgramWithSource"));
	PRINT_SUCCESS();

	/* Build program */
	PRINT_STEP("Building program...");
	fRet = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clBuildProgram"));
	PRINT_SUCCESS();

	/* Create nw_kernel1 kernel */
	PRINT_STEP("Creating kernel \"nw_kernel1\" from program...");
	kernelNw_Kernel1 = clCreateKernel(program, "nw_kernel1", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	referenceK = clCreateBuffer(context, CL_MEM_READ_ONLY, 4196352 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (referenceK)"));
	dataK = clCreateBuffer(context, CL_MEM_READ_WRITE, 4196352 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (dataK)"));
	input_vK = clCreateBuffer(context, CL_MEM_READ_ONLY, 2049 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (input_vK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for nw_kernel1 */
	PRINT_STEP("Setting kernel arguments for \"nw_kernel1\"...");
	fRet = clSetKernelArg(kernelNw_Kernel1, 0, sizeof(cl_mem), &referenceK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (referenceK)"));
	fRet = clSetKernelArg(kernelNw_Kernel1, 1, sizeof(cl_mem), &dataK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (dataK)"));
	fRet = clSetKernelArg(kernelNw_Kernel1, 2, sizeof(cl_mem), &input_vK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (input_vK)"));
	fRet = clSetKernelArg(kernelNw_Kernel1, 3, sizeof(int), &dim);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (dim)"));
	fRet = clSetKernelArg(kernelNw_Kernel1, 4, sizeof(int), &penalty);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (penalty)"));
	fRet = clSetKernelArg(kernelNw_Kernel1, 5, sizeof(int), &loop_exit);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (loop_exit)"));
	fRet = clSetKernelArg(kernelNw_Kernel1, 6, sizeof(int), &block_offset);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (block_offset)"));
	PRINT_SUCCESS();

	do {
		/* Calling loop preamble function */
		PRINT_STEP("[%d] Calling loop preamble function...", i);
		LOOPPREAMBLE(reference, 4196352, data, 4196352, dataC, 4196352, input_v, 2049, dim, penalty, loop_exit, block_offset, loopFlag);
		PRINT_SUCCESS();

		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clEnqueueWriteBuffer(queueNw_Kernel1, referenceK, CL_TRUE, 0, 4196352 * sizeof(int), reference, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (referenceK)"));
		fRet = clEnqueueWriteBuffer(queueNw_Kernel1, dataK, CL_TRUE, 0, 4196352 * sizeof(int), data, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (dataK)"));
		fRet = clEnqueueWriteBuffer(queueNw_Kernel1, input_vK, CL_TRUE, 0, 2049 * sizeof(int), input_v, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (input_vK)"));
		fRet = clSetKernelArg(kernelNw_Kernel1, 3, sizeof(int), &dim);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (dim)"));
		fRet = clSetKernelArg(kernelNw_Kernel1, 4, sizeof(int), &penalty);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (penalty)"));
		fRet = clSetKernelArg(kernelNw_Kernel1, 5, sizeof(int), &loop_exit);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (loop_exit)"));
		fRet = clSetKernelArg(kernelNw_Kernel1, 6, sizeof(int), &block_offset);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (block_offset)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueNw_Kernel1, kernelNw_Kernel1, workDimNw_Kernel1, NULL, globalSizeNw_Kernel1, localSizeNw_Kernel1, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueNw_Kernel1);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueNw_Kernel1, dataK, CL_TRUE, 0, 4196352 * sizeof(int), data, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueReadBuffer"));
		PRINT_SUCCESS();

		/* Calling loop postamble function */
		PRINT_STEP("[%d] Calling loop postamble function...", i);
		LOOPPOSTAMBLE(reference, 4196352, data, 4196352, dataC, 4196352, input_v, 2049, dim, penalty, loop_exit, block_offset, loopFlag);
		PRINT_SUCCESS();
		timersub(&tNow, &tThen, &tDelta);
		timeradd(&tExecTime, &tDelta, &tExecTime);
		i++;
	} while(loopFlag);


	/* Print profiling results */
	totalTime = (1000000 * tExecTime.tv_sec) + tExecTime.tv_usec;
	printf("Elapsed time spent on kernels: %ld us; Average time per iteration: %lf us.\n", totalTime, totalTime / (double) i);

	/* Validate received data */
	PRINT_STEP("Validating received data...");
	for(i = 0; i < 4196352; i++) {
		if(dataC[i] != data[i]) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable data[%d]: expected %d got %d.\n", i, dataC[i], data[i]);
		}
	}
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(referenceK)
		clReleaseMemObject(referenceK);
	if(dataK)
		clReleaseMemObject(dataK);
	if(input_vK)
		clReleaseMemObject(input_vK);

	/* Dealloc variables */
	free(reference);
	free(data);
	free(dataC);
	free(input_v);

	/* Dealloc kernels */
	if(kernelNw_Kernel1)
		clReleaseKernel(kernelNw_Kernel1);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueNw_Kernel1)
		clReleaseCommandQueue(queueNw_Kernel1);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);


	return rv;
}
