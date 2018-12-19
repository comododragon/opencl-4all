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
 *            PREAMBLE(reference_d, reference_dSz, input_itemsets_d, input_itemsets_dSz, cols, penalty, blk, block_width, worksize, offset_r, offset_c);
 *            POSTAMBLE(reference_d, reference_dSz, input_itemsets_d, input_itemsets_dSz, cols, penalty, blk, block_width, worksize, offset_r, offset_c);
 *            LOOPPREAMBLE(reference_d, reference_dSz, input_itemsets_d, input_itemsets_dSz, cols, penalty, blk, block_width, worksize, offset_r, offset_c, loopFlag);
 *            LOOPPOSTAMBLE(reference_d, reference_dSz, input_itemsets_d, input_itemsets_dSz, cols, penalty, blk, block_width, worksize, offset_r, offset_c, loopFlag);
 *            CLEANUP(reference_d, reference_dSz, input_itemsets_d, input_itemsets_dSz, cols, penalty, blk, block_width, worksize, offset_r, offset_c);
 *        where:
 *            reference_d: variable (int *);
 *            reference_dSz: number of members in variable (unsigned int);
 *            input_itemsets_d: variable (int *);
 *            input_itemsets_dSz: number of members in variable (unsigned int);
 *            cols: variable (int);
 *            penalty: variable (int);
 *            blk: variable (int);
 *            block_width: variable (int);
 *            worksize: variable (int);
 *            offset_r: variable (int);
 *            offset_c: variable (int);
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
	cl_command_queue queueNw_Kernel2 = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelNw_Kernel2 = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimNw_Kernel2 = 1;
	size_t globalSizeNw_Kernel2[1] = {
		16
	};
	size_t localSizeNw_Kernel2[1] = {
		16
	};

	/* Input/output variables */
	int *reference_d = malloc(4198401 * sizeof(int));
	cl_mem reference_dK = NULL;
	int *input_itemsets_d = malloc(4198401 * sizeof(int));
	cl_mem input_itemsets_dK = NULL;
	int cols = 2049;
	int penalty = 10;
	int blk = 127;
	int block_width = 128;
	int worksize = 2048;
	int offset_r = 0;
	int offset_c = 0;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(reference_d, 4198401, input_itemsets_d, 4198401, cols, penalty, blk, block_width, worksize, offset_r, offset_c);
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

	/* Create command queue for nw_kernel2 kernel */
	PRINT_STEP("Creating command queue for \"nw_kernel2\"...");
	queueNw_Kernel2 = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create nw_kernel2 kernel */
	PRINT_STEP("Creating kernel \"nw_kernel2\" from program...");
	kernelNw_Kernel2 = clCreateKernel(program, "nw_kernel2", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	reference_dK = clCreateBuffer(context, CL_MEM_READ_ONLY, 4198401 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (reference_dK)"));
	input_itemsets_dK = clCreateBuffer(context, CL_MEM_READ_WRITE, 4198401 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (input_itemsets_dK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for nw_kernel2 */
	PRINT_STEP("Setting kernel arguments for \"nw_kernel2\"...");
	fRet = clSetKernelArg(kernelNw_Kernel2, 0, sizeof(cl_mem), &reference_dK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (reference_dK)"));
	fRet = clSetKernelArg(kernelNw_Kernel2, 1, sizeof(cl_mem), &input_itemsets_dK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (input_itemsets_dK)"));
	fRet = clSetKernelArg(kernelNw_Kernel2, 2, 289 * sizeof(int), NULL);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (__local 2)"));
	fRet = clSetKernelArg(kernelNw_Kernel2, 3, 256 * sizeof(int), NULL);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (__local 3)"));
	fRet = clSetKernelArg(kernelNw_Kernel2, 4, sizeof(int), &cols);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (cols)"));
	fRet = clSetKernelArg(kernelNw_Kernel2, 5, sizeof(int), &penalty);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (penalty)"));
	fRet = clSetKernelArg(kernelNw_Kernel2, 6, sizeof(int), &blk);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (blk)"));
	fRet = clSetKernelArg(kernelNw_Kernel2, 7, sizeof(int), &block_width);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (block_width)"));
	fRet = clSetKernelArg(kernelNw_Kernel2, 8, sizeof(int), &worksize);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (worksize)"));
	fRet = clSetKernelArg(kernelNw_Kernel2, 9, sizeof(int), &offset_r);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (offset_r)"));
	fRet = clSetKernelArg(kernelNw_Kernel2, 10, sizeof(int), &offset_c);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (offset_c)"));
	PRINT_SUCCESS();

	do {
		/* Calling loop preamble function */
		PRINT_STEP("[%d] Calling loop preamble function...", i);
		LOOPPREAMBLE(reference_d, 4198401, input_itemsets_d, 4198401, cols, penalty, blk, block_width, worksize, offset_r, offset_c, loopFlag);
		PRINT_SUCCESS();

		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clEnqueueWriteBuffer(queueNw_Kernel2, reference_dK, CL_TRUE, 0, 4198401 * sizeof(int), reference_d, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (reference_dK)"));
		fRet = clEnqueueWriteBuffer(queueNw_Kernel2, input_itemsets_dK, CL_TRUE, 0, 4198401 * sizeof(int), input_itemsets_d, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (input_itemsets_dK)"));
		fRet = clSetKernelArg(kernelNw_Kernel2, 4, sizeof(int), &cols);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (cols)"));
		fRet = clSetKernelArg(kernelNw_Kernel2, 5, sizeof(int), &penalty);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (penalty)"));
		fRet = clSetKernelArg(kernelNw_Kernel2, 6, sizeof(int), &blk);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (blk)"));
		fRet = clSetKernelArg(kernelNw_Kernel2, 7, sizeof(int), &block_width);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (block_width)"));
		fRet = clSetKernelArg(kernelNw_Kernel2, 8, sizeof(int), &worksize);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (worksize)"));
		fRet = clSetKernelArg(kernelNw_Kernel2, 9, sizeof(int), &offset_r);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (offset_r)"));
		fRet = clSetKernelArg(kernelNw_Kernel2, 10, sizeof(int), &offset_c);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (offset_c)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueNw_Kernel2, kernelNw_Kernel2, workDimNw_Kernel2, NULL, globalSizeNw_Kernel2, localSizeNw_Kernel2, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueNw_Kernel2);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueNw_Kernel2, input_itemsets_dK, CL_TRUE, 0, 4198401 * sizeof(int), input_itemsets_d, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueReadBuffer"));
		PRINT_SUCCESS();

		/* Calling loop postamble function */
		PRINT_STEP("[%d] Calling loop postamble function...", i);
		LOOPPOSTAMBLE(reference_d, 4198401, input_itemsets_d, 4198401, cols, penalty, blk, block_width, worksize, offset_r, offset_c, loopFlag);
		PRINT_SUCCESS();
		timersub(&tNow, &tThen, &tDelta);
		timeradd(&tExecTime, &tDelta, &tExecTime);
		i++;
	} while(loopFlag);

	/* Calling postamble function */
	PRINT_STEP("Calling postamble function...");
	POSTAMBLE(reference_d, 4198401, input_itemsets_d, 4198401, cols, penalty, blk, block_width, worksize, offset_r, offset_c);
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
	if(reference_dK)
		clReleaseMemObject(reference_dK);
	if(input_itemsets_dK)
		clReleaseMemObject(input_itemsets_dK);

	/* Dealloc variables */
	free(reference_d);
	free(input_itemsets_d);

	/* Dealloc kernels */
	if(kernelNw_Kernel2)
		clReleaseKernel(kernelNw_Kernel2);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueNw_Kernel2)
		clReleaseCommandQueue(queueNw_Kernel2);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);


	return rv;
}
