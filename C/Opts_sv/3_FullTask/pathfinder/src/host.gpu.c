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
 *            PREAMBLE(wall, wallSz, src, srcSz, dst, dstSz, cols, rem_rows, starting_row, comp_exit);
 *            POSTAMBLE(wall, wallSz, src, srcSz, dst, dstSz, cols, rem_rows, starting_row, comp_exit);
 *            LOOPPREAMBLE(wall, wallSz, src, srcSz, dst, dstSz, cols, rem_rows, starting_row, comp_exit, loopFlag);
 *            LOOPPOSTAMBLE(wall, wallSz, src, srcSz, dst, dstSz, cols, rem_rows, starting_row, comp_exit, loopFlag);
 *            CLEANUP(wall, wallSz, src, srcSz, dst, dstSz, cols, rem_rows, starting_row, comp_exit);
 *        where:
 *            wall: variable (int *);
 *            wallSz: number of members in variable (unsigned int);
 *            src: variable (int *);
 *            srcSz: number of members in variable (unsigned int);
 *            dst: variable (int *);
 *            dstSz: number of members in variable (unsigned int);
 *            cols: variable (int);
 *            rem_rows: variable (int);
 *            starting_row: variable (int);
 *            comp_exit: variable (int);
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
	cl_command_queue queueDynproc_Kernel = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelDynproc_Kernel = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	long totalTime;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimDynproc_Kernel = 1;
	size_t globalSizeDynproc_Kernel[1] = {
		1
	};
	size_t localSizeDynproc_Kernel[1] = {
		1
	};

	/* Input/output variables */
	int *wall = malloc(990000 * sizeof(int));
	cl_mem wallK = NULL;
	int *src = malloc(10000 * sizeof(int));
	cl_mem srcK = NULL;
	int *dst = malloc(10000 * sizeof(int));
	cl_mem dstK = NULL;
	int cols = 10000;
	int rem_rows;
	int starting_row;
	int comp_exit;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(wall, 990000, src, 10000, dst, 10000, cols, rem_rows, starting_row, comp_exit);
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

	/* Create command queue for dynproc_kernel kernel */
	PRINT_STEP("Creating command queue for \"dynproc_kernel\"...");
	queueDynproc_Kernel = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create dynproc_kernel kernel */
	PRINT_STEP("Creating kernel \"dynproc_kernel\" from program...");
	kernelDynproc_Kernel = clCreateKernel(program, "dynproc_kernel", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	wallK = clCreateBuffer(context, CL_MEM_READ_ONLY, 990000 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (wallK)"));
	srcK = clCreateBuffer(context, CL_MEM_READ_ONLY, 10000 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (srcK)"));
	dstK = clCreateBuffer(context, CL_MEM_READ_WRITE, 10000 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (dstK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for dynproc_kernel */
	PRINT_STEP("Setting kernel arguments for \"dynproc_kernel\"...");
	fRet = clSetKernelArg(kernelDynproc_Kernel, 0, sizeof(cl_mem), &wallK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (wallK)"));
	fRet = clSetKernelArg(kernelDynproc_Kernel, 1, sizeof(cl_mem), &srcK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (srcK)"));
	fRet = clSetKernelArg(kernelDynproc_Kernel, 2, sizeof(cl_mem), &dstK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (dstK)"));
	fRet = clSetKernelArg(kernelDynproc_Kernel, 3, sizeof(int), &cols);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (cols)"));
	fRet = clSetKernelArg(kernelDynproc_Kernel, 4, sizeof(int), &rem_rows);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (rem_rows)"));
	fRet = clSetKernelArg(kernelDynproc_Kernel, 5, sizeof(int), &starting_row);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (starting_row)"));
	fRet = clSetKernelArg(kernelDynproc_Kernel, 6, sizeof(int), &comp_exit);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (comp_exit)"));
	PRINT_SUCCESS();

	do {
		/* Calling loop preamble function */
		PRINT_STEP("[%d] Calling loop preamble function...", i);
		LOOPPREAMBLE(wall, 990000, src, 10000, dst, 10000, cols, rem_rows, starting_row, comp_exit, loopFlag);
		PRINT_SUCCESS();

		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clEnqueueWriteBuffer(queueDynproc_Kernel, wallK, CL_TRUE, 0, 990000 * sizeof(int), wall, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (wallK)"));
		fRet = clEnqueueWriteBuffer(queueDynproc_Kernel, srcK, CL_TRUE, 0, 10000 * sizeof(int), src, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (srcK)"));
		fRet = clEnqueueWriteBuffer(queueDynproc_Kernel, dstK, CL_TRUE, 0, 10000 * sizeof(int), dst, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (dstK)"));
		fRet = clSetKernelArg(kernelDynproc_Kernel, 3, sizeof(int), &cols);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (cols)"));
		fRet = clSetKernelArg(kernelDynproc_Kernel, 4, sizeof(int), &rem_rows);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (rem_rows)"));
		fRet = clSetKernelArg(kernelDynproc_Kernel, 5, sizeof(int), &starting_row);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (starting_row)"));
		fRet = clSetKernelArg(kernelDynproc_Kernel, 6, sizeof(int), &comp_exit);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (comp_exit)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueDynproc_Kernel, kernelDynproc_Kernel, workDimDynproc_Kernel, NULL, globalSizeDynproc_Kernel, localSizeDynproc_Kernel, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueDynproc_Kernel);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueDynproc_Kernel, dstK, CL_TRUE, 0, 10000 * sizeof(int), dst, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueReadBuffer"));
		PRINT_SUCCESS();

		/* Calling loop postamble function */
		PRINT_STEP("[%d] Calling loop postamble function...", i);
		LOOPPOSTAMBLE(wall, 990000, src, 10000, dst, 10000, cols, rem_rows, starting_row, comp_exit, loopFlag);
		PRINT_SUCCESS();
		timersub(&tNow, &tThen, &tDelta);
		timeradd(&tExecTime, &tDelta, &tExecTime);
		i++;
	} while(loopFlag);

	/* Calling postamble function */
	PRINT_STEP("Calling postamble function...");
	POSTAMBLE(wall, 990000, src, 10000, dst, 10000, cols, rem_rows, starting_row, comp_exit);
	PRINT_SUCCESS();

	/* Print profiling results */
	totalTime = (1000000 * tExecTime.tv_sec) + tExecTime.tv_usec;
	printf("Elapsed time spent on kernels: %ld us; Average time per iteration: %lf us.\n", totalTime, totalTime / (double) i);

	/* Validate received data */
	PRINT_STEP("Validating received data...");
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(wallK)
		clReleaseMemObject(wallK);
	if(srcK)
		clReleaseMemObject(srcK);
	if(dstK)
		clReleaseMemObject(dstK);

	/* Dealloc variables */
	free(wall);
	free(src);
	free(dst);

	/* Dealloc kernels */
	if(kernelDynproc_Kernel)
		clReleaseKernel(kernelDynproc_Kernel);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueDynproc_Kernel)
		clReleaseCommandQueue(queueDynproc_Kernel);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);

	/* Calling cleanup function */
	CLEANUP(wall, 990000, src, 10000, dst, 10000, cols, rem_rows, starting_row, comp_exit);

	return rv;
}
