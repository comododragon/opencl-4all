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
 *            PREAMBLE(levels, levelsSz, levelsC, levelsCSz, edgeArray, edgeArraySz, edgeArrayAux, edgeArrayAuxSz, W_SZ, CHUNK_SZ, numVertices, curr, flag);
 *            POSTAMBLE(levels, levelsSz, levelsC, levelsCSz, edgeArray, edgeArraySz, edgeArrayAux, edgeArrayAuxSz, W_SZ, CHUNK_SZ, numVertices, curr, flag);
 *            LOOPPREAMBLE(levels, levelsSz, levelsC, levelsCSz, edgeArray, edgeArraySz, edgeArrayAux, edgeArrayAuxSz, W_SZ, CHUNK_SZ, numVertices, curr, flag, loopFlag);
 *            LOOPPOSTAMBLE(levels, levelsSz, levelsC, levelsCSz, edgeArray, edgeArraySz, edgeArrayAux, edgeArrayAuxSz, W_SZ, CHUNK_SZ, numVertices, curr, flag, loopFlag);
 *            CLEANUP(levels, levelsSz, levelsC, levelsCSz, edgeArray, edgeArraySz, edgeArrayAux, edgeArrayAuxSz, W_SZ, CHUNK_SZ, numVertices, curr, flag);
 *        where:
 *            levels: variable (unsigned int *);
 *            levelsSz: number of members in variable (unsigned int);
 *            levelsC: variable (unsigned int *);
 *            levelsCSz: number of members in variable (unsigned int);
 *            edgeArray: variable (unsigned int *);
 *            edgeArraySz: number of members in variable (unsigned int);
 *            edgeArrayAux: variable (unsigned int *);
 *            edgeArrayAuxSz: number of members in variable (unsigned int);
 *            W_SZ: variable (int);
 *            CHUNK_SZ: variable (int);
 *            numVertices: variable (unsigned int);
 *            curr: variable (int);
 *            flag: variable (int);
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
	cl_command_queue queueBfs_Kernel_Warp = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelBfs_Kernel_Warp = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimBfs_Kernel_Warp = 1;
	size_t globalSizeBfs_Kernel_Warp[1] = {
		8192
	};
	size_t localSizeBfs_Kernel_Warp[1] = {
		8192
	};

	/* Input/output variables */
	unsigned int *levels = malloc(1000 * sizeof(unsigned int));
	unsigned int *levelsC = malloc(1000 * sizeof(unsigned int));
	cl_mem levelsK = NULL;
	unsigned int *edgeArray = malloc(1001 * sizeof(unsigned int));
	cl_mem edgeArrayK = NULL;
	unsigned int *edgeArrayAux = malloc(1998 * sizeof(unsigned int));
	cl_mem edgeArrayAuxK = NULL;
	int W_SZ = 32;
	int CHUNK_SZ = 32;
	unsigned int numVertices;
	int curr = 0;
	int flag;
	cl_mem flagK = NULL;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(levels, 1000, levelsC, 1000, edgeArray, 1001, edgeArrayAux, 1998, W_SZ, CHUNK_SZ, numVertices, curr, flag);
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

	/* Create command queue for BFS_kernel_warp kernel */
	PRINT_STEP("Creating command queue for \"BFS_kernel_warp\"...");
	queueBfs_Kernel_Warp = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create BFS_kernel_warp kernel */
	PRINT_STEP("Creating kernel \"BFS_kernel_warp\" from program...");
	kernelBfs_Kernel_Warp = clCreateKernel(program, "BFS_kernel_warp", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	levelsK = clCreateBuffer(context, CL_MEM_READ_WRITE, 1000 * sizeof(unsigned int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (levelsK)"));
	edgeArrayK = clCreateBuffer(context, CL_MEM_READ_ONLY, 1001 * sizeof(unsigned int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (edgeArrayK)"));
	edgeArrayAuxK = clCreateBuffer(context, CL_MEM_READ_ONLY, 1998 * sizeof(unsigned int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (edgeArrayAuxK)"));
	flagK = clCreateBuffer(context, CL_MEM_READ_WRITE, 1 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (flagK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for BFS_kernel_warp */
	PRINT_STEP("Setting kernel arguments for \"BFS_kernel_warp\"...");
	fRet = clSetKernelArg(kernelBfs_Kernel_Warp, 0, sizeof(cl_mem), &levelsK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (levelsK)"));
	fRet = clSetKernelArg(kernelBfs_Kernel_Warp, 1, sizeof(cl_mem), &edgeArrayK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (edgeArrayK)"));
	fRet = clSetKernelArg(kernelBfs_Kernel_Warp, 2, sizeof(cl_mem), &edgeArrayAuxK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (edgeArrayAuxK)"));
	fRet = clSetKernelArg(kernelBfs_Kernel_Warp, 3, sizeof(int), &W_SZ);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (W_SZ)"));
	fRet = clSetKernelArg(kernelBfs_Kernel_Warp, 4, sizeof(int), &CHUNK_SZ);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (CHUNK_SZ)"));
	fRet = clSetKernelArg(kernelBfs_Kernel_Warp, 5, sizeof(unsigned int), &numVertices);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (numVertices)"));
	fRet = clSetKernelArg(kernelBfs_Kernel_Warp, 6, sizeof(int), &curr);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (curr)"));
	fRet = clSetKernelArg(kernelBfs_Kernel_Warp, 7, sizeof(cl_mem), &flagK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (flagK)"));
	PRINT_SUCCESS();

	do {
		/* Calling loop preamble function */
		PRINT_STEP("[%d] Calling loop preamble function...", i);
		LOOPPREAMBLE(levels, 1000, levelsC, 1000, edgeArray, 1001, edgeArrayAux, 1998, W_SZ, CHUNK_SZ, numVertices, curr, flag, loopFlag);
		PRINT_SUCCESS();

		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clEnqueueWriteBuffer(queueBfs_Kernel_Warp, levelsK, CL_TRUE, 0, 1000 * sizeof(unsigned int), levels, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (levelsK)"));
		fRet = clEnqueueWriteBuffer(queueBfs_Kernel_Warp, edgeArrayK, CL_TRUE, 0, 1001 * sizeof(unsigned int), edgeArray, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (edgeArrayK)"));
		fRet = clEnqueueWriteBuffer(queueBfs_Kernel_Warp, edgeArrayAuxK, CL_TRUE, 0, 1998 * sizeof(unsigned int), edgeArrayAux, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (edgeArrayAuxK)"));
		fRet = clSetKernelArg(kernelBfs_Kernel_Warp, 3, sizeof(int), &W_SZ);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (W_SZ)"));
		fRet = clSetKernelArg(kernelBfs_Kernel_Warp, 4, sizeof(int), &CHUNK_SZ);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (CHUNK_SZ)"));
		fRet = clSetKernelArg(kernelBfs_Kernel_Warp, 5, sizeof(unsigned int), &numVertices);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (numVertices)"));
		fRet = clSetKernelArg(kernelBfs_Kernel_Warp, 6, sizeof(int), &curr);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (curr)"));
		fRet = clEnqueueWriteBuffer(queueBfs_Kernel_Warp, flagK, CL_TRUE, 0, sizeof(int), &flag, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (flagK)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueBfs_Kernel_Warp, kernelBfs_Kernel_Warp, workDimBfs_Kernel_Warp, NULL, globalSizeBfs_Kernel_Warp, localSizeBfs_Kernel_Warp, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueBfs_Kernel_Warp);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueBfs_Kernel_Warp, levelsK, CL_TRUE, 0, 1000 * sizeof(unsigned int), levels, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueBfs_Kernel_Warp, flagK, CL_TRUE, 0, 1 * sizeof(int), &flag, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueReadBuffer"));
		PRINT_SUCCESS();

		/* Calling loop postamble function */
		PRINT_STEP("[%d] Calling loop postamble function...", i);
		LOOPPOSTAMBLE(levels, 1000, levelsC, 1000, edgeArray, 1001, edgeArrayAux, 1998, W_SZ, CHUNK_SZ, numVertices, curr, flag, loopFlag);
		PRINT_SUCCESS();
		timersub(&tNow, &tThen, &tDelta);
		timeradd(&tExecTime, &tDelta, &tExecTime);
		i++;
	} while(loopFlag);


	/* Print profiling results */
	long totalTime = (1000000 * tExecTime.tv_sec) + tExecTime.tv_usec;
	printf("Elapsed time spent on kernels: %ld us; Average time per iteration: %lf us.\n", totalTime, totalTime / (double) i);

	/* Validate received data */
	PRINT_STEP("Validating received data...");
	for(i = 0; i < 1000; i++) {
		if(levelsC[i] != levels[i]) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable levels[%d]: expected %u got %u.\n", i, levelsC[i], levels[i]);
		}
	}
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(levelsK)
		clReleaseMemObject(levelsK);
	if(edgeArrayK)
		clReleaseMemObject(edgeArrayK);
	if(edgeArrayAuxK)
		clReleaseMemObject(edgeArrayAuxK);

	/* Dealloc variables */
	free(levels);
	free(levelsC);
	free(edgeArray);
	free(edgeArrayAux);

	/* Dealloc kernels */
	if(kernelBfs_Kernel_Warp)
		clReleaseKernel(kernelBfs_Kernel_Warp);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueBfs_Kernel_Warp)
		clReleaseCommandQueue(queueBfs_Kernel_Warp);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);

	/* Calling cleanup function */
	CLEANUP(levels, 1000, levelsC, 1000, edgeArray, 1001, edgeArrayAux, 1998, W_SZ, CHUNK_SZ, numVertices, curr, flag);

	return rv;
}
