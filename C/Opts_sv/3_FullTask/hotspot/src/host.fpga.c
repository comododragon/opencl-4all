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
 *            PREAMBLE(power, powerSz, src, srcSz, dst, dstSz, grid_cols, grid_rows, sdc, Rx_1, Ry_1, Rz_1, comp_exit);
 *            POSTAMBLE(power, powerSz, src, srcSz, dst, dstSz, grid_cols, grid_rows, sdc, Rx_1, Ry_1, Rz_1, comp_exit);
 *            LOOPPREAMBLE(power, powerSz, src, srcSz, dst, dstSz, grid_cols, grid_rows, sdc, Rx_1, Ry_1, Rz_1, comp_exit, loopFlag);
 *            LOOPPOSTAMBLE(power, powerSz, src, srcSz, dst, dstSz, grid_cols, grid_rows, sdc, Rx_1, Ry_1, Rz_1, comp_exit, loopFlag);
 *            CLEANUP(power, powerSz, src, srcSz, dst, dstSz, grid_cols, grid_rows, sdc, Rx_1, Ry_1, Rz_1, comp_exit);
 *        where:
 *            power: variable (float *);
 *            powerSz: number of members in variable (unsigned int);
 *            src: variable (float *);
 *            srcSz: number of members in variable (unsigned int);
 *            dst: variable (float *);
 *            dstSz: number of members in variable (unsigned int);
 *            grid_cols: variable (int);
 *            grid_rows: variable (int);
 *            sdc: variable (float);
 *            Rx_1: variable (float);
 *            Ry_1: variable (float);
 *            Rz_1: variable (float);
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
	cl_command_queue queueHotspot = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelHotspot = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	long totalTime;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimHotspot = 1;
	size_t globalSizeHotspot[1] = {
		1
	};
	size_t localSizeHotspot[1] = {
		1
	};

	/* Input/output variables */
	float *power = malloc(262144 * sizeof(float));
	cl_mem powerK = NULL;
	float *src = malloc(262144 * sizeof(float));
	cl_mem srcK = NULL;
	float *dst = malloc(262144 * sizeof(float));
	cl_mem dstK = NULL;
	int grid_cols = 512;
	int grid_rows = 512;
	float sdc;
	float Rx_1;
	float Ry_1;
	float Rz_1;
	int comp_exit;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(power, 262144, src, 262144, dst, 262144, grid_cols, grid_rows, sdc, Rx_1, Ry_1, Rz_1, comp_exit);
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

	/* Create command queue for hotspot kernel */
	PRINT_STEP("Creating command queue for \"hotspot\"...");
	queueHotspot = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create hotspot kernel */
	PRINT_STEP("Creating kernel \"hotspot\" from program...");
	kernelHotspot = clCreateKernel(program, "hotspot", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	powerK = clCreateBuffer(context, CL_MEM_READ_ONLY, 262144 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (powerK)"));
	srcK = clCreateBuffer(context, CL_MEM_READ_WRITE, 262144 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (srcK)"));
	dstK = clCreateBuffer(context, CL_MEM_READ_WRITE, 262144 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (dstK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for hotspot */
	PRINT_STEP("Setting kernel arguments for \"hotspot\"...");
	fRet = clSetKernelArg(kernelHotspot, 0, sizeof(cl_mem), &powerK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (powerK)"));
	fRet = clSetKernelArg(kernelHotspot, 1, sizeof(cl_mem), &srcK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (srcK)"));
	fRet = clSetKernelArg(kernelHotspot, 2, sizeof(cl_mem), &dstK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (dstK)"));
	fRet = clSetKernelArg(kernelHotspot, 3, sizeof(int), &grid_cols);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (grid_cols)"));
	fRet = clSetKernelArg(kernelHotspot, 4, sizeof(int), &grid_rows);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (grid_rows)"));
	fRet = clSetKernelArg(kernelHotspot, 5, sizeof(float), &sdc);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (sdc)"));
	fRet = clSetKernelArg(kernelHotspot, 6, sizeof(float), &Rx_1);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (Rx_1)"));
	fRet = clSetKernelArg(kernelHotspot, 7, sizeof(float), &Ry_1);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (Ry_1)"));
	fRet = clSetKernelArg(kernelHotspot, 8, sizeof(float), &Rz_1);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (Rz_1)"));
	fRet = clSetKernelArg(kernelHotspot, 9, sizeof(int), &comp_exit);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (comp_exit)"));
	PRINT_SUCCESS();

	do {
		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clEnqueueWriteBuffer(queueHotspot, powerK, CL_TRUE, 0, 262144 * sizeof(float), power, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (powerK)"));
		fRet = clEnqueueWriteBuffer(queueHotspot, srcK, CL_TRUE, 0, 262144 * sizeof(float), src, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (srcK)"));
		fRet = clEnqueueWriteBuffer(queueHotspot, dstK, CL_TRUE, 0, 262144 * sizeof(float), dst, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (dstK)"));
		fRet = clSetKernelArg(kernelHotspot, 3, sizeof(int), &grid_cols);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (grid_cols)"));
		fRet = clSetKernelArg(kernelHotspot, 4, sizeof(int), &grid_rows);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (grid_rows)"));
		fRet = clSetKernelArg(kernelHotspot, 5, sizeof(float), &sdc);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (sdc)"));
		fRet = clSetKernelArg(kernelHotspot, 6, sizeof(float), &Rx_1);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (Rx_1)"));
		fRet = clSetKernelArg(kernelHotspot, 7, sizeof(float), &Ry_1);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (Ry_1)"));
		fRet = clSetKernelArg(kernelHotspot, 8, sizeof(float), &Rz_1);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (Rz_1)"));
		fRet = clSetKernelArg(kernelHotspot, 9, sizeof(int), &comp_exit);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (comp_exit)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueHotspot, kernelHotspot, workDimHotspot, NULL, globalSizeHotspot, localSizeHotspot, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueHotspot);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueHotspot, srcK, CL_TRUE, 0, 262144 * sizeof(float), src, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueHotspot, dstK, CL_TRUE, 0, 262144 * sizeof(float), dst, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueReadBuffer"));
		PRINT_SUCCESS();

		/* Calling loop postamble function */
		PRINT_STEP("[%d] Calling loop postamble function...", i);
		LOOPPOSTAMBLE(power, 262144, src, 262144, dst, 262144, grid_cols, grid_rows, sdc, Rx_1, Ry_1, Rz_1, comp_exit, loopFlag);
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
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(powerK)
		clReleaseMemObject(powerK);
	if(srcK)
		clReleaseMemObject(srcK);
	if(dstK)
		clReleaseMemObject(dstK);

	/* Dealloc variables */
	free(power);
	free(src);
	free(dst);

	/* Dealloc kernels */
	if(kernelHotspot)
		clReleaseKernel(kernelHotspot);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueHotspot)
		clReleaseCommandQueue(queueHotspot);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);


	return rv;
}
