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
 *            PREAMBLE(iteration, power, powerSz, temp_src, temp_srcSz, temp_dst, temp_dstSz, grid_cols, grid_rows, border_cols, border_rows, Cap, Rx, Ry, Rz, step);
 *            POSTAMBLE(iteration, power, powerSz, temp_src, temp_srcSz, temp_dst, temp_dstSz, grid_cols, grid_rows, border_cols, border_rows, Cap, Rx, Ry, Rz, step);
 *            LOOPPREAMBLE(iteration, power, powerSz, temp_src, temp_srcSz, temp_dst, temp_dstSz, grid_cols, grid_rows, border_cols, border_rows, Cap, Rx, Ry, Rz, step, loopFlag);
 *            LOOPPOSTAMBLE(iteration, power, powerSz, temp_src, temp_srcSz, temp_dst, temp_dstSz, grid_cols, grid_rows, border_cols, border_rows, Cap, Rx, Ry, Rz, step, loopFlag);
 *            CLEANUP(iteration, power, powerSz, temp_src, temp_srcSz, temp_dst, temp_dstSz, grid_cols, grid_rows, border_cols, border_rows, Cap, Rx, Ry, Rz, step);
 *        where:
 *            iteration: variable (int);
 *            power: variable (float *);
 *            powerSz: number of members in variable (unsigned int);
 *            temp_src: variable (float *);
 *            temp_srcSz: number of members in variable (unsigned int);
 *            temp_dst: variable (float *);
 *            temp_dstSz: number of members in variable (unsigned int);
 *            grid_cols: variable (int);
 *            grid_rows: variable (int);
 *            border_cols: variable (int);
 *            border_rows: variable (int);
 *            Cap: variable (float);
 *            Rx: variable (float);
 *            Ry: variable (float);
 *            Rz: variable (float);
 *            step: variable (float);
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
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimHotspot = 2;
	size_t globalSizeHotspot[2] = {
		688, 688
	};
	size_t localSizeHotspot[2] = {
		16, 16
	};

	/* Input/output variables */
	int iteration;
	float *power = malloc(262144 * sizeof(float));
	cl_mem powerK = NULL;
	float *temp_src = malloc(262144 * sizeof(float));
	cl_mem temp_srcK = NULL;
	float *temp_dst = malloc(262144 * sizeof(float));
	cl_mem temp_dstK = NULL;
	int grid_cols = 512;
	int grid_rows = 512;
	int border_cols;
	int border_rows;
	float Cap;
	float Rx;
	float Ry;
	float Rz;
	float step;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(iteration, power, 262144, temp_src, 262144, temp_dst, 262144, grid_cols, grid_rows, border_cols, border_rows, Cap, Rx, Ry, Rz, step);
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

	/* Create hotspot kernel */
	PRINT_STEP("Creating kernel \"hotspot\" from program...");
	kernelHotspot = clCreateKernel(program, "hotspot", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	powerK = clCreateBuffer(context, CL_MEM_READ_ONLY, 262144 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (powerK)"));
	temp_srcK = clCreateBuffer(context, CL_MEM_READ_WRITE, 262144 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (temp_srcK)"));
	temp_dstK = clCreateBuffer(context, CL_MEM_READ_WRITE, 262144 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (temp_dstK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for hotspot */
	PRINT_STEP("Setting kernel arguments for \"hotspot\"...");
	fRet = clSetKernelArg(kernelHotspot, 0, sizeof(int), &iteration);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (iteration)"));
	fRet = clSetKernelArg(kernelHotspot, 1, sizeof(cl_mem), &powerK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (powerK)"));
	fRet = clSetKernelArg(kernelHotspot, 2, sizeof(cl_mem), &temp_srcK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (temp_srcK)"));
	fRet = clSetKernelArg(kernelHotspot, 3, sizeof(cl_mem), &temp_dstK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (temp_dstK)"));
	fRet = clSetKernelArg(kernelHotspot, 4, sizeof(int), &grid_cols);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (grid_cols)"));
	fRet = clSetKernelArg(kernelHotspot, 5, sizeof(int), &grid_rows);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (grid_rows)"));
	fRet = clSetKernelArg(kernelHotspot, 6, sizeof(int), &border_cols);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (border_cols)"));
	fRet = clSetKernelArg(kernelHotspot, 7, sizeof(int), &border_rows);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (border_rows)"));
	fRet = clSetKernelArg(kernelHotspot, 8, sizeof(float), &Cap);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (Cap)"));
	fRet = clSetKernelArg(kernelHotspot, 9, sizeof(float), &Rx);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (Rx)"));
	fRet = clSetKernelArg(kernelHotspot, 10, sizeof(float), &Ry);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (Ry)"));
	fRet = clSetKernelArg(kernelHotspot, 11, sizeof(float), &Rz);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (Rz)"));
	fRet = clSetKernelArg(kernelHotspot, 12, sizeof(float), &step);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (step)"));
	PRINT_SUCCESS();

	do {
		/* Calling loop preamble function */
		PRINT_STEP("[%d] Calling loop preamble function...", i);
		LOOPPREAMBLE(iteration, power, 262144, temp_src, 262144, temp_dst, 262144, grid_cols, grid_rows, border_cols, border_rows, Cap, Rx, Ry, Rz, step, loopFlag);
		PRINT_SUCCESS();

		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clSetKernelArg(kernelHotspot, 0, sizeof(int), &iteration);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (iteration)"));
		fRet = clEnqueueWriteBuffer(queueHotspot, powerK, CL_TRUE, 0, 262144 * sizeof(float), power, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (powerK)"));
		fRet = clEnqueueWriteBuffer(queueHotspot, temp_srcK, CL_TRUE, 0, 262144 * sizeof(float), temp_src, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (temp_srcK)"));
		fRet = clEnqueueWriteBuffer(queueHotspot, temp_dstK, CL_TRUE, 0, 262144 * sizeof(float), temp_dst, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (temp_dstK)"));
		fRet = clSetKernelArg(kernelHotspot, 4, sizeof(int), &grid_cols);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (grid_cols)"));
		fRet = clSetKernelArg(kernelHotspot, 5, sizeof(int), &grid_rows);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (grid_rows)"));
		fRet = clSetKernelArg(kernelHotspot, 6, sizeof(int), &border_cols);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (border_cols)"));
		fRet = clSetKernelArg(kernelHotspot, 7, sizeof(int), &border_rows);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (border_rows)"));
		fRet = clSetKernelArg(kernelHotspot, 8, sizeof(float), &Cap);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (Cap)"));
		fRet = clSetKernelArg(kernelHotspot, 9, sizeof(float), &Rx);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (Rx)"));
		fRet = clSetKernelArg(kernelHotspot, 10, sizeof(float), &Ry);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (Ry)"));
		fRet = clSetKernelArg(kernelHotspot, 11, sizeof(float), &Rz);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (Rz)"));
		fRet = clSetKernelArg(kernelHotspot, 12, sizeof(float), &step);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (step)"));
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
		fRet = clEnqueueReadBuffer(queueHotspot, temp_srcK, CL_TRUE, 0, 262144 * sizeof(float), temp_src, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueHotspot, temp_dstK, CL_TRUE, 0, 262144 * sizeof(float), temp_dst, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueReadBuffer"));
		PRINT_SUCCESS();

		/* Calling loop postamble function */
		PRINT_STEP("[%d] Calling loop postamble function...", i);
		LOOPPOSTAMBLE(iteration, power, 262144, temp_src, 262144, temp_dst, 262144, grid_cols, grid_rows, border_cols, border_rows, Cap, Rx, Ry, Rz, step, loopFlag);
		PRINT_SUCCESS();
		timersub(&tNow, &tThen, &tDelta);
		timeradd(&tExecTime, &tDelta, &tExecTime);
		i++;
	} while(loopFlag);

	/* Calling postamble function */
	PRINT_STEP("Calling postamble function...");
	POSTAMBLE(iteration, power, 262144, temp_src, 262144, temp_dst, 262144, grid_cols, grid_rows, border_cols, border_rows, Cap, Rx, Ry, Rz, step);
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
	if(powerK)
		clReleaseMemObject(powerK);
	if(temp_srcK)
		clReleaseMemObject(temp_srcK);
	if(temp_dstK)
		clReleaseMemObject(temp_dstK);

	/* Dealloc variables */
	free(power);
	free(temp_src);
	free(temp_dst);

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
