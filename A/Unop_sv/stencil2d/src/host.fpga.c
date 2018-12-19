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
 *            PREAMBLE(data, dataSz, newData, newDataSz, newDataC, newDataCSz, alignment, wCenter, wCardinal, wDiagonal);
 *            POSTAMBLE(data, dataSz, newData, newDataSz, newDataC, newDataCSz, alignment, wCenter, wCardinal, wDiagonal);
 *            LOOPPREAMBLE(data, dataSz, newData, newDataSz, newDataC, newDataCSz, alignment, wCenter, wCardinal, wDiagonal, loopFlag);
 *            LOOPPOSTAMBLE(data, dataSz, newData, newDataSz, newDataC, newDataCSz, alignment, wCenter, wCardinal, wDiagonal, loopFlag);
 *            CLEANUP(data, dataSz, newData, newDataSz, newDataC, newDataCSz, alignment, wCenter, wCardinal, wDiagonal);
 *        where:
 *            data: variable (float *);
 *            dataSz: number of members in variable (unsigned int);
 *            newData: variable (float *);
 *            newDataSz: number of members in variable (unsigned int);
 *            newDataC: variable (float *);
 *            newDataCSz: number of members in variable (unsigned int);
 *            alignment: variable (int);
 *            wCenter: variable (float);
 *            wCardinal: variable (float);
 *            wDiagonal: variable (float);
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
	cl_command_queue queueStencilkernel = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelStencilkernel = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimStencilkernel = 2;
	size_t globalSizeStencilkernel[2] = {
		64, 512
	};
	size_t localSizeStencilkernel[2] = {
		1, 256
	};

	/* Input/output variables */
	float *data = malloc(271392 * sizeof(float));
	cl_mem dataK = NULL;
	float *newData = malloc(271392 * sizeof(float));
	float *newDataC = malloc(271392 * sizeof(float));
	double newDataEpsilon = 0.1;
	cl_mem newDataK = NULL;
	int alignment = 16;
	float wCenter = 0.25;
	float wCardinal = 0.15;
	float wDiagonal = 0.05;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(data, 271392, newData, 271392, newDataC, 271392, alignment, wCenter, wCardinal, wDiagonal);
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

	/* Create command queue for StencilKernel kernel */
	PRINT_STEP("Creating command queue for \"StencilKernel\"...");
	queueStencilkernel = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create StencilKernel kernel */
	PRINT_STEP("Creating kernel \"StencilKernel\" from program...");
	kernelStencilkernel = clCreateKernel(program, "StencilKernel", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	dataK = clCreateBuffer(context, CL_MEM_READ_ONLY, 271392 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (dataK)"));
	newDataK = clCreateBuffer(context, CL_MEM_READ_WRITE, 271392 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (newDataK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for StencilKernel */
	PRINT_STEP("Setting kernel arguments for \"StencilKernel\"...");
	fRet = clSetKernelArg(kernelStencilkernel, 0, sizeof(cl_mem), &dataK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (dataK)"));
	fRet = clSetKernelArg(kernelStencilkernel, 1, sizeof(cl_mem), &newDataK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (newDataK)"));
	fRet = clSetKernelArg(kernelStencilkernel, 2, sizeof(int), &alignment);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (alignment)"));
	fRet = clSetKernelArg(kernelStencilkernel, 3, sizeof(float), &wCenter);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (wCenter)"));
	fRet = clSetKernelArg(kernelStencilkernel, 4, sizeof(float), &wCardinal);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (wCardinal)"));
	fRet = clSetKernelArg(kernelStencilkernel, 5, sizeof(float), &wDiagonal);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (wDiagonal)"));
	fRet = clSetKernelArg(kernelStencilkernel, 6, 2580 * sizeof(float), NULL);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (__local 6)"));
	PRINT_SUCCESS();

	do {
		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clEnqueueWriteBuffer(queueStencilkernel, dataK, CL_TRUE, 0, 271392 * sizeof(float), data, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (dataK)"));
		fRet = clEnqueueWriteBuffer(queueStencilkernel, newDataK, CL_TRUE, 0, 271392 * sizeof(float), newData, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (newDataK)"));
		fRet = clSetKernelArg(kernelStencilkernel, 2, sizeof(int), &alignment);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (alignment)"));
		fRet = clSetKernelArg(kernelStencilkernel, 3, sizeof(float), &wCenter);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (wCenter)"));
		fRet = clSetKernelArg(kernelStencilkernel, 4, sizeof(float), &wCardinal);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (wCardinal)"));
		fRet = clSetKernelArg(kernelStencilkernel, 5, sizeof(float), &wDiagonal);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (wDiagonal)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueStencilkernel, kernelStencilkernel, workDimStencilkernel, NULL, globalSizeStencilkernel, localSizeStencilkernel, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueStencilkernel);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueStencilkernel, newDataK, CL_TRUE, 0, 271392 * sizeof(float), newData, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueReadBuffer"));
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
	for(i = 0; i < 271392; i++) {
		if(TEST_EPSILON(newDataC[i],  newData[i], newDataEpsilon)) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable newData[%d]: expected %f got %f (with epsilon).\n", i, newDataC[i], newData[i]);
		}
	}
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(dataK)
		clReleaseMemObject(dataK);
	if(newDataK)
		clReleaseMemObject(newDataK);

	/* Dealloc variables */
	free(data);
	free(newData);
	free(newDataC);

	/* Dealloc kernels */
	if(kernelStencilkernel)
		clReleaseKernel(kernelStencilkernel);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueStencilkernel)
		clReleaseCommandQueue(queueStencilkernel);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);


	return rv;
}
