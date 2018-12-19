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
 *            PREAMBLE(d_Result, d_ResultSz, d_ResultC, d_ResultCSz, d_Data, d_DataSz, minimum, maximum, dataCount);
 *            POSTAMBLE(d_Result, d_ResultSz, d_ResultC, d_ResultCSz, d_Data, d_DataSz, minimum, maximum, dataCount);
 *            LOOPPREAMBLE(d_Result, d_ResultSz, d_ResultC, d_ResultCSz, d_Data, d_DataSz, minimum, maximum, dataCount, loopFlag);
 *            LOOPPOSTAMBLE(d_Result, d_ResultSz, d_ResultC, d_ResultCSz, d_Data, d_DataSz, minimum, maximum, dataCount, loopFlag);
 *            CLEANUP(d_Result, d_ResultSz, d_ResultC, d_ResultCSz, d_Data, d_DataSz, minimum, maximum, dataCount);
 *        where:
 *            d_Result: variable (unsigned int *);
 *            d_ResultSz: number of members in variable (unsigned int);
 *            d_ResultC: variable (unsigned int *);
 *            d_ResultCSz: number of members in variable (unsigned int);
 *            d_Data: variable (float *);
 *            d_DataSz: number of members in variable (unsigned int);
 *            minimum: variable (float);
 *            maximum: variable (float);
 *            dataCount: variable (unsigned int);
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
	cl_command_queue queueHistogram1024Kernel = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelHistogram1024Kernel = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimHistogram1024Kernel = 1;
	size_t globalSizeHistogram1024Kernel[1] = {
		6144
	};
	size_t localSizeHistogram1024Kernel[1] = {
		96
	};

	/* Input/output variables */
	unsigned int *d_Result = malloc(1024 * sizeof(unsigned int));
	unsigned int *d_ResultC = malloc(1024 * sizeof(unsigned int));
	double d_ResultEpsilon = 50;
	cl_mem d_ResultK = NULL;
	float *d_Data = malloc(1000000 * sizeof(float));
	cl_mem d_DataK = NULL;
	float minimum = 0.000001;
	float maximum = 0.999998;
	unsigned int dataCount = 1000000;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(d_Result, 1024, d_ResultC, 1024, d_Data, 1000000, minimum, maximum, dataCount);
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

	/* Create command queue for histogram1024Kernel kernel */
	PRINT_STEP("Creating command queue for \"histogram1024Kernel\"...");
	queueHistogram1024Kernel = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create histogram1024Kernel kernel */
	PRINT_STEP("Creating kernel \"histogram1024Kernel\" from program...");
	kernelHistogram1024Kernel = clCreateKernel(program, "histogram1024Kernel", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	d_ResultK = clCreateBuffer(context, CL_MEM_READ_WRITE, 1024 * sizeof(unsigned int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (d_ResultK)"));
	d_DataK = clCreateBuffer(context, CL_MEM_READ_ONLY, 1000000 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (d_DataK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for histogram1024Kernel */
	PRINT_STEP("Setting kernel arguments for \"histogram1024Kernel\"...");
	fRet = clSetKernelArg(kernelHistogram1024Kernel, 0, sizeof(cl_mem), &d_ResultK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_ResultK)"));
	fRet = clSetKernelArg(kernelHistogram1024Kernel, 1, sizeof(cl_mem), &d_DataK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_DataK)"));
	fRet = clSetKernelArg(kernelHistogram1024Kernel, 2, sizeof(float), &minimum);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (minimum)"));
	fRet = clSetKernelArg(kernelHistogram1024Kernel, 3, sizeof(float), &maximum);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (maximum)"));
	fRet = clSetKernelArg(kernelHistogram1024Kernel, 4, sizeof(unsigned int), &dataCount);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (dataCount)"));
	PRINT_SUCCESS();

	do {
		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clEnqueueWriteBuffer(queueHistogram1024Kernel, d_ResultK, CL_TRUE, 0, 1024 * sizeof(unsigned int), d_Result, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (d_ResultK)"));
		fRet = clEnqueueWriteBuffer(queueHistogram1024Kernel, d_DataK, CL_TRUE, 0, 1000000 * sizeof(float), d_Data, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (d_DataK)"));
		fRet = clSetKernelArg(kernelHistogram1024Kernel, 2, sizeof(float), &minimum);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (minimum)"));
		fRet = clSetKernelArg(kernelHistogram1024Kernel, 3, sizeof(float), &maximum);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (maximum)"));
		fRet = clSetKernelArg(kernelHistogram1024Kernel, 4, sizeof(unsigned int), &dataCount);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (dataCount)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueHistogram1024Kernel, kernelHistogram1024Kernel, workDimHistogram1024Kernel, NULL, globalSizeHistogram1024Kernel, localSizeHistogram1024Kernel, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueHistogram1024Kernel);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueHistogram1024Kernel, d_ResultK, CL_TRUE, 0, 1024 * sizeof(unsigned int), d_Result, 0, NULL, NULL);
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
	for(i = 0; i < 1024; i++) {
		if(TEST_EPSILON(d_ResultC[i],  d_Result[i], d_ResultEpsilon)) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable d_Result[%d]: expected %u got %u (with epsilon).\n", i, d_ResultC[i], d_Result[i]);
		}
	}
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(d_ResultK)
		clReleaseMemObject(d_ResultK);
	if(d_DataK)
		clReleaseMemObject(d_DataK);

	/* Dealloc variables */
	free(d_Result);
	free(d_ResultC);
	free(d_Data);

	/* Dealloc kernels */
	if(kernelHistogram1024Kernel)
		clReleaseKernel(kernelHistogram1024Kernel);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueHistogram1024Kernel)
		clReleaseCommandQueue(queueHistogram1024Kernel);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);


	return rv;
}
