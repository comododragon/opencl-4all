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
 *            PREAMBLE(delta, deltaSz, hid, ly, lySz, lyC, lyCSz, in, w, wSz, wC, wCSz, oldw, oldwSz);
 *            POSTAMBLE(delta, deltaSz, hid, ly, lySz, lyC, lyCSz, in, w, wSz, wC, wCSz, oldw, oldwSz);
 *            LOOPPREAMBLE(delta, deltaSz, hid, ly, lySz, lyC, lyCSz, in, w, wSz, wC, wCSz, oldw, oldwSz, loopFlag);
 *            LOOPPOSTAMBLE(delta, deltaSz, hid, ly, lySz, lyC, lyCSz, in, w, wSz, wC, wCSz, oldw, oldwSz, loopFlag);
 *            CLEANUP(delta, deltaSz, hid, ly, lySz, lyC, lyCSz, in, w, wSz, wC, wCSz, oldw, oldwSz);
 *        where:
 *            delta: variable (float *);
 *            deltaSz: number of members in variable (unsigned int);
 *            hid: variable (int);
 *            ly: variable (float *);
 *            lySz: number of members in variable (unsigned int);
 *            lyC: variable (float *);
 *            lyCSz: number of members in variable (unsigned int);
 *            in: variable (int);
 *            w: variable (float *);
 *            wSz: number of members in variable (unsigned int);
 *            wC: variable (float *);
 *            wCSz: number of members in variable (unsigned int);
 *            oldw: variable (float *);
 *            oldwSz: number of members in variable (unsigned int);
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
	cl_command_queue queueBpnn_Adjust_Weights_Ocl = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelBpnn_Adjust_Weights_Ocl = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimBpnn_Adjust_Weights_Ocl = 3;
	size_t globalSizeBpnn_Adjust_Weights_Ocl[3] = {
		16, 65536, 1
	};
	size_t localSizeBpnn_Adjust_Weights_Ocl[3] = {
		16, 16, 1
	};

	/* Input/output variables */
	float *delta = malloc(17 * sizeof(float));
	cl_mem deltaK = NULL;
	int hid = 16;
	float *ly = malloc(65537 * sizeof(float));
	float *lyC = malloc(65537 * sizeof(float));
	cl_mem lyK = NULL;
	int in = 65536;
	float *w = malloc(1114129 * sizeof(float));
	float *wC = malloc(1114129 * sizeof(float));
	double wEpsilon = 0.001;
	cl_mem wK = NULL;
	float *oldw = malloc(1114129 * sizeof(float));
	cl_mem oldwK = NULL;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(delta, 17, hid, ly, 65537, lyC, 65537, in, w, 1114129, wC, 1114129, oldw, 1114129);
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

	/* Create command queue for bpnn_adjust_weights_ocl kernel */
	PRINT_STEP("Creating command queue for \"bpnn_adjust_weights_ocl\"...");
	queueBpnn_Adjust_Weights_Ocl = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create bpnn_adjust_weights_ocl kernel */
	PRINT_STEP("Creating kernel \"bpnn_adjust_weights_ocl\" from program...");
	kernelBpnn_Adjust_Weights_Ocl = clCreateKernel(program, "bpnn_adjust_weights_ocl", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	deltaK = clCreateBuffer(context, CL_MEM_READ_ONLY, 17 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (deltaK)"));
	lyK = clCreateBuffer(context, CL_MEM_READ_WRITE, 65537 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (lyK)"));
	wK = clCreateBuffer(context, CL_MEM_READ_WRITE, 1114129 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (wK)"));
	oldwK = clCreateBuffer(context, CL_MEM_READ_WRITE, 1114129 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (oldwK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for bpnn_adjust_weights_ocl */
	PRINT_STEP("Setting kernel arguments for \"bpnn_adjust_weights_ocl\"...");
	fRet = clSetKernelArg(kernelBpnn_Adjust_Weights_Ocl, 0, sizeof(cl_mem), &deltaK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (deltaK)"));
	fRet = clSetKernelArg(kernelBpnn_Adjust_Weights_Ocl, 1, sizeof(int), &hid);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (hid)"));
	fRet = clSetKernelArg(kernelBpnn_Adjust_Weights_Ocl, 2, sizeof(cl_mem), &lyK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (lyK)"));
	fRet = clSetKernelArg(kernelBpnn_Adjust_Weights_Ocl, 3, sizeof(int), &in);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (in)"));
	fRet = clSetKernelArg(kernelBpnn_Adjust_Weights_Ocl, 4, sizeof(cl_mem), &wK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (wK)"));
	fRet = clSetKernelArg(kernelBpnn_Adjust_Weights_Ocl, 5, sizeof(cl_mem), &oldwK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (oldwK)"));
	PRINT_SUCCESS();

	do {
		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clEnqueueWriteBuffer(queueBpnn_Adjust_Weights_Ocl, deltaK, CL_TRUE, 0, 17 * sizeof(float), delta, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (deltaK)"));
		fRet = clSetKernelArg(kernelBpnn_Adjust_Weights_Ocl, 1, sizeof(int), &hid);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (hid)"));
		fRet = clEnqueueWriteBuffer(queueBpnn_Adjust_Weights_Ocl, lyK, CL_TRUE, 0, 65537 * sizeof(float), ly, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (lyK)"));
		fRet = clSetKernelArg(kernelBpnn_Adjust_Weights_Ocl, 3, sizeof(int), &in);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (in)"));
		fRet = clEnqueueWriteBuffer(queueBpnn_Adjust_Weights_Ocl, wK, CL_TRUE, 0, 1114129 * sizeof(float), w, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (wK)"));
		fRet = clEnqueueWriteBuffer(queueBpnn_Adjust_Weights_Ocl, oldwK, CL_TRUE, 0, 1114129 * sizeof(float), oldw, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (oldwK)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueBpnn_Adjust_Weights_Ocl, kernelBpnn_Adjust_Weights_Ocl, workDimBpnn_Adjust_Weights_Ocl, NULL, globalSizeBpnn_Adjust_Weights_Ocl, localSizeBpnn_Adjust_Weights_Ocl, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueBpnn_Adjust_Weights_Ocl);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueBpnn_Adjust_Weights_Ocl, lyK, CL_TRUE, 0, 65537 * sizeof(float), ly, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueBpnn_Adjust_Weights_Ocl, wK, CL_TRUE, 0, 1114129 * sizeof(float), w, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueBpnn_Adjust_Weights_Ocl, oldwK, CL_TRUE, 0, 1114129 * sizeof(float), oldw, 0, NULL, NULL);
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
	for(i = 0; i < 65537; i++) {
		if(lyC[i] != ly[i]) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable ly[%d]: expected %f got %f.\n", i, lyC[i], ly[i]);
		}
	}
	for(i = 0; i < 1114129; i++) {
		if(TEST_EPSILON(wC[i],  w[i], wEpsilon)) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable w[%d]: expected %f got %f (with epsilon).\n", i, wC[i], w[i]);
		}
	}
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(deltaK)
		clReleaseMemObject(deltaK);
	if(lyK)
		clReleaseMemObject(lyK);
	if(wK)
		clReleaseMemObject(wK);
	if(oldwK)
		clReleaseMemObject(oldwK);

	/* Dealloc variables */
	free(delta);
	free(ly);
	free(lyC);
	free(w);
	free(wC);
	free(oldw);

	/* Dealloc kernels */
	if(kernelBpnn_Adjust_Weights_Ocl)
		clReleaseKernel(kernelBpnn_Adjust_Weights_Ocl);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueBpnn_Adjust_Weights_Ocl)
		clReleaseCommandQueue(queueBpnn_Adjust_Weights_Ocl);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);


	return rv;
}
