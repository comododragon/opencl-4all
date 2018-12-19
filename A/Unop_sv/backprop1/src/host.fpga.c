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
 *            PREAMBLE(input_cuda, input_cudaSz, input_hidden_cuda, input_hidden_cudaSz, hidden_partial_sum, hidden_partial_sumSz, hidden_partial_sumC, hidden_partial_sumCSz, in, hid);
 *            POSTAMBLE(input_cuda, input_cudaSz, input_hidden_cuda, input_hidden_cudaSz, hidden_partial_sum, hidden_partial_sumSz, hidden_partial_sumC, hidden_partial_sumCSz, in, hid);
 *            LOOPPREAMBLE(input_cuda, input_cudaSz, input_hidden_cuda, input_hidden_cudaSz, hidden_partial_sum, hidden_partial_sumSz, hidden_partial_sumC, hidden_partial_sumCSz, in, hid, loopFlag);
 *            LOOPPOSTAMBLE(input_cuda, input_cudaSz, input_hidden_cuda, input_hidden_cudaSz, hidden_partial_sum, hidden_partial_sumSz, hidden_partial_sumC, hidden_partial_sumCSz, in, hid, loopFlag);
 *            CLEANUP(input_cuda, input_cudaSz, input_hidden_cuda, input_hidden_cudaSz, hidden_partial_sum, hidden_partial_sumSz, hidden_partial_sumC, hidden_partial_sumCSz, in, hid);
 *        where:
 *            input_cuda: variable (float *);
 *            input_cudaSz: number of members in variable (unsigned int);
 *            input_hidden_cuda: variable (float *);
 *            input_hidden_cudaSz: number of members in variable (unsigned int);
 *            hidden_partial_sum: variable (float *);
 *            hidden_partial_sumSz: number of members in variable (unsigned int);
 *            hidden_partial_sumC: variable (float *);
 *            hidden_partial_sumCSz: number of members in variable (unsigned int);
 *            in: variable (int);
 *            hid: variable (int);
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
	cl_command_queue queueBpnn_Layerforward_Ocl = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelBpnn_Layerforward_Ocl = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimBpnn_Layerforward_Ocl = 3;
	size_t globalSizeBpnn_Layerforward_Ocl[3] = {
		16, 65536, 1
	};
	size_t localSizeBpnn_Layerforward_Ocl[3] = {
		16, 16, 1
	};

	/* Input/output variables */
	float *input_cuda = malloc(65537 * sizeof(float));
	cl_mem input_cudaK = NULL;
	float *input_hidden_cuda = malloc(1114129 * sizeof(float));
	cl_mem input_hidden_cudaK = NULL;
	float *hidden_partial_sum = malloc(65536 * sizeof(float));
	float *hidden_partial_sumC = malloc(65536 * sizeof(float));
	double hidden_partial_sumEpsilon = 0.001;
	cl_mem hidden_partial_sumK = NULL;
	int in = 65536;
	int hid = 16;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(input_cuda, 65537, input_hidden_cuda, 1114129, hidden_partial_sum, 65536, hidden_partial_sumC, 65536, in, hid);
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

	/* Create command queue for bpnn_layerforward_ocl kernel */
	PRINT_STEP("Creating command queue for \"bpnn_layerforward_ocl\"...");
	queueBpnn_Layerforward_Ocl = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create bpnn_layerforward_ocl kernel */
	PRINT_STEP("Creating kernel \"bpnn_layerforward_ocl\" from program...");
	kernelBpnn_Layerforward_Ocl = clCreateKernel(program, "bpnn_layerforward_ocl", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	input_cudaK = clCreateBuffer(context, CL_MEM_READ_ONLY, 65537 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (input_cudaK)"));
	input_hidden_cudaK = clCreateBuffer(context, CL_MEM_READ_ONLY, 1114129 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (input_hidden_cudaK)"));
	hidden_partial_sumK = clCreateBuffer(context, CL_MEM_READ_WRITE, 65536 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (hidden_partial_sumK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for bpnn_layerforward_ocl */
	PRINT_STEP("Setting kernel arguments for \"bpnn_layerforward_ocl\"...");
	fRet = clSetKernelArg(kernelBpnn_Layerforward_Ocl, 0, sizeof(cl_mem), &input_cudaK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (input_cudaK)"));
	fRet = clSetKernelArg(kernelBpnn_Layerforward_Ocl, 1, sizeof(cl_mem), &input_hidden_cudaK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (input_hidden_cudaK)"));
	fRet = clSetKernelArg(kernelBpnn_Layerforward_Ocl, 2, sizeof(cl_mem), &hidden_partial_sumK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (hidden_partial_sumK)"));
	fRet = clSetKernelArg(kernelBpnn_Layerforward_Ocl, 3, 16 * sizeof(float), NULL);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (__local 3)"));
	fRet = clSetKernelArg(kernelBpnn_Layerforward_Ocl, 4, 256 * sizeof(float), NULL);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (__local 4)"));
	fRet = clSetKernelArg(kernelBpnn_Layerforward_Ocl, 5, sizeof(int), &in);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (in)"));
	fRet = clSetKernelArg(kernelBpnn_Layerforward_Ocl, 6, sizeof(int), &hid);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (hid)"));
	PRINT_SUCCESS();

	do {
		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clEnqueueWriteBuffer(queueBpnn_Layerforward_Ocl, input_cudaK, CL_TRUE, 0, 65537 * sizeof(float), input_cuda, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (input_cudaK)"));
		fRet = clEnqueueWriteBuffer(queueBpnn_Layerforward_Ocl, input_hidden_cudaK, CL_TRUE, 0, 1114129 * sizeof(float), input_hidden_cuda, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (input_hidden_cudaK)"));
		fRet = clEnqueueWriteBuffer(queueBpnn_Layerforward_Ocl, hidden_partial_sumK, CL_TRUE, 0, 65536 * sizeof(float), hidden_partial_sum, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (hidden_partial_sumK)"));
		fRet = clSetKernelArg(kernelBpnn_Layerforward_Ocl, 5, sizeof(int), &in);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (in)"));
		fRet = clSetKernelArg(kernelBpnn_Layerforward_Ocl, 6, sizeof(int), &hid);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (hid)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueBpnn_Layerforward_Ocl, kernelBpnn_Layerforward_Ocl, workDimBpnn_Layerforward_Ocl, NULL, globalSizeBpnn_Layerforward_Ocl, localSizeBpnn_Layerforward_Ocl, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueBpnn_Layerforward_Ocl);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueBpnn_Layerforward_Ocl, hidden_partial_sumK, CL_TRUE, 0, 65536 * sizeof(float), hidden_partial_sum, 0, NULL, NULL);
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
	for(i = 0; i < 65536; i++) {
		if(TEST_EPSILON(hidden_partial_sumC[i],  hidden_partial_sum[i], hidden_partial_sumEpsilon)) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable hidden_partial_sum[%d]: expected %f got %f (with epsilon).\n", i, hidden_partial_sumC[i], hidden_partial_sum[i]);
		}
	}
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(input_cudaK)
		clReleaseMemObject(input_cudaK);
	if(input_hidden_cudaK)
		clReleaseMemObject(input_hidden_cudaK);
	if(hidden_partial_sumK)
		clReleaseMemObject(hidden_partial_sumK);

	/* Dealloc variables */
	free(input_cuda);
	free(input_hidden_cuda);
	free(hidden_partial_sum);
	free(hidden_partial_sumC);

	/* Dealloc kernels */
	if(kernelBpnn_Layerforward_Ocl)
		clReleaseKernel(kernelBpnn_Layerforward_Ocl);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueBpnn_Layerforward_Ocl)
		clReleaseCommandQueue(queueBpnn_Layerforward_Ocl);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);

	/* Calling cleanup function */
	CLEANUP(input_cuda, 65537, input_hidden_cuda, 1114129, hidden_partial_sum, 65536, hidden_partial_sumC, 65536, in, hid);

	return rv;
}
