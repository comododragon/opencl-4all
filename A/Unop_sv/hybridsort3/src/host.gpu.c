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
 *            PREAMBLE(input, inputSz, result, resultSz, resultC, resultCSz, nrElems, threadsPerDiv, constStartAddr, constStartAddrSz);
 *            POSTAMBLE(input, inputSz, result, resultSz, resultC, resultCSz, nrElems, threadsPerDiv, constStartAddr, constStartAddrSz);
 *            LOOPPREAMBLE(input, inputSz, result, resultSz, resultC, resultCSz, nrElems, threadsPerDiv, constStartAddr, constStartAddrSz, loopFlag);
 *            LOOPPOSTAMBLE(input, inputSz, result, resultSz, resultC, resultCSz, nrElems, threadsPerDiv, constStartAddr, constStartAddrSz, loopFlag);
 *            CLEANUP(input, inputSz, result, resultSz, resultC, resultCSz, nrElems, threadsPerDiv, constStartAddr, constStartAddrSz);
 *        where:
 *            input: variable (cl_float4 *);
 *            inputSz: number of members in variable (unsigned int);
 *            result: variable (cl_float4 *);
 *            resultSz: number of members in variable (unsigned int);
 *            resultC: variable (cl_float4 *);
 *            resultCSz: number of members in variable (unsigned int);
 *            nrElems: variable (int);
 *            threadsPerDiv: variable (int);
 *            constStartAddr: variable (int *);
 *            constStartAddrSz: number of members in variable (unsigned int);
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
	cl_command_queue queueMergesortpass = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelMergesortpass = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimMergesortpass = 3;
	size_t globalSizeMergesortpass[3] = {
		248976, 1, 1
	};
	size_t localSizeMergesortpass[3] = {
		208, 1, 1
	};

	/* Input/output variables */
	cl_float4 *input = malloc(250383 * sizeof(cl_float4));
	cl_mem inputK = NULL;
	cl_float4 *result = malloc(250383 * sizeof(cl_float4));
	cl_float4 *resultC = malloc(250383 * sizeof(cl_float4));
	cl_mem resultK = NULL;
	int nrElems = 2;
	int threadsPerDiv = 243;
	int *constStartAddr = malloc(1025 * sizeof(int));
	cl_mem constStartAddrK = NULL;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(input, 250383, result, 250383, resultC, 250383, nrElems, threadsPerDiv, constStartAddr, 1025);
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

	/* Create command queue for mergeSortPass kernel */
	PRINT_STEP("Creating command queue for \"mergeSortPass\"...");
	queueMergesortpass = clCreateCommandQueue(context, devices[0], 0, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateCommandQueue"));
	PRINT_SUCCESS();

	/* Open binary file */
	PRINT_STEP("Opening program binary...");
	programFile = fopen("kern.cl", "rb");
	ASSERT_CALL(programFile, POSIX_ERROR_STATEMENTS("kernel.cl"));
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

	/* Create mergeSortPass kernel */
	PRINT_STEP("Creating kernel \"mergeSortPass\" from program...");
	kernelMergesortpass = clCreateKernel(program, "mergeSortPass", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	inputK = clCreateBuffer(context, CL_MEM_READ_ONLY, 250383 * sizeof(cl_float4), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (inputK)"));
	resultK = clCreateBuffer(context, CL_MEM_READ_WRITE, 250383 * sizeof(cl_float4), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (resultK)"));
	constStartAddrK = clCreateBuffer(context, CL_MEM_READ_ONLY, 1025 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (constStartAddrK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for mergeSortPass */
	PRINT_STEP("Setting kernel arguments for \"mergeSortPass\"...");
	fRet = clSetKernelArg(kernelMergesortpass, 0, sizeof(cl_mem), &inputK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (inputK)"));
	fRet = clSetKernelArg(kernelMergesortpass, 1, sizeof(cl_mem), &resultK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (resultK)"));
	fRet = clSetKernelArg(kernelMergesortpass, 2, sizeof(int), &nrElems);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (nrElems)"));
	fRet = clSetKernelArg(kernelMergesortpass, 3, sizeof(int), &threadsPerDiv);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (threadsPerDiv)"));
	fRet = clSetKernelArg(kernelMergesortpass, 4, sizeof(cl_mem), &constStartAddrK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (constStartAddrK)"));
	PRINT_SUCCESS();

	do {
		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clEnqueueWriteBuffer(queueMergesortpass, inputK, CL_TRUE, 0, 250383 * sizeof(cl_float4), input, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (inputK)"));
		fRet = clEnqueueWriteBuffer(queueMergesortpass, resultK, CL_TRUE, 0, 250383 * sizeof(cl_float4), result, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (resultK)"));
		fRet = clSetKernelArg(kernelMergesortpass, 2, sizeof(int), &nrElems);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (nrElems)"));
		fRet = clSetKernelArg(kernelMergesortpass, 3, sizeof(int), &threadsPerDiv);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (threadsPerDiv)"));
		fRet = clEnqueueWriteBuffer(queueMergesortpass, constStartAddrK, CL_TRUE, 0, 1025 * sizeof(int), constStartAddr, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (constStartAddrK)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueMergesortpass, kernelMergesortpass, workDimMergesortpass, NULL, globalSizeMergesortpass, localSizeMergesortpass, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueMergesortpass);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueMergesortpass, resultK, CL_TRUE, 0, 250383 * sizeof(cl_float4), result, 0, NULL, NULL);
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
	for(i = 0; i < 250383; i++) {
		for(j = 0; j < 4; j++) {
			if(resultC[i].s[j] != result[i].s[j]) {
				if(!invalidDataFound) {
					PRINT_FAIL();
					invalidDataFound = true;
				}
				printf("Variable result[%d].s[%d]: expected %f got %f.\n", i, j, resultC[i].s[j], result[i].s[j]);
			}
		}
	}
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(inputK)
		clReleaseMemObject(inputK);
	if(resultK)
		clReleaseMemObject(resultK);
	if(constStartAddrK)
		clReleaseMemObject(constStartAddrK);

	/* Dealloc variables */
	free(input);
	free(result);
	free(resultC);
	free(constStartAddr);

	/* Dealloc kernels */
	if(kernelMergesortpass)
		clReleaseKernel(kernelMergesortpass);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueMergesortpass)
		clReleaseCommandQueue(queueMergesortpass);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);


	return rv;
}
