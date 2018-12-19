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
 *            PREAMBLE(input, inputSz, indice, indiceSz, output, outputSz, outputC, outputCSz, size, d_prefixoffsets, d_prefixoffsetsSz, l_offsets, l_offsetsSz);
 *            POSTAMBLE(input, inputSz, indice, indiceSz, output, outputSz, outputC, outputCSz, size, d_prefixoffsets, d_prefixoffsetsSz, l_offsets, l_offsetsSz);
 *            LOOPPREAMBLE(input, inputSz, indice, indiceSz, output, outputSz, outputC, outputCSz, size, d_prefixoffsets, d_prefixoffsetsSz, l_offsets, l_offsetsSz, loopFlag);
 *            LOOPPOSTAMBLE(input, inputSz, indice, indiceSz, output, outputSz, outputC, outputCSz, size, d_prefixoffsets, d_prefixoffsetsSz, l_offsets, l_offsetsSz, loopFlag);
 *            CLEANUP(input, inputSz, indice, indiceSz, output, outputSz, outputC, outputCSz, size, d_prefixoffsets, d_prefixoffsetsSz, l_offsets, l_offsetsSz);
 *        where:
 *            input: variable (float *);
 *            inputSz: number of members in variable (unsigned int);
 *            indice: variable (int *);
 *            indiceSz: number of members in variable (unsigned int);
 *            output: variable (float *);
 *            outputSz: number of members in variable (unsigned int);
 *            outputC: variable (float *);
 *            outputCSz: number of members in variable (unsigned int);
 *            size: variable (int);
 *            d_prefixoffsets: variable (unsigned int *);
 *            d_prefixoffsetsSz: number of members in variable (unsigned int);
 *            l_offsets: variable (unsigned int *);
 *            l_offsetsSz: number of members in variable (unsigned int);
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
	cl_command_queue queueBucketsort = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelBucketsort = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimBucketsort = 3;
	size_t globalSizeBucketsort[3] = {
		7840, 1, 1
	};
	size_t localSizeBucketsort[3] = {
		32, 1, 1
	};

	/* Input/output variables */
	float *input = malloc(1004096 * sizeof(float));
	cl_mem inputK = NULL;
	int *indice = malloc(1000000 * sizeof(int));
	cl_mem indiceK = NULL;
	float *output = malloc(1004096 * sizeof(float));
	float *outputC = malloc(1004096 * sizeof(float));
	cl_mem outputK = NULL;
	int size = 1000000;
	unsigned int *d_prefixoffsets = malloc(250880 * sizeof(unsigned int));
	cl_mem d_prefixoffsetsK = NULL;
	unsigned int *l_offsets = malloc(1024 * sizeof(unsigned int));
	cl_mem l_offsetsK = NULL;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(input, 1004096, indice, 1000000, output, 1004096, outputC, 1004096, size, d_prefixoffsets, 250880, l_offsets, 1024);
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

	/* Create command queue for bucketsort kernel */
	PRINT_STEP("Creating command queue for \"bucketsort\"...");
	queueBucketsort = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create bucketsort kernel */
	PRINT_STEP("Creating kernel \"bucketsort\" from program...");
	kernelBucketsort = clCreateKernel(program, "bucketsort", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	inputK = clCreateBuffer(context, CL_MEM_READ_ONLY, 1004096 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (inputK)"));
	indiceK = clCreateBuffer(context, CL_MEM_READ_ONLY, 1000000 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (indiceK)"));
	outputK = clCreateBuffer(context, CL_MEM_READ_WRITE, 1004096 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (outputK)"));
	d_prefixoffsetsK = clCreateBuffer(context, CL_MEM_READ_ONLY, 250880 * sizeof(unsigned int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (d_prefixoffsetsK)"));
	l_offsetsK = clCreateBuffer(context, CL_MEM_READ_ONLY, 1024 * sizeof(unsigned int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (l_offsetsK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for bucketsort */
	PRINT_STEP("Setting kernel arguments for \"bucketsort\"...");
	fRet = clSetKernelArg(kernelBucketsort, 0, sizeof(cl_mem), &inputK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (inputK)"));
	fRet = clSetKernelArg(kernelBucketsort, 1, sizeof(cl_mem), &indiceK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (indiceK)"));
	fRet = clSetKernelArg(kernelBucketsort, 2, sizeof(cl_mem), &outputK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (outputK)"));
	fRet = clSetKernelArg(kernelBucketsort, 3, sizeof(int), &size);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (size)"));
	fRet = clSetKernelArg(kernelBucketsort, 4, sizeof(cl_mem), &d_prefixoffsetsK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_prefixoffsetsK)"));
	fRet = clSetKernelArg(kernelBucketsort, 5, sizeof(cl_mem), &l_offsetsK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (l_offsetsK)"));
	PRINT_SUCCESS();

	do {
		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clEnqueueWriteBuffer(queueBucketsort, inputK, CL_TRUE, 0, 1004096 * sizeof(float), input, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (inputK)"));
		fRet = clEnqueueWriteBuffer(queueBucketsort, indiceK, CL_TRUE, 0, 1000000 * sizeof(int), indice, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (indiceK)"));
		fRet = clEnqueueWriteBuffer(queueBucketsort, outputK, CL_TRUE, 0, 1004096 * sizeof(float), output, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (outputK)"));
		fRet = clSetKernelArg(kernelBucketsort, 3, sizeof(int), &size);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (size)"));
		fRet = clEnqueueWriteBuffer(queueBucketsort, d_prefixoffsetsK, CL_TRUE, 0, 250880 * sizeof(unsigned int), d_prefixoffsets, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (d_prefixoffsetsK)"));
		fRet = clEnqueueWriteBuffer(queueBucketsort, l_offsetsK, CL_TRUE, 0, 1024 * sizeof(unsigned int), l_offsets, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (l_offsetsK)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueBucketsort, kernelBucketsort, workDimBucketsort, NULL, globalSizeBucketsort, localSizeBucketsort, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueBucketsort);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueBucketsort, outputK, CL_TRUE, 0, 1004096 * sizeof(float), output, 0, NULL, NULL);
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
	for(i = 0; i < 1004096; i++) {
		if(outputC[i] != output[i]) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable output[%d]: expected %f got %f.\n", i, outputC[i], output[i]);
		}
	}
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(inputK)
		clReleaseMemObject(inputK);
	if(indiceK)
		clReleaseMemObject(indiceK);
	if(outputK)
		clReleaseMemObject(outputK);
	if(d_prefixoffsetsK)
		clReleaseMemObject(d_prefixoffsetsK);
	if(l_offsetsK)
		clReleaseMemObject(l_offsetsK);

	/* Dealloc variables */
	free(input);
	free(indice);
	free(output);
	free(outputC);
	free(d_prefixoffsets);
	free(l_offsets);

	/* Dealloc kernels */
	if(kernelBucketsort)
		clReleaseKernel(kernelBucketsort);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueBucketsort)
		clReleaseCommandQueue(queueBucketsort);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);


	return rv;
}
