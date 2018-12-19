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
 *            PREAMBLE(searchDigest0, searchDigest1, searchDigest2, searchDigest3, keyspace, byteLength, valsPerByte, foundIndex, foundIndexC, foundKey, foundKeySz, foundKeyC, foundKeyCSz, foundDigest, foundDigestSz, foundDigestC, foundDigestCSz);
 *            POSTAMBLE(searchDigest0, searchDigest1, searchDigest2, searchDigest3, keyspace, byteLength, valsPerByte, foundIndex, foundIndexC, foundKey, foundKeySz, foundKeyC, foundKeyCSz, foundDigest, foundDigestSz, foundDigestC, foundDigestCSz);
 *            LOOPPREAMBLE(searchDigest0, searchDigest1, searchDigest2, searchDigest3, keyspace, byteLength, valsPerByte, foundIndex, foundIndexC, foundKey, foundKeySz, foundKeyC, foundKeyCSz, foundDigest, foundDigestSz, foundDigestC, foundDigestCSz, loopFlag);
 *            LOOPPOSTAMBLE(searchDigest0, searchDigest1, searchDigest2, searchDigest3, keyspace, byteLength, valsPerByte, foundIndex, foundIndexC, foundKey, foundKeySz, foundKeyC, foundKeyCSz, foundDigest, foundDigestSz, foundDigestC, foundDigestCSz, loopFlag);
 *            CLEANUP(searchDigest0, searchDigest1, searchDigest2, searchDigest3, keyspace, byteLength, valsPerByte, foundIndex, foundIndexC, foundKey, foundKeySz, foundKeyC, foundKeyCSz, foundDigest, foundDigestSz, foundDigestC, foundDigestCSz);
 *        where:
 *            searchDigest0: variable (unsigned int);
 *            searchDigest1: variable (unsigned int);
 *            searchDigest2: variable (unsigned int);
 *            searchDigest3: variable (unsigned int);
 *            keyspace: variable (int);
 *            byteLength: variable (int);
 *            valsPerByte: variable (int);
 *            foundIndex: variable (int *);
 *            foundIndexC: variable (int *);
 *            foundKey: variable (unsigned char *);
 *            foundKeySz: number of members in variable (unsigned int);
 *            foundKeyC: variable (unsigned char *);
 *            foundKeyCSz: number of members in variable (unsigned int);
 *            foundDigest: variable (unsigned int *);
 *            foundDigestSz: number of members in variable (unsigned int);
 *            foundDigestC: variable (unsigned int *);
 *            foundDigestCSz: number of members in variable (unsigned int);
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
	cl_command_queue queueFindkeywithdigest_Kernel = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelFindkeywithdigest_Kernel = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimFindkeywithdigest_Kernel = 1;
	size_t globalSizeFindkeywithdigest_Kernel[1] = {
		1000192
	};
	size_t localSizeFindkeywithdigest_Kernel[1] = {
		256
	};

	/* Input/output variables */
	unsigned int searchDigest0;
	unsigned int searchDigest1;
	unsigned int searchDigest2;
	unsigned int searchDigest3;
	int keyspace;
	int byteLength = 7;
	int valsPerByte = 10;
	int *foundIndex = malloc(1 * sizeof(int));
	int *foundIndexC = malloc(1 * sizeof(int));
	cl_mem foundIndexK = NULL;
	unsigned char *foundKey = malloc(8 * sizeof(unsigned char));
	unsigned char *foundKeyC = malloc(8 * sizeof(unsigned char));
	cl_mem foundKeyK = NULL;
	unsigned int *foundDigest = malloc(4 * sizeof(unsigned int));
	unsigned int *foundDigestC = malloc(4 * sizeof(unsigned int));
	cl_mem foundDigestK = NULL;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(searchDigest0, searchDigest1, searchDigest2, searchDigest3, keyspace, byteLength, valsPerByte, foundIndex, foundIndexC, foundKey, 8, foundKeyC, 8, foundDigest, 4, foundDigestC, 4);
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

	/* Create command queue for FindKeyWithDigest_Kernel kernel */
	PRINT_STEP("Creating command queue for \"FindKeyWithDigest_Kernel\"...");
	queueFindkeywithdigest_Kernel = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create FindKeyWithDigest_Kernel kernel */
	PRINT_STEP("Creating kernel \"FindKeyWithDigest_Kernel\" from program...");
	kernelFindkeywithdigest_Kernel = clCreateKernel(program, "FindKeyWithDigest_Kernel", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	foundIndexK = clCreateBuffer(context, CL_MEM_READ_WRITE, 1 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (foundIndexK)"));
	foundKeyK = clCreateBuffer(context, CL_MEM_READ_WRITE, 8 * sizeof(unsigned char), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (foundKeyK)"));
	foundDigestK = clCreateBuffer(context, CL_MEM_READ_WRITE, 4 * sizeof(unsigned int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (foundDigestK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for FindKeyWithDigest_Kernel */
	PRINT_STEP("Setting kernel arguments for \"FindKeyWithDigest_Kernel\"...");
	fRet = clSetKernelArg(kernelFindkeywithdigest_Kernel, 0, sizeof(unsigned int), &searchDigest0);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (searchDigest0)"));
	fRet = clSetKernelArg(kernelFindkeywithdigest_Kernel, 1, sizeof(unsigned int), &searchDigest1);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (searchDigest1)"));
	fRet = clSetKernelArg(kernelFindkeywithdigest_Kernel, 2, sizeof(unsigned int), &searchDigest2);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (searchDigest2)"));
	fRet = clSetKernelArg(kernelFindkeywithdigest_Kernel, 3, sizeof(unsigned int), &searchDigest3);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (searchDigest3)"));
	fRet = clSetKernelArg(kernelFindkeywithdigest_Kernel, 4, sizeof(int), &keyspace);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (keyspace)"));
	fRet = clSetKernelArg(kernelFindkeywithdigest_Kernel, 5, sizeof(int), &byteLength);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (byteLength)"));
	fRet = clSetKernelArg(kernelFindkeywithdigest_Kernel, 6, sizeof(int), &valsPerByte);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (valsPerByte)"));
	fRet = clSetKernelArg(kernelFindkeywithdigest_Kernel, 7, sizeof(cl_mem), &foundIndexK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (foundIndexK)"));
	fRet = clSetKernelArg(kernelFindkeywithdigest_Kernel, 8, sizeof(cl_mem), &foundKeyK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (foundKeyK)"));
	fRet = clSetKernelArg(kernelFindkeywithdigest_Kernel, 9, sizeof(cl_mem), &foundDigestK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (foundDigestK)"));
	PRINT_SUCCESS();

	do {
		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clSetKernelArg(kernelFindkeywithdigest_Kernel, 0, sizeof(unsigned int), &searchDigest0);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (searchDigest0)"));
		fRet = clSetKernelArg(kernelFindkeywithdigest_Kernel, 1, sizeof(unsigned int), &searchDigest1);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (searchDigest1)"));
		fRet = clSetKernelArg(kernelFindkeywithdigest_Kernel, 2, sizeof(unsigned int), &searchDigest2);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (searchDigest2)"));
		fRet = clSetKernelArg(kernelFindkeywithdigest_Kernel, 3, sizeof(unsigned int), &searchDigest3);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (searchDigest3)"));
		fRet = clSetKernelArg(kernelFindkeywithdigest_Kernel, 4, sizeof(int), &keyspace);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (keyspace)"));
		fRet = clSetKernelArg(kernelFindkeywithdigest_Kernel, 5, sizeof(int), &byteLength);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (byteLength)"));
		fRet = clSetKernelArg(kernelFindkeywithdigest_Kernel, 6, sizeof(int), &valsPerByte);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (valsPerByte)"));
		fRet = clEnqueueWriteBuffer(queueFindkeywithdigest_Kernel, foundIndexK, CL_TRUE, 0, 1 * sizeof(int), foundIndex, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (foundIndexK)"));
		fRet = clEnqueueWriteBuffer(queueFindkeywithdigest_Kernel, foundKeyK, CL_TRUE, 0, 8 * sizeof(unsigned char), foundKey, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (foundKeyK)"));
		fRet = clEnqueueWriteBuffer(queueFindkeywithdigest_Kernel, foundDigestK, CL_TRUE, 0, 4 * sizeof(unsigned int), foundDigest, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (foundDigestK)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueFindkeywithdigest_Kernel, kernelFindkeywithdigest_Kernel, workDimFindkeywithdigest_Kernel, NULL, globalSizeFindkeywithdigest_Kernel, localSizeFindkeywithdigest_Kernel, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueFindkeywithdigest_Kernel);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueFindkeywithdigest_Kernel, foundIndexK, CL_TRUE, 0, 1 * sizeof(int), foundIndex, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueFindkeywithdigest_Kernel, foundKeyK, CL_TRUE, 0, 8 * sizeof(unsigned char), foundKey, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueFindkeywithdigest_Kernel, foundDigestK, CL_TRUE, 0, 4 * sizeof(unsigned int), foundDigest, 0, NULL, NULL);
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
	for(i = 0; i < 1; i++) {
		if(foundIndexC[i] != foundIndex[i]) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable foundIndex[%d]: expected %d got %d.\n", i, foundIndexC[i], foundIndex[i]);
		}
	}
	for(i = 0; i < 8; i++) {
		if(foundKeyC[i] != foundKey[i]) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable foundKey[%d]: expected %c got %c.\n", i, foundKeyC[i], foundKey[i]);
		}
	}
	for(i = 0; i < 4; i++) {
		if(foundDigestC[i] != foundDigest[i]) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable foundDigest[%d]: expected %u got %u.\n", i, foundDigestC[i], foundDigest[i]);
		}
	}
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(foundIndexK)
		clReleaseMemObject(foundIndexK);
	if(foundKeyK)
		clReleaseMemObject(foundKeyK);
	if(foundDigestK)
		clReleaseMemObject(foundDigestK);

	/* Dealloc variables */
	free(foundIndex);
	free(foundIndexC);
	free(foundKey);
	free(foundKeyC);
	free(foundDigest);
	free(foundDigestC);

	/* Dealloc kernels */
	if(kernelFindkeywithdigest_Kernel)
		clReleaseKernel(kernelFindkeywithdigest_Kernel);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueFindkeywithdigest_Kernel)
		clReleaseCommandQueue(queueFindkeywithdigest_Kernel);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);


	return rv;
}
