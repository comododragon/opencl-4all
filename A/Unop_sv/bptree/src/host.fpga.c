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
 *            PREAMBLE(height, knodesDLocation, knodesDLocationSz, knodesDIndices, knodesDIndicesSz, knodesDKeys, knodesDKeysSz, knodesDIsLeaf, knodesDIsLeafSz, knodesDNumKeys, knodesDNumKeysSz, knodes_elem, recordsD, recordsDSz, currKnodeD, currKnodeDSz, currKnodeDC, currKnodeDCSz, offsetD, offsetDSz, offsetDC, offsetDCSz, keysD, keysDSz, ansD, ansDSz, ansDC, ansDCSz);
 *            POSTAMBLE(height, knodesDLocation, knodesDLocationSz, knodesDIndices, knodesDIndicesSz, knodesDKeys, knodesDKeysSz, knodesDIsLeaf, knodesDIsLeafSz, knodesDNumKeys, knodesDNumKeysSz, knodes_elem, recordsD, recordsDSz, currKnodeD, currKnodeDSz, currKnodeDC, currKnodeDCSz, offsetD, offsetDSz, offsetDC, offsetDCSz, keysD, keysDSz, ansD, ansDSz, ansDC, ansDCSz);
 *            LOOPPREAMBLE(height, knodesDLocation, knodesDLocationSz, knodesDIndices, knodesDIndicesSz, knodesDKeys, knodesDKeysSz, knodesDIsLeaf, knodesDIsLeafSz, knodesDNumKeys, knodesDNumKeysSz, knodes_elem, recordsD, recordsDSz, currKnodeD, currKnodeDSz, currKnodeDC, currKnodeDCSz, offsetD, offsetDSz, offsetDC, offsetDCSz, keysD, keysDSz, ansD, ansDSz, ansDC, ansDCSz, loopFlag);
 *            LOOPPOSTAMBLE(height, knodesDLocation, knodesDLocationSz, knodesDIndices, knodesDIndicesSz, knodesDKeys, knodesDKeysSz, knodesDIsLeaf, knodesDIsLeafSz, knodesDNumKeys, knodesDNumKeysSz, knodes_elem, recordsD, recordsDSz, currKnodeD, currKnodeDSz, currKnodeDC, currKnodeDCSz, offsetD, offsetDSz, offsetDC, offsetDCSz, keysD, keysDSz, ansD, ansDSz, ansDC, ansDCSz, loopFlag);
 *            CLEANUP(height, knodesDLocation, knodesDLocationSz, knodesDIndices, knodesDIndicesSz, knodesDKeys, knodesDKeysSz, knodesDIsLeaf, knodesDIsLeafSz, knodesDNumKeys, knodesDNumKeysSz, knodes_elem, recordsD, recordsDSz, currKnodeD, currKnodeDSz, currKnodeDC, currKnodeDCSz, offsetD, offsetDSz, offsetDC, offsetDCSz, keysD, keysDSz, ansD, ansDSz, ansDC, ansDCSz);
 *        where:
 *            height: variable (long);
 *            knodesDLocation: variable (int *);
 *            knodesDLocationSz: number of members in variable (unsigned int);
 *            knodesDIndices: variable (int *);
 *            knodesDIndicesSz: number of members in variable (unsigned int);
 *            knodesDKeys: variable (int *);
 *            knodesDKeysSz: number of members in variable (unsigned int);
 *            knodesDIsLeaf: variable (bool *);
 *            knodesDIsLeafSz: number of members in variable (unsigned int);
 *            knodesDNumKeys: variable (int *);
 *            knodesDNumKeysSz: number of members in variable (unsigned int);
 *            knodes_elem: variable (long);
 *            recordsD: variable (int *);
 *            recordsDSz: number of members in variable (unsigned int);
 *            currKnodeD: variable (long *);
 *            currKnodeDSz: number of members in variable (unsigned int);
 *            currKnodeDC: variable (long *);
 *            currKnodeDCSz: number of members in variable (unsigned int);
 *            offsetD: variable (long *);
 *            offsetDSz: number of members in variable (unsigned int);
 *            offsetDC: variable (long *);
 *            offsetDCSz: number of members in variable (unsigned int);
 *            keysD: variable (int *);
 *            keysDSz: number of members in variable (unsigned int);
 *            ansD: variable (int *);
 *            ansDSz: number of members in variable (unsigned int);
 *            ansDC: variable (int *);
 *            ansDCSz: number of members in variable (unsigned int);
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
	cl_command_queue queueFindk = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelFindk = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimFindk = 1;
	size_t globalSizeFindk[1] = {
		2560000
	};
	size_t localSizeFindk[1] = {
		256
	};

	/* Input/output variables */
	long height = 2;
	int *knodesDLocation = malloc(7874 * sizeof(int));
	cl_mem knodesDLocationK = NULL;
	int *knodesDIndices = malloc(2023618 * sizeof(int));
	cl_mem knodesDIndicesK = NULL;
	int *knodesDKeys = malloc(2023618 * sizeof(int));
	cl_mem knodesDKeysK = NULL;
	bool *knodesDIsLeaf = malloc(7874 * sizeof(bool));
	cl_mem knodesDIsLeafK = NULL;
	int *knodesDNumKeys = malloc(7874 * sizeof(int));
	cl_mem knodesDNumKeysK = NULL;
	long knodes_elem = 7874;
	int *recordsD = malloc(1000000 * sizeof(int));
	cl_mem recordsDK = NULL;
	long *currKnodeD = malloc(10000 * sizeof(long));
	long *currKnodeDC = malloc(10000 * sizeof(long));
	cl_mem currKnodeDK = NULL;
	long *offsetD = malloc(10000 * sizeof(long));
	long *offsetDC = malloc(10000 * sizeof(long));
	cl_mem offsetDK = NULL;
	int *keysD = malloc(10000 * sizeof(int));
	cl_mem keysDK = NULL;
	int *ansD = malloc(10000 * sizeof(int));
	int *ansDC = malloc(10000 * sizeof(int));
	cl_mem ansDK = NULL;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(height, knodesDLocation, 7874, knodesDIndices, 2023618, knodesDKeys, 2023618, knodesDIsLeaf, 7874, knodesDNumKeys, 7874, knodes_elem, recordsD, 1000000, currKnodeD, 10000, currKnodeDC, 10000, offsetD, 10000, offsetDC, 10000, keysD, 10000, ansD, 10000, ansDC, 10000);
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

	/* Create command queue for findK kernel */
	PRINT_STEP("Creating command queue for \"findK\"...");
	queueFindk = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create findK kernel */
	PRINT_STEP("Creating kernel \"findK\" from program...");
	kernelFindk = clCreateKernel(program, "findK", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	knodesDLocationK = clCreateBuffer(context, CL_MEM_READ_ONLY, 7874 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (knodesDLocationK)"));
	knodesDIndicesK = clCreateBuffer(context, CL_MEM_READ_ONLY, 2023618 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (knodesDIndicesK)"));
	knodesDKeysK = clCreateBuffer(context, CL_MEM_READ_ONLY, 2023618 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (knodesDKeysK)"));
	knodesDIsLeafK = clCreateBuffer(context, CL_MEM_READ_ONLY, 7874 * sizeof(bool), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (knodesDIsLeafK)"));
	knodesDNumKeysK = clCreateBuffer(context, CL_MEM_READ_ONLY, 7874 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (knodesDNumKeysK)"));
	recordsDK = clCreateBuffer(context, CL_MEM_READ_ONLY, 1000000 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (recordsDK)"));
	currKnodeDK = clCreateBuffer(context, CL_MEM_READ_WRITE, 10000 * sizeof(long), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (currKnodeDK)"));
	offsetDK = clCreateBuffer(context, CL_MEM_READ_WRITE, 10000 * sizeof(long), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (offsetDK)"));
	keysDK = clCreateBuffer(context, CL_MEM_READ_ONLY, 10000 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (keysDK)"));
	ansDK = clCreateBuffer(context, CL_MEM_READ_WRITE, 10000 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (ansDK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for findK */
	PRINT_STEP("Setting kernel arguments for \"findK\"...");
	fRet = clSetKernelArg(kernelFindk, 0, sizeof(long), &height);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (height)"));
	fRet = clSetKernelArg(kernelFindk, 1, sizeof(cl_mem), &knodesDLocationK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (knodesDLocationK)"));
	fRet = clSetKernelArg(kernelFindk, 2, sizeof(cl_mem), &knodesDIndicesK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (knodesDIndicesK)"));
	fRet = clSetKernelArg(kernelFindk, 3, sizeof(cl_mem), &knodesDKeysK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (knodesDKeysK)"));
	fRet = clSetKernelArg(kernelFindk, 4, sizeof(cl_mem), &knodesDIsLeafK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (knodesDIsLeafK)"));
	fRet = clSetKernelArg(kernelFindk, 5, sizeof(cl_mem), &knodesDNumKeysK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (knodesDNumKeysK)"));
	fRet = clSetKernelArg(kernelFindk, 6, sizeof(long), &knodes_elem);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (knodes_elem)"));
	fRet = clSetKernelArg(kernelFindk, 7, sizeof(cl_mem), &recordsDK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (recordsDK)"));
	fRet = clSetKernelArg(kernelFindk, 8, sizeof(cl_mem), &currKnodeDK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (currKnodeDK)"));
	fRet = clSetKernelArg(kernelFindk, 9, sizeof(cl_mem), &offsetDK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (offsetDK)"));
	fRet = clSetKernelArg(kernelFindk, 10, sizeof(cl_mem), &keysDK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (keysDK)"));
	fRet = clSetKernelArg(kernelFindk, 11, sizeof(cl_mem), &ansDK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (ansDK)"));
	PRINT_SUCCESS();

	do {
		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clSetKernelArg(kernelFindk, 0, sizeof(long), &height);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (height)"));
		fRet = clEnqueueWriteBuffer(queueFindk, knodesDLocationK, CL_TRUE, 0, 7874 * sizeof(int), knodesDLocation, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (knodesDLocationK)"));
		fRet = clEnqueueWriteBuffer(queueFindk, knodesDIndicesK, CL_TRUE, 0, 2023618 * sizeof(int), knodesDIndices, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (knodesDIndicesK)"));
		fRet = clEnqueueWriteBuffer(queueFindk, knodesDKeysK, CL_TRUE, 0, 2023618 * sizeof(int), knodesDKeys, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (knodesDKeysK)"));
		fRet = clEnqueueWriteBuffer(queueFindk, knodesDIsLeafK, CL_TRUE, 0, 7874 * sizeof(bool), knodesDIsLeaf, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (knodesDIsLeafK)"));
		fRet = clEnqueueWriteBuffer(queueFindk, knodesDNumKeysK, CL_TRUE, 0, 7874 * sizeof(int), knodesDNumKeys, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (knodesDNumKeysK)"));
		fRet = clSetKernelArg(kernelFindk, 6, sizeof(long), &knodes_elem);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (knodes_elem)"));
		fRet = clEnqueueWriteBuffer(queueFindk, recordsDK, CL_TRUE, 0, 1000000 * sizeof(int), recordsD, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (recordsDK)"));
		fRet = clEnqueueWriteBuffer(queueFindk, currKnodeDK, CL_TRUE, 0, 10000 * sizeof(long), currKnodeD, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (currKnodeDK)"));
		fRet = clEnqueueWriteBuffer(queueFindk, offsetDK, CL_TRUE, 0, 10000 * sizeof(long), offsetD, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (offsetDK)"));
		fRet = clEnqueueWriteBuffer(queueFindk, keysDK, CL_TRUE, 0, 10000 * sizeof(int), keysD, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (keysDK)"));
		fRet = clEnqueueWriteBuffer(queueFindk, ansDK, CL_TRUE, 0, 10000 * sizeof(int), ansD, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (ansDK)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueFindk, kernelFindk, workDimFindk, NULL, globalSizeFindk, localSizeFindk, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueFindk);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueFindk, currKnodeDK, CL_TRUE, 0, 10000 * sizeof(long), currKnodeD, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueFindk, offsetDK, CL_TRUE, 0, 10000 * sizeof(long), offsetD, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueFindk, ansDK, CL_TRUE, 0, 10000 * sizeof(int), ansD, 0, NULL, NULL);
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
	for(i = 0; i < 10000; i++) {
		if(currKnodeDC[i] != currKnodeD[i]) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable currKnodeD[%d]: expected %ld got %ld.\n", i, currKnodeDC[i], currKnodeD[i]);
		}
	}
	for(i = 0; i < 10000; i++) {
		if(offsetDC[i] != offsetD[i]) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable offsetD[%d]: expected %ld got %ld.\n", i, offsetDC[i], offsetD[i]);
		}
	}
	for(i = 0; i < 10000; i++) {
		if(ansDC[i] != ansD[i]) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable ansD[%d]: expected %d got %d.\n", i, ansDC[i], ansD[i]);
		}
	}
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(knodesDLocationK)
		clReleaseMemObject(knodesDLocationK);
	if(knodesDIndicesK)
		clReleaseMemObject(knodesDIndicesK);
	if(knodesDKeysK)
		clReleaseMemObject(knodesDKeysK);
	if(knodesDIsLeafK)
		clReleaseMemObject(knodesDIsLeafK);
	if(knodesDNumKeysK)
		clReleaseMemObject(knodesDNumKeysK);
	if(recordsDK)
		clReleaseMemObject(recordsDK);
	if(currKnodeDK)
		clReleaseMemObject(currKnodeDK);
	if(offsetDK)
		clReleaseMemObject(offsetDK);
	if(keysDK)
		clReleaseMemObject(keysDK);
	if(ansDK)
		clReleaseMemObject(ansDK);

	/* Dealloc variables */
	free(knodesDLocation);
	free(knodesDIndices);
	free(knodesDKeys);
	free(knodesDIsLeaf);
	free(knodesDNumKeys);
	free(recordsD);
	free(currKnodeD);
	free(currKnodeDC);
	free(offsetD);
	free(offsetDC);
	free(keysD);
	free(ansD);
	free(ansDC);

	/* Dealloc kernels */
	if(kernelFindk)
		clReleaseKernel(kernelFindk);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueFindk)
		clReleaseCommandQueue(queueFindk);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);


	return rv;
}
