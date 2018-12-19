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
 *            PREAMBLE(lA, lASz, errLocOut, errLocOutSz, errLocOutC, errLocOutCSz, alphaInvOut, alphaInvOutSz, alphaInvOutC, alphaInvOutCSz, errCnt, errCntSz, errCntC, errCntCSz, loopCount);
 *            POSTAMBLE(lA, lASz, errLocOut, errLocOutSz, errLocOutC, errLocOutCSz, alphaInvOut, alphaInvOutSz, alphaInvOutC, alphaInvOutCSz, errCnt, errCntSz, errCntC, errCntCSz, loopCount);
 *            LOOPPREAMBLE(lA, lASz, errLocOut, errLocOutSz, errLocOutC, errLocOutCSz, alphaInvOut, alphaInvOutSz, alphaInvOutC, alphaInvOutCSz, errCnt, errCntSz, errCntC, errCntCSz, loopCount, loopFlag);
 *            LOOPPOSTAMBLE(lA, lASz, errLocOut, errLocOutSz, errLocOutC, errLocOutCSz, alphaInvOut, alphaInvOutSz, alphaInvOutC, alphaInvOutCSz, errCnt, errCntSz, errCntC, errCntCSz, loopCount, loopFlag);
 *            CLEANUP(lA, lASz, errLocOut, errLocOutSz, errLocOutC, errLocOutCSz, alphaInvOut, alphaInvOutSz, alphaInvOutC, alphaInvOutCSz, errCnt, errCntSz, errCntC, errCntCSz, loopCount);
 *        where:
 *            lA: variable (short *);
 *            lASz: number of members in variable (unsigned int);
 *            errLocOut: variable (short *);
 *            errLocOutSz: number of members in variable (unsigned int);
 *            errLocOutC: variable (short *);
 *            errLocOutCSz: number of members in variable (unsigned int);
 *            alphaInvOut: variable (short *);
 *            alphaInvOutSz: number of members in variable (unsigned int);
 *            alphaInvOutC: variable (short *);
 *            alphaInvOutCSz: number of members in variable (unsigned int);
 *            errCnt: variable (short *);
 *            errCntSz: number of members in variable (unsigned int);
 *            errCntC: variable (short *);
 *            errCntCSz: number of members in variable (unsigned int);
 *            loopCount: variable (unsigned char);
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
	cl_command_queue queueChien = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelChien = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimChien = 1;
	size_t globalSizeChien[1] = {
		256
	};
	size_t localSizeChien[1] = {
		256
	};

	/* Input/output variables */
	short *lA = malloc(4080 * sizeof(short));
	cl_mem lAK = NULL;
	short *errLocOut = malloc(4080 * sizeof(short));
	short *errLocOutC = malloc(4080 * sizeof(short));
	cl_mem errLocOutK = NULL;
	short *alphaInvOut = malloc(4080 * sizeof(short));
	short *alphaInvOutC = malloc(4080 * sizeof(short));
	cl_mem alphaInvOutK = NULL;
	short *errCnt = malloc(255 * sizeof(short));
	short *errCntC = malloc(255 * sizeof(short));
	cl_mem errCntK = NULL;
	unsigned char loopCount = 255;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(lA, 4080, errLocOut, 4080, errLocOutC, 4080, alphaInvOut, 4080, alphaInvOutC, 4080, errCnt, 255, errCntC, 255, loopCount);
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

	/* Create command queue for chien kernel */
	PRINT_STEP("Creating command queue for \"chien\"...");
	queueChien = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create chien kernel */
	PRINT_STEP("Creating kernel \"chien\" from program...");
	kernelChien = clCreateKernel(program, "chien", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	lAK = clCreateBuffer(context, CL_MEM_READ_ONLY, 4080 * sizeof(short), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (lAK)"));
	errLocOutK = clCreateBuffer(context, CL_MEM_READ_WRITE, 4080 * sizeof(short), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (errLocOutK)"));
	alphaInvOutK = clCreateBuffer(context, CL_MEM_READ_WRITE, 4080 * sizeof(short), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (alphaInvOutK)"));
	errCntK = clCreateBuffer(context, CL_MEM_READ_WRITE, 255 * sizeof(short), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (errCntK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for chien */
	PRINT_STEP("Setting kernel arguments for \"chien\"...");
	fRet = clSetKernelArg(kernelChien, 0, sizeof(cl_mem), &lAK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (lAK)"));
	fRet = clSetKernelArg(kernelChien, 1, sizeof(cl_mem), &errLocOutK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (errLocOutK)"));
	fRet = clSetKernelArg(kernelChien, 2, sizeof(cl_mem), &alphaInvOutK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (alphaInvOutK)"));
	fRet = clSetKernelArg(kernelChien, 3, sizeof(cl_mem), &errCntK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (errCntK)"));
	fRet = clSetKernelArg(kernelChien, 4, sizeof(unsigned char), &loopCount);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (loopCount)"));
	PRINT_SUCCESS();

	do {
		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clEnqueueWriteBuffer(queueChien, lAK, CL_TRUE, 0, 4080 * sizeof(short), lA, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (lAK)"));
		fRet = clEnqueueWriteBuffer(queueChien, errLocOutK, CL_TRUE, 0, 4080 * sizeof(short), errLocOut, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (errLocOutK)"));
		fRet = clEnqueueWriteBuffer(queueChien, alphaInvOutK, CL_TRUE, 0, 4080 * sizeof(short), alphaInvOut, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (alphaInvOutK)"));
		fRet = clEnqueueWriteBuffer(queueChien, errCntK, CL_TRUE, 0, 255 * sizeof(short), errCnt, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (errCntK)"));
		fRet = clSetKernelArg(kernelChien, 4, sizeof(unsigned char), &loopCount);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (loopCount)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueChien, kernelChien, workDimChien, NULL, globalSizeChien, localSizeChien, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueChien);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueChien, errLocOutK, CL_TRUE, 0, 4080 * sizeof(short), errLocOut, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueChien, alphaInvOutK, CL_TRUE, 0, 4080 * sizeof(short), alphaInvOut, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueChien, errCntK, CL_TRUE, 0, 255 * sizeof(short), errCnt, 0, NULL, NULL);
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
	for(i = 0; i < 4080; i++) {
		if(errLocOutC[i] != errLocOut[i]) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable errLocOut[%d]: expected %hd got %hd.\n", i, errLocOutC[i], errLocOut[i]);
		}
	}
	for(i = 0; i < 4080; i++) {
		if(alphaInvOutC[i] != alphaInvOut[i]) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable alphaInvOut[%d]: expected %hd got %hd.\n", i, alphaInvOutC[i], alphaInvOut[i]);
		}
	}
	for(i = 0; i < 255; i++) {
		if(errCntC[i] != errCnt[i]) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable errCnt[%d]: expected %hd got %hd.\n", i, errCntC[i], errCnt[i]);
		}
	}
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(lAK)
		clReleaseMemObject(lAK);
	if(errLocOutK)
		clReleaseMemObject(errLocOutK);
	if(alphaInvOutK)
		clReleaseMemObject(alphaInvOutK);
	if(errCntK)
		clReleaseMemObject(errCntK);

	/* Dealloc variables */
	free(lA);
	free(errLocOut);
	free(errLocOutC);
	free(alphaInvOut);
	free(alphaInvOutC);
	free(errCnt);
	free(errCntC);

	/* Dealloc kernels */
	if(kernelChien)
		clReleaseKernel(kernelChien);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueChien)
		clReleaseCommandQueue(queueChien);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);


	return rv;
}
