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
 *            PREAMBLE(lambda, lambdaSz, omega, omegaSz, errCnt, errCntSz, errLoc, errLocSz, alphaInv, alphaInvSz, errOut, errOutSz, errOutC, errOutCSz, loopCount);
 *            POSTAMBLE(lambda, lambdaSz, omega, omegaSz, errCnt, errCntSz, errLoc, errLocSz, alphaInv, alphaInvSz, errOut, errOutSz, errOutC, errOutCSz, loopCount);
 *            LOOPPREAMBLE(lambda, lambdaSz, omega, omegaSz, errCnt, errCntSz, errLoc, errLocSz, alphaInv, alphaInvSz, errOut, errOutSz, errOutC, errOutCSz, loopCount, loopFlag);
 *            LOOPPOSTAMBLE(lambda, lambdaSz, omega, omegaSz, errCnt, errCntSz, errLoc, errLocSz, alphaInv, alphaInvSz, errOut, errOutSz, errOutC, errOutCSz, loopCount, loopFlag);
 *            CLEANUP(lambda, lambdaSz, omega, omegaSz, errCnt, errCntSz, errLoc, errLocSz, alphaInv, alphaInvSz, errOut, errOutSz, errOutC, errOutCSz, loopCount);
 *        where:
 *            lambda: variable (short *);
 *            lambdaSz: number of members in variable (unsigned int);
 *            omega: variable (short *);
 *            omegaSz: number of members in variable (unsigned int);
 *            errCnt: variable (short *);
 *            errCntSz: number of members in variable (unsigned int);
 *            errLoc: variable (short *);
 *            errLocSz: number of members in variable (unsigned int);
 *            alphaInv: variable (short *);
 *            alphaInvSz: number of members in variable (unsigned int);
 *            errOut: variable (short *);
 *            errOutSz: number of members in variable (unsigned int);
 *            errOutC: variable (short *);
 *            errOutCSz: number of members in variable (unsigned int);
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
	cl_command_queue queueForneys = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelForneys = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimForneys = 1;
	size_t globalSizeForneys[1] = {
		256
	};
	size_t localSizeForneys[1] = {
		256
	};

	/* Input/output variables */
	short *lambda = malloc(4080 * sizeof(short));
	cl_mem lambdaK = NULL;
	short *omega = malloc(4080 * sizeof(short));
	cl_mem omegaK = NULL;
	short *errCnt = malloc(255 * sizeof(short));
	cl_mem errCntK = NULL;
	short *errLoc = malloc(4080 * sizeof(short));
	cl_mem errLocK = NULL;
	short *alphaInv = malloc(4080 * sizeof(short));
	cl_mem alphaInvK = NULL;
	short *errOut = malloc(56865 * sizeof(short));
	short *errOutC = malloc(56865 * sizeof(short));
	cl_mem errOutK = NULL;
	unsigned char loopCount = 255;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(lambda, 4080, omega, 4080, errCnt, 255, errLoc, 4080, alphaInv, 4080, errOut, 56865, errOutC, 56865, loopCount);
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

	/* Create command queue for forneys kernel */
	PRINT_STEP("Creating command queue for \"forneys\"...");
	queueForneys = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create forneys kernel */
	PRINT_STEP("Creating kernel \"forneys\" from program...");
	kernelForneys = clCreateKernel(program, "forneys", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	lambdaK = clCreateBuffer(context, CL_MEM_READ_ONLY, 4080 * sizeof(short), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (lambdaK)"));
	omegaK = clCreateBuffer(context, CL_MEM_READ_ONLY, 4080 * sizeof(short), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (omegaK)"));
	errCntK = clCreateBuffer(context, CL_MEM_READ_ONLY, 255 * sizeof(short), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (errCntK)"));
	errLocK = clCreateBuffer(context, CL_MEM_READ_ONLY, 4080 * sizeof(short), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (errLocK)"));
	alphaInvK = clCreateBuffer(context, CL_MEM_READ_ONLY, 4080 * sizeof(short), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (alphaInvK)"));
	errOutK = clCreateBuffer(context, CL_MEM_READ_WRITE, 56865 * sizeof(short), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (errOutK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for forneys */
	PRINT_STEP("Setting kernel arguments for \"forneys\"...");
	fRet = clSetKernelArg(kernelForneys, 0, sizeof(cl_mem), &lambdaK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (lambdaK)"));
	fRet = clSetKernelArg(kernelForneys, 1, sizeof(cl_mem), &omegaK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (omegaK)"));
	fRet = clSetKernelArg(kernelForneys, 2, sizeof(cl_mem), &errCntK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (errCntK)"));
	fRet = clSetKernelArg(kernelForneys, 3, sizeof(cl_mem), &errLocK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (errLocK)"));
	fRet = clSetKernelArg(kernelForneys, 4, sizeof(cl_mem), &alphaInvK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (alphaInvK)"));
	fRet = clSetKernelArg(kernelForneys, 5, sizeof(cl_mem), &errOutK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (errOutK)"));
	fRet = clSetKernelArg(kernelForneys, 6, sizeof(unsigned char), &loopCount);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (loopCount)"));
	PRINT_SUCCESS();

	do {
		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clEnqueueWriteBuffer(queueForneys, lambdaK, CL_TRUE, 0, 4080 * sizeof(short), lambda, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (lambdaK)"));
		fRet = clEnqueueWriteBuffer(queueForneys, omegaK, CL_TRUE, 0, 4080 * sizeof(short), omega, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (omegaK)"));
		fRet = clEnqueueWriteBuffer(queueForneys, errCntK, CL_TRUE, 0, 255 * sizeof(short), errCnt, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (errCntK)"));
		fRet = clEnqueueWriteBuffer(queueForneys, errLocK, CL_TRUE, 0, 4080 * sizeof(short), errLoc, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (errLocK)"));
		fRet = clEnqueueWriteBuffer(queueForneys, alphaInvK, CL_TRUE, 0, 4080 * sizeof(short), alphaInv, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (alphaInvK)"));
		fRet = clEnqueueWriteBuffer(queueForneys, errOutK, CL_TRUE, 0, 56865 * sizeof(short), errOut, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (errOutK)"));
		fRet = clSetKernelArg(kernelForneys, 6, sizeof(unsigned char), &loopCount);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (loopCount)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueForneys, kernelForneys, workDimForneys, NULL, globalSizeForneys, localSizeForneys, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueForneys);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueForneys, errOutK, CL_TRUE, 0, 56865 * sizeof(short), errOut, 0, NULL, NULL);
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
	for(i = 0; i < 56865; i++) {
		if(errOutC[i] != errOut[i]) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable errOut[%d]: expected %hd got %hd.\n", i, errOutC[i], errOut[i]);
		}
	}
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(lambdaK)
		clReleaseMemObject(lambdaK);
	if(omegaK)
		clReleaseMemObject(omegaK);
	if(errCntK)
		clReleaseMemObject(errCntK);
	if(errLocK)
		clReleaseMemObject(errLocK);
	if(alphaInvK)
		clReleaseMemObject(alphaInvK);
	if(errOutK)
		clReleaseMemObject(errOutK);

	/* Dealloc variables */
	free(lambda);
	free(omega);
	free(errCnt);
	free(errLoc);
	free(alphaInv);
	free(errOut);
	free(errOutC);

	/* Dealloc kernels */
	if(kernelForneys)
		clReleaseKernel(kernelForneys);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueForneys)
		clReleaseCommandQueue(queueForneys);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);


	return rv;
}
