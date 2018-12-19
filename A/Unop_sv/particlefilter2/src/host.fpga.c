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
 *            PREAMBLE(weights, weightsSz, weightsC, weightsCSz, Nparticles, partial_sums, partial_sumsSz, CDF, CDFSz, u, uSz, uC, uCSz, seed, seedSz);
 *            POSTAMBLE(weights, weightsSz, weightsC, weightsCSz, Nparticles, partial_sums, partial_sumsSz, CDF, CDFSz, u, uSz, uC, uCSz, seed, seedSz);
 *            LOOPPREAMBLE(weights, weightsSz, weightsC, weightsCSz, Nparticles, partial_sums, partial_sumsSz, CDF, CDFSz, u, uSz, uC, uCSz, seed, seedSz, loopFlag);
 *            LOOPPOSTAMBLE(weights, weightsSz, weightsC, weightsCSz, Nparticles, partial_sums, partial_sumsSz, CDF, CDFSz, u, uSz, uC, uCSz, seed, seedSz, loopFlag);
 *            CLEANUP(weights, weightsSz, weightsC, weightsCSz, Nparticles, partial_sums, partial_sumsSz, CDF, CDFSz, u, uSz, uC, uCSz, seed, seedSz);
 *        where:
 *            weights: variable (double *);
 *            weightsSz: number of members in variable (unsigned int);
 *            weightsC: variable (double *);
 *            weightsCSz: number of members in variable (unsigned int);
 *            Nparticles: variable (int);
 *            partial_sums: variable (double *);
 *            partial_sumsSz: number of members in variable (unsigned int);
 *            CDF: variable (double *);
 *            CDFSz: number of members in variable (unsigned int);
 *            u: variable (double *);
 *            uSz: number of members in variable (unsigned int);
 *            uC: variable (double *);
 *            uCSz: number of members in variable (unsigned int);
 *            seed: variable (int *);
 *            seedSz: number of members in variable (unsigned int);
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
	cl_command_queue queueNormalize_Weights_Kernel = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelNormalize_Weights_Kernel = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimNormalize_Weights_Kernel = 3;
	size_t globalSizeNormalize_Weights_Kernel[3] = {
		40448, 1, 1
	};
	size_t localSizeNormalize_Weights_Kernel[3] = {
		512, 1, 1
	};

	/* Input/output variables */
	double *weights = malloc(40000 * sizeof(double));
	double *weightsC = malloc(40000 * sizeof(double));
	double weightsEpsilon = 0.01;
	cl_mem weightsK = NULL;
	int Nparticles = 40000;
	double *partial_sums = malloc(79 * sizeof(double));
	cl_mem partial_sumsK = NULL;
	double *CDF = malloc(40000 * sizeof(double));
	cl_mem CDFK = NULL;
	double *u = malloc(40000 * sizeof(double));
	double *uC = malloc(40000 * sizeof(double));
	double uEpsilon = 0.01;
	cl_mem uK = NULL;
	int *seed = malloc(40000 * sizeof(int));
	cl_mem seedK = NULL;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(weights, 40000, weightsC, 40000, Nparticles, partial_sums, 79, CDF, 40000, u, 40000, uC, 40000, seed, 40000);
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

	/* Create command queue for normalize_weights_kernel kernel */
	PRINT_STEP("Creating command queue for \"normalize_weights_kernel\"...");
	queueNormalize_Weights_Kernel = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create normalize_weights_kernel kernel */
	PRINT_STEP("Creating kernel \"normalize_weights_kernel\" from program...");
	kernelNormalize_Weights_Kernel = clCreateKernel(program, "normalize_weights_kernel", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	weightsK = clCreateBuffer(context, CL_MEM_READ_WRITE, 40000 * sizeof(double), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (weightsK)"));
	partial_sumsK = clCreateBuffer(context, CL_MEM_READ_ONLY, 79 * sizeof(double), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (partial_sumsK)"));
	CDFK = clCreateBuffer(context, CL_MEM_READ_WRITE, 40000 * sizeof(double), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (CDFK)"));
	uK = clCreateBuffer(context, CL_MEM_READ_WRITE, 40000 * sizeof(double), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (uK)"));
	seedK = clCreateBuffer(context, CL_MEM_READ_ONLY, 40000 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (seedK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for normalize_weights_kernel */
	PRINT_STEP("Setting kernel arguments for \"normalize_weights_kernel\"...");
	fRet = clSetKernelArg(kernelNormalize_Weights_Kernel, 0, sizeof(cl_mem), &weightsK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (weightsK)"));
	fRet = clSetKernelArg(kernelNormalize_Weights_Kernel, 1, sizeof(int), &Nparticles);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (Nparticles)"));
	fRet = clSetKernelArg(kernelNormalize_Weights_Kernel, 2, sizeof(cl_mem), &partial_sumsK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (partial_sumsK)"));
	fRet = clSetKernelArg(kernelNormalize_Weights_Kernel, 3, sizeof(cl_mem), &CDFK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (CDFK)"));
	fRet = clSetKernelArg(kernelNormalize_Weights_Kernel, 4, sizeof(cl_mem), &uK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (uK)"));
	fRet = clSetKernelArg(kernelNormalize_Weights_Kernel, 5, sizeof(cl_mem), &seedK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (seedK)"));
	PRINT_SUCCESS();

	do {
		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clEnqueueWriteBuffer(queueNormalize_Weights_Kernel, weightsK, CL_TRUE, 0, 40000 * sizeof(double), weights, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (weightsK)"));
		fRet = clSetKernelArg(kernelNormalize_Weights_Kernel, 1, sizeof(int), &Nparticles);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (Nparticles)"));
		fRet = clEnqueueWriteBuffer(queueNormalize_Weights_Kernel, partial_sumsK, CL_TRUE, 0, 79 * sizeof(double), partial_sums, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (partial_sumsK)"));
		fRet = clEnqueueWriteBuffer(queueNormalize_Weights_Kernel, CDFK, CL_TRUE, 0, 40000 * sizeof(double), CDF, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (CDFK)"));
		fRet = clEnqueueWriteBuffer(queueNormalize_Weights_Kernel, uK, CL_TRUE, 0, 40000 * sizeof(double), u, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (uK)"));
		fRet = clEnqueueWriteBuffer(queueNormalize_Weights_Kernel, seedK, CL_TRUE, 0, 40000 * sizeof(int), seed, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (seedK)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueNormalize_Weights_Kernel, kernelNormalize_Weights_Kernel, workDimNormalize_Weights_Kernel, NULL, globalSizeNormalize_Weights_Kernel, localSizeNormalize_Weights_Kernel, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueNormalize_Weights_Kernel);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueNormalize_Weights_Kernel, weightsK, CL_TRUE, 0, 40000 * sizeof(double), weights, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueNormalize_Weights_Kernel, CDFK, CL_TRUE, 0, 40000 * sizeof(double), CDF, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueNormalize_Weights_Kernel, uK, CL_TRUE, 0, 40000 * sizeof(double), u, 0, NULL, NULL);
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
	for(i = 0; i < 40000; i++) {
		if(TEST_EPSILON(weightsC[i],  weights[i], weightsEpsilon)) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable weights[%d]: expected %lf got %lf (with epsilon).\n", i, weightsC[i], weights[i]);
		}
	}
	for(i = 0; i < 40000; i++) {
		if(TEST_EPSILON(uC[i],  u[i], uEpsilon)) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable u[%d]: expected %lf got %lf (with epsilon).\n", i, uC[i], u[i]);
		}
	}
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(weightsK)
		clReleaseMemObject(weightsK);
	if(partial_sumsK)
		clReleaseMemObject(partial_sumsK);
	if(CDFK)
		clReleaseMemObject(CDFK);
	if(uK)
		clReleaseMemObject(uK);
	if(seedK)
		clReleaseMemObject(seedK);

	/* Dealloc variables */
	free(weights);
	free(weightsC);
	free(partial_sums);
	free(CDF);
	free(u);
	free(uC);
	free(seed);

	/* Dealloc kernels */
	if(kernelNormalize_Weights_Kernel)
		clReleaseKernel(kernelNormalize_Weights_Kernel);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueNormalize_Weights_Kernel)
		clReleaseCommandQueue(queueNormalize_Weights_Kernel);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);


	return rv;
}
