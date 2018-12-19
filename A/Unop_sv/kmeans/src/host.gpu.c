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
 *            PREAMBLE(feature, featureSz, clusters, clustersSz, membership, membershipSz, npoints, nclusters, nfeatures, offset, size);
 *            POSTAMBLE(feature, featureSz, clusters, clustersSz, membership, membershipSz, npoints, nclusters, nfeatures, offset, size);
 *            LOOPPREAMBLE(feature, featureSz, clusters, clustersSz, membership, membershipSz, npoints, nclusters, nfeatures, offset, size, loopFlag);
 *            LOOPPOSTAMBLE(feature, featureSz, clusters, clustersSz, membership, membershipSz, npoints, nclusters, nfeatures, offset, size, loopFlag);
 *            CLEANUP(feature, featureSz, clusters, clustersSz, membership, membershipSz, npoints, nclusters, nfeatures, offset, size);
 *        where:
 *            feature: variable (float *);
 *            featureSz: number of members in variable (unsigned int);
 *            clusters: variable (float *);
 *            clustersSz: number of members in variable (unsigned int);
 *            membership: variable (int *);
 *            membershipSz: number of members in variable (unsigned int);
 *            npoints: variable (int);
 *            nclusters: variable (int);
 *            nfeatures: variable (int);
 *            offset: variable (int);
 *            size: variable (int);
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
	cl_command_queue queueKmeans_Kernel_C = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelKmeans_Kernel_C = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimKmeans_Kernel_C = 1;
	size_t globalSizeKmeans_Kernel_C[1] = {
		30208
	};
	size_t localSizeKmeans_Kernel_C[1] = {
		256
	};

	/* Input/output variables */
	float *feature = malloc(1020000 * sizeof(float));
	cl_mem featureK = NULL;
	float *clusters = malloc(170 * sizeof(float));
	cl_mem clustersK = NULL;
	int *membership = malloc(30000 * sizeof(int));
	cl_mem membershipK = NULL;
	int npoints = 30000;
	int nclusters = 5;
	int nfeatures = 34;
	int offset = 0;
	int size = 0;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(feature, 1020000, clusters, 170, membership, 30000, npoints, nclusters, nfeatures, offset, size);
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

	/* Create command queue for kmeans_kernel_c kernel */
	PRINT_STEP("Creating command queue for \"kmeans_kernel_c\"...");
	queueKmeans_Kernel_C = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create kmeans_kernel_c kernel */
	PRINT_STEP("Creating kernel \"kmeans_kernel_c\" from program...");
	kernelKmeans_Kernel_C = clCreateKernel(program, "kmeans_kernel_c", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	featureK = clCreateBuffer(context, CL_MEM_READ_ONLY, 1020000 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (featureK)"));
	clustersK = clCreateBuffer(context, CL_MEM_READ_ONLY, 170 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (clustersK)"));
	membershipK = clCreateBuffer(context, CL_MEM_READ_WRITE, 30000 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (membershipK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for kmeans_kernel_c */
	PRINT_STEP("Setting kernel arguments for \"kmeans_kernel_c\"...");
	fRet = clSetKernelArg(kernelKmeans_Kernel_C, 0, sizeof(cl_mem), &featureK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (featureK)"));
	fRet = clSetKernelArg(kernelKmeans_Kernel_C, 1, sizeof(cl_mem), &clustersK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (clustersK)"));
	fRet = clSetKernelArg(kernelKmeans_Kernel_C, 2, sizeof(cl_mem), &membershipK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (membershipK)"));
	fRet = clSetKernelArg(kernelKmeans_Kernel_C, 3, sizeof(int), &npoints);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (npoints)"));
	fRet = clSetKernelArg(kernelKmeans_Kernel_C, 4, sizeof(int), &nclusters);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (nclusters)"));
	fRet = clSetKernelArg(kernelKmeans_Kernel_C, 5, sizeof(int), &nfeatures);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (nfeatures)"));
	fRet = clSetKernelArg(kernelKmeans_Kernel_C, 6, sizeof(int), &offset);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (offset)"));
	fRet = clSetKernelArg(kernelKmeans_Kernel_C, 7, sizeof(int), &size);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (size)"));
	PRINT_SUCCESS();

	do {
		/* Calling loop preamble function */
		PRINT_STEP("[%d] Calling loop preamble function...", i);
		LOOPPREAMBLE(feature, 1020000, clusters, 170, membership, 30000, npoints, nclusters, nfeatures, offset, size, loopFlag);
		PRINT_SUCCESS();

		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clEnqueueWriteBuffer(queueKmeans_Kernel_C, featureK, CL_TRUE, 0, 1020000 * sizeof(float), feature, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (featureK)"));
		fRet = clEnqueueWriteBuffer(queueKmeans_Kernel_C, clustersK, CL_TRUE, 0, 170 * sizeof(float), clusters, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (clustersK)"));
		fRet = clEnqueueWriteBuffer(queueKmeans_Kernel_C, membershipK, CL_TRUE, 0, 30000 * sizeof(int), membership, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (membershipK)"));
		fRet = clSetKernelArg(kernelKmeans_Kernel_C, 3, sizeof(int), &npoints);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (npoints)"));
		fRet = clSetKernelArg(kernelKmeans_Kernel_C, 4, sizeof(int), &nclusters);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (nclusters)"));
		fRet = clSetKernelArg(kernelKmeans_Kernel_C, 5, sizeof(int), &nfeatures);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (nfeatures)"));
		fRet = clSetKernelArg(kernelKmeans_Kernel_C, 6, sizeof(int), &offset);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (offset)"));
		fRet = clSetKernelArg(kernelKmeans_Kernel_C, 7, sizeof(int), &size);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (size)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueKmeans_Kernel_C, kernelKmeans_Kernel_C, workDimKmeans_Kernel_C, NULL, globalSizeKmeans_Kernel_C, localSizeKmeans_Kernel_C, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueKmeans_Kernel_C);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueKmeans_Kernel_C, membershipK, CL_TRUE, 0, 30000 * sizeof(int), membership, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueReadBuffer"));
		PRINT_SUCCESS();

		/* Calling loop postamble function */
		PRINT_STEP("[%d] Calling loop postamble function...", i);
		LOOPPOSTAMBLE(feature, 1020000, clusters, 170, membership, 30000, npoints, nclusters, nfeatures, offset, size, loopFlag);
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
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(featureK)
		clReleaseMemObject(featureK);
	if(clustersK)
		clReleaseMemObject(clustersK);
	if(membershipK)
		clReleaseMemObject(membershipK);

	/* Dealloc variables */
	free(feature);
	free(clusters);
	free(membership);

	/* Dealloc kernels */
	if(kernelKmeans_Kernel_C)
		clReleaseKernel(kernelKmeans_Kernel_C);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueKmeans_Kernel_C)
		clReleaseCommandQueue(queueKmeans_Kernel_C);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);

	/* Calling cleanup function */
	CLEANUP(feature, 1020000, clusters, 170, membership, 30000, npoints, nclusters, nfeatures, offset, size);

	return rv;
}
