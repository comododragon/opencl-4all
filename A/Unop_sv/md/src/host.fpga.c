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
 *            PREAMBLE(force, forceSz, forceC, forceCSz, position, positionSz, maxNeighbors, neighborList, neighborListSz, cutsq, lj1, lj2, nAtom);
 *            POSTAMBLE(force, forceSz, forceC, forceCSz, position, positionSz, maxNeighbors, neighborList, neighborListSz, cutsq, lj1, lj2, nAtom);
 *            LOOPPREAMBLE(force, forceSz, forceC, forceCSz, position, positionSz, maxNeighbors, neighborList, neighborListSz, cutsq, lj1, lj2, nAtom, loopFlag);
 *            LOOPPOSTAMBLE(force, forceSz, forceC, forceCSz, position, positionSz, maxNeighbors, neighborList, neighborListSz, cutsq, lj1, lj2, nAtom, loopFlag);
 *            CLEANUP(force, forceSz, forceC, forceCSz, position, positionSz, maxNeighbors, neighborList, neighborListSz, cutsq, lj1, lj2, nAtom);
 *        where:
 *            force: variable (cl_float3 *);
 *            forceSz: number of members in variable (unsigned int);
 *            forceC: variable (cl_float3 *);
 *            forceCSz: number of members in variable (unsigned int);
 *            position: variable (cl_float3 *);
 *            positionSz: number of members in variable (unsigned int);
 *            maxNeighbors: variable (int);
 *            neighborList: variable (int *);
 *            neighborListSz: number of members in variable (unsigned int);
 *            cutsq: variable (float);
 *            lj1: variable (float);
 *            lj2: variable (float);
 *            nAtom: variable (int);
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
	cl_command_queue queueCompute_Lj_Force = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelCompute_Lj_Force = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimCompute_Lj_Force = 1;
	size_t globalSizeCompute_Lj_Force[1] = {
		12288
	};
	size_t localSizeCompute_Lj_Force[1] = {
		128
	};

	/* Input/output variables */
	cl_float3 *force = malloc(12288 * sizeof(cl_float3));
	cl_float3 *forceC = malloc(12288 * sizeof(cl_float3));
	double forceEpsilon = 0.1;
	cl_mem forceK = NULL;
	cl_float3 *position = malloc(12288 * sizeof(cl_float3));
	cl_mem positionK = NULL;
	int maxNeighbors = 128;
	int *neighborList = malloc(1572864 * sizeof(int));
	cl_mem neighborListK = NULL;
	float cutsq = 16;
	float lj1 = 1.5;
	float lj2 = 2;
	int nAtom = 12288;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(force, 12288, forceC, 12288, position, 12288, maxNeighbors, neighborList, 1572864, cutsq, lj1, lj2, nAtom);
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

	/* Create command queue for compute_lj_force kernel */
	PRINT_STEP("Creating command queue for \"compute_lj_force\"...");
	queueCompute_Lj_Force = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create compute_lj_force kernel */
	PRINT_STEP("Creating kernel \"compute_lj_force\" from program...");
	kernelCompute_Lj_Force = clCreateKernel(program, "compute_lj_force", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	forceK = clCreateBuffer(context, CL_MEM_READ_WRITE, 12288 * sizeof(cl_float3), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (forceK)"));
	positionK = clCreateBuffer(context, CL_MEM_READ_ONLY, 12288 * sizeof(cl_float3), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (positionK)"));
	neighborListK = clCreateBuffer(context, CL_MEM_READ_ONLY, 1572864 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (neighborListK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for compute_lj_force */
	PRINT_STEP("Setting kernel arguments for \"compute_lj_force\"...");
	fRet = clSetKernelArg(kernelCompute_Lj_Force, 0, sizeof(cl_mem), &forceK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (forceK)"));
	fRet = clSetKernelArg(kernelCompute_Lj_Force, 1, sizeof(cl_mem), &positionK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (positionK)"));
	fRet = clSetKernelArg(kernelCompute_Lj_Force, 2, sizeof(int), &maxNeighbors);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (maxNeighbors)"));
	fRet = clSetKernelArg(kernelCompute_Lj_Force, 3, sizeof(cl_mem), &neighborListK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (neighborListK)"));
	fRet = clSetKernelArg(kernelCompute_Lj_Force, 4, sizeof(float), &cutsq);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (cutsq)"));
	fRet = clSetKernelArg(kernelCompute_Lj_Force, 5, sizeof(float), &lj1);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (lj1)"));
	fRet = clSetKernelArg(kernelCompute_Lj_Force, 6, sizeof(float), &lj2);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (lj2)"));
	fRet = clSetKernelArg(kernelCompute_Lj_Force, 7, sizeof(int), &nAtom);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (nAtom)"));
	PRINT_SUCCESS();

	do {
		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clEnqueueWriteBuffer(queueCompute_Lj_Force, forceK, CL_TRUE, 0, 12288 * sizeof(cl_float3), force, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (forceK)"));
		fRet = clEnqueueWriteBuffer(queueCompute_Lj_Force, positionK, CL_TRUE, 0, 12288 * sizeof(cl_float3), position, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (positionK)"));
		fRet = clSetKernelArg(kernelCompute_Lj_Force, 2, sizeof(int), &maxNeighbors);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (maxNeighbors)"));
		fRet = clEnqueueWriteBuffer(queueCompute_Lj_Force, neighborListK, CL_TRUE, 0, 1572864 * sizeof(int), neighborList, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (neighborListK)"));
		fRet = clSetKernelArg(kernelCompute_Lj_Force, 4, sizeof(float), &cutsq);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (cutsq)"));
		fRet = clSetKernelArg(kernelCompute_Lj_Force, 5, sizeof(float), &lj1);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (lj1)"));
		fRet = clSetKernelArg(kernelCompute_Lj_Force, 6, sizeof(float), &lj2);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (lj2)"));
		fRet = clSetKernelArg(kernelCompute_Lj_Force, 7, sizeof(int), &nAtom);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (nAtom)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueCompute_Lj_Force, kernelCompute_Lj_Force, workDimCompute_Lj_Force, NULL, globalSizeCompute_Lj_Force, localSizeCompute_Lj_Force, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueCompute_Lj_Force);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueCompute_Lj_Force, forceK, CL_TRUE, 0, 12288 * sizeof(cl_float3), force, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueReadBuffer"));
		PRINT_SUCCESS();

		timersub(&tNow, &tThen, &tDelta);
		timeradd(&tExecTime, &tDelta, &tExecTime);
		i++;
	} while(loopFlag);

	/* Calling postamble function */
	PRINT_STEP("Calling postamble function...");
	POSTAMBLE(force, 12288, forceC, 12288, position, 12288, maxNeighbors, neighborList, 1572864, cutsq, lj1, lj2, nAtom);
	PRINT_SUCCESS();

	/* Print profiling results */
	long totalTime = (1000000 * tExecTime.tv_sec) + tExecTime.tv_usec;
	printf("Elapsed time spent on kernels: %ld us; Average time per iteration: %lf us.\n", totalTime, totalTime / (double) i);

	/* Validate received data */
	PRINT_STEP("Validating received data...");
	for(i = 0; i < 12288; i++) {
		for(j = 0; j < 3; j++) {
			if(TEST_EPSILON(forceC[i].s[j], force[i].s[j], forceEpsilon)) {
				if(!invalidDataFound) {
					PRINT_FAIL();
					invalidDataFound = true;
				}
				printf("Variable force[%d].s[%d]: expected %f got %f (with epsilon).\n", i, j, forceC[i].s[j], force[i].s[j]);
			}
		}
	}
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(forceK)
		clReleaseMemObject(forceK);
	if(positionK)
		clReleaseMemObject(positionK);
	if(neighborListK)
		clReleaseMemObject(neighborListK);

	/* Dealloc variables */
	free(force);
	free(forceC);
	free(position);
	free(neighborList);

	/* Dealloc kernels */
	if(kernelCompute_Lj_Force)
		clReleaseKernel(kernelCompute_Lj_Force);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueCompute_Lj_Force)
		clReleaseCommandQueue(queueCompute_Lj_Force);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);


	return rv;
}
