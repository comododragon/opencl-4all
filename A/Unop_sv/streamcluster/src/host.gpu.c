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
 *  * @brief Header where pre/postamble macro functions should be located.
 *   *        Function headers:
 *    *            PREAMBLE(p_weight, p_weightSz, p_assign, p_assignSz, p_cost, p_costSz, coord_d, coord_dSz, work_mem_d, work_mem_dSz, work_mem_dC, work_mem_dCSz, center_table_d, center_table_dSz, switch_membership_d, switch_membership_dSz, switch_membership_dC, switch_membership_dCSz, dim, x, K);
 *     *            POSTAMBLE(p_weight, p_weightSz, p_assign, p_assignSz, p_cost, p_costSz, coord_d, coord_dSz, work_mem_d, work_mem_dSz, work_mem_dC, work_mem_dCSz, center_table_d, center_table_dSz, switch_membership_d, switch_membership_dSz, switch_membership_dC, switch_membership_dCSz, dim, x, K);
 *      *            LOOPPREAMBLE(p_weight, p_weightSz, p_assign, p_assignSz, p_cost, p_costSz, coord_d, coord_dSz, work_mem_d, work_mem_dSz, work_mem_dC, work_mem_dCSz, center_table_d, center_table_dSz, switch_membership_d, switch_membership_dSz, switch_membership_dC, switch_membership_dCSz, dim, x, K, loopFlag);
 *       *            LOOPPOSTAMBLE(p_weight, p_weightSz, p_assign, p_assignSz, p_cost, p_costSz, coord_d, coord_dSz, work_mem_d, work_mem_dSz, work_mem_dC, work_mem_dCSz, center_table_d, center_table_dSz, switch_membership_d, switch_membership_dSz, switch_membership_dC, switch_membership_dCSz, dim, x, K, loopFlag);
 *        *            CLEANUP(p_weight, p_weightSz, p_assign, p_assignSz, p_cost, p_costSz, coord_d, coord_dSz, work_mem_d, work_mem_dSz, work_mem_dC, work_mem_dCSz, center_table_d, center_table_dSz, switch_membership_d, switch_membership_dSz, switch_membership_dC, switch_membership_dCSz, dim, x, K);
 *         *        where:
 *          *            p_weight: variable (float *);
 *           *            p_weightSz: number of members in variable (unsigned int);
 *            *            p_assign: variable (long *);
 *             *            p_assignSz: number of members in variable (unsigned int);
 *              *            p_cost: variable (float *);
 *               *            p_costSz: number of members in variable (unsigned int);
 *                *            coord_d: variable (float *);
 *                 *            coord_dSz: number of members in variable (unsigned int);
 *                  *            work_mem_d: variable (float *);
 *                   *            work_mem_dSz: number of members in variable (unsigned int);
 *                    *            work_mem_dC: variable (float *);
 *                     *            work_mem_dCSz: number of members in variable (unsigned int);
 *                      *            center_table_d: variable (int *);
 *                       *            center_table_dSz: number of members in variable (unsigned int);
 *                        *            switch_membership_d: variable (char *);
 *                         *            switch_membership_dSz: number of members in variable (unsigned int);
 *                          *            switch_membership_dC: variable (char *);
 *                           *            switch_membership_dCSz: number of members in variable (unsigned int);
 *                            *            dim: variable (int);
 *                             *            x: variable (long);
 *                              *            K: variable (int);
 *                               *            loopFlag: loop condition variable (bool).
 *                                */
#include "prepostambles.h"

/**
 *  * @brief Test if two operands are outside an epsilon range.
 *   *
 *    * @param a First operand.
 *     * @param b Second operand.
 *      * @param e Epsilon value.
 *       */
#define TEST_EPSILON(a, b, e) (((a > b) && (a - b > e)) || ((b >= a) && (b - a > e)))

/**
 *  * @brief Standard statements for function error handling and printing.
 *   *
 *    * @param funcName Function name that failed.
 *     */
#define FUNCTION_ERROR_STATEMENTS(funcName) {\
	rv = EXIT_FAILURE;\
	PRINT_FAIL();\
	fprintf(stderr, "Error: %s failed with return code %d.\n", funcName, fRet);\
}

/**
 *  * @brief Standard statements for POSIX error handling and printing.
 *   *
 *    * @param arg Arbitrary string to the printed at the end of error string.
 *     */
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
	cl_command_queue queuePgain_Kernel = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelPgain_Kernel = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimPgain_Kernel = 2;
	size_t globalSizePgain_Kernel[2] = {
		65536, 1
	};
	size_t localSizePgain_Kernel[2] = {
		256, 1
	};

	/* Input/output variables */
	float *p_weight = malloc(65536 * sizeof(float));
	cl_mem p_weightK = NULL;
	long *p_assign = malloc(65536 * sizeof(long));
	cl_mem p_assignK = NULL;
	float *p_cost = malloc(65536 * sizeof(float));
	cl_mem p_costK = NULL;
	float *coord_d = malloc(4194304 * sizeof(float));
	cl_mem coord_dK = NULL;
	float *work_mem_d = malloc(786432 * sizeof(float));
	float *work_mem_dC = malloc(786432 * sizeof(float));
	double work_mem_dEpsilon = 0.01;
	cl_mem work_mem_dK = NULL;
	int *center_table_d = malloc(65536 * sizeof(int));
	cl_mem center_table_dK = NULL;
	char *switch_membership_d = malloc(65536 * sizeof(char));
	char *switch_membership_dC = malloc(65536 * sizeof(char));
	cl_mem switch_membership_dK = NULL;
	int dim = 64;
	long x = 48541;
	int K = 11;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(p_weight, 65536, p_assign, 65536, p_cost, 65536, coord_d, 4194304, work_mem_d, 786432, work_mem_dC, 786432, center_table_d, 65536, switch_membership_d, 65536, switch_membership_dC, 65536, dim, x, K);
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

	/* Create command queue for pgain_kernel kernel */
	PRINT_STEP("Creating command queue for \"pgain_kernel\"...");
	queuePgain_Kernel = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create pgain_kernel kernel */
	PRINT_STEP("Creating kernel \"pgain_kernel\" from program...");
	kernelPgain_Kernel = clCreateKernel(program, "pgain_kernel", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	p_weightK = clCreateBuffer(context, CL_MEM_READ_ONLY, 65536 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (p_weightK)"));
	p_assignK = clCreateBuffer(context, CL_MEM_READ_ONLY, 65536 * sizeof(long), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (p_assignK)"));
	p_costK = clCreateBuffer(context, CL_MEM_READ_ONLY, 65536 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (p_costK)"));
	coord_dK = clCreateBuffer(context, CL_MEM_READ_ONLY, 4194304 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (coord_dK)"));
	work_mem_dK = clCreateBuffer(context, CL_MEM_READ_WRITE, 786432 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (work_mem_dK)"));
	center_table_dK = clCreateBuffer(context, CL_MEM_READ_ONLY, 65536 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (center_table_dK)"));
	switch_membership_dK = clCreateBuffer(context, CL_MEM_READ_WRITE, 65536 * sizeof(char), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (switch_membership_dK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for pgain_kernel */
	PRINT_STEP("Setting kernel arguments for \"pgain_kernel\"...");
	fRet = clSetKernelArg(kernelPgain_Kernel, 0, sizeof(cl_mem), &p_weightK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (p_weightK)"));
	fRet = clSetKernelArg(kernelPgain_Kernel, 1, sizeof(cl_mem), &p_assignK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (p_assignK)"));
	fRet = clSetKernelArg(kernelPgain_Kernel, 2, sizeof(cl_mem), &p_costK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (p_costK)"));
	fRet = clSetKernelArg(kernelPgain_Kernel, 3, sizeof(cl_mem), &coord_dK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (coord_dK)"));
	fRet = clSetKernelArg(kernelPgain_Kernel, 4, sizeof(cl_mem), &work_mem_dK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (work_mem_dK)"));
	fRet = clSetKernelArg(kernelPgain_Kernel, 5, sizeof(cl_mem), &center_table_dK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (center_table_dK)"));
	fRet = clSetKernelArg(kernelPgain_Kernel, 6, sizeof(cl_mem), &switch_membership_dK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (switch_membership_dK)"));
	fRet = clSetKernelArg(kernelPgain_Kernel, 7, 256 * sizeof(float), NULL);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (__local 7)"));
	fRet = clSetKernelArg(kernelPgain_Kernel, 8, sizeof(int), &dim);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (dim)"));
	fRet = clSetKernelArg(kernelPgain_Kernel, 9, sizeof(long), &x);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (x)"));
	fRet = clSetKernelArg(kernelPgain_Kernel, 10, sizeof(int), &K);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (K)"));
	PRINT_SUCCESS();

	do {
		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clEnqueueWriteBuffer(queuePgain_Kernel, p_weightK, CL_TRUE, 0, 65536 * sizeof(float), p_weight, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (p_weightK)"));
		fRet = clEnqueueWriteBuffer(queuePgain_Kernel, p_assignK, CL_TRUE, 0, 65536 * sizeof(long), p_assign, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (p_assignK)"));
		fRet = clEnqueueWriteBuffer(queuePgain_Kernel, p_costK, CL_TRUE, 0, 65536 * sizeof(float), p_cost, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (p_costK)"));
		fRet = clEnqueueWriteBuffer(queuePgain_Kernel, coord_dK, CL_TRUE, 0, 4194304 * sizeof(float), coord_d, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (coord_dK)"));
		fRet = clEnqueueWriteBuffer(queuePgain_Kernel, work_mem_dK, CL_TRUE, 0, 786432 * sizeof(float), work_mem_d, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (work_mem_dK)"));
		fRet = clEnqueueWriteBuffer(queuePgain_Kernel, center_table_dK, CL_TRUE, 0, 65536 * sizeof(int), center_table_d, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (center_table_dK)"));
		fRet = clEnqueueWriteBuffer(queuePgain_Kernel, switch_membership_dK, CL_TRUE, 0, 65536 * sizeof(char), switch_membership_d, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (switch_membership_dK)"));
		fRet = clSetKernelArg(kernelPgain_Kernel, 8, sizeof(int), &dim);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (dim)"));
		fRet = clSetKernelArg(kernelPgain_Kernel, 9, sizeof(long), &x);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (x)"));
		fRet = clSetKernelArg(kernelPgain_Kernel, 10, sizeof(int), &K);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (K)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queuePgain_Kernel, kernelPgain_Kernel, workDimPgain_Kernel, NULL, globalSizePgain_Kernel, localSizePgain_Kernel, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queuePgain_Kernel);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queuePgain_Kernel, work_mem_dK, CL_TRUE, 0, 786432 * sizeof(float), work_mem_d, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queuePgain_Kernel, switch_membership_dK, CL_TRUE, 0, 65536 * sizeof(char), switch_membership_d, 0, NULL, NULL);
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
	for(i = 0; i < 786432; i++) {
		if(TEST_EPSILON(work_mem_dC[i],  work_mem_d[i], work_mem_dEpsilon)) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable work_mem_d[%d]: expected %f got %f (with epsilon).\n", i, work_mem_dC[i], work_mem_d[i]);
		}
	}
	for(i = 0; i < 65536; i++) {
		if(switch_membership_dC[i] != switch_membership_d[i]) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable switch_membership_d[%d]: expected %c got %c.\n", i, switch_membership_dC[i], switch_membership_d[i]);
		}
	}
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(p_weightK)
		clReleaseMemObject(p_weightK);
	if(p_assignK)
		clReleaseMemObject(p_assignK);
	if(p_costK)
		clReleaseMemObject(p_costK);
	if(coord_dK)
		clReleaseMemObject(coord_dK);
	if(work_mem_dK)
		clReleaseMemObject(work_mem_dK);
	if(center_table_dK)
		clReleaseMemObject(center_table_dK);
	if(switch_membership_dK)
		clReleaseMemObject(switch_membership_dK);

	/* Dealloc variables */
	free(p_weight);
	free(p_assign);
	free(p_cost);
	free(coord_d);
	free(work_mem_d);
	free(work_mem_dC);
	free(center_table_d);
	free(switch_membership_d);
	free(switch_membership_dC);

	/* Dealloc kernels */
	if(kernelPgain_Kernel)
		clReleaseKernel(kernelPgain_Kernel);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queuePgain_Kernel)
		clReleaseCommandQueue(queuePgain_Kernel);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);


	return rv;
}
