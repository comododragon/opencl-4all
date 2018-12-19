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
 *            PREAMBLE(variables, variablesSz, areas, areasSz, step_factors, step_factorsSz, step_factorsC, step_factorsCSz, nelr);
 *            POSTAMBLE(variables, variablesSz, areas, areasSz, step_factors, step_factorsSz, step_factorsC, step_factorsCSz, nelr);
 *            LOOPPREAMBLE(variables, variablesSz, areas, areasSz, step_factors, step_factorsSz, step_factorsC, step_factorsCSz, nelr, loopFlag);
 *            LOOPPOSTAMBLE(variables, variablesSz, areas, areasSz, step_factors, step_factorsSz, step_factorsC, step_factorsCSz, nelr, loopFlag);
 *            CLEANUP(variables, variablesSz, areas, areasSz, step_factors, step_factorsSz, step_factorsC, step_factorsCSz, nelr);
 *        where:
 *            variables: variable (float *);
 *            variablesSz: number of members in variable (unsigned int);
 *            areas: variable (float *);
 *            areasSz: number of members in variable (unsigned int);
 *            step_factors: variable (float *);
 *            step_factorsSz: number of members in variable (unsigned int);
 *            step_factorsC: variable (float *);
 *            step_factorsCSz: number of members in variable (unsigned int);
 *            nelr: variable (int);
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
	cl_command_queue queueCompute_Step_Factor = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelCompute_Step_Factor = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimCompute_Step_Factor = 1;
	size_t globalSizeCompute_Step_Factor[1] = {
		97152
	};
	size_t localSizeCompute_Step_Factor[1] = {
		192
	};

	/* Input/output variables */
	float *variables = malloc(485760 * sizeof(float));
	cl_mem variablesK = NULL;
	float *areas = malloc(97152 * sizeof(float));
	cl_mem areasK = NULL;
	float *step_factors = malloc(97152 * sizeof(float));
	float *step_factorsC = malloc(97152 * sizeof(float));
	double step_factorsEpsilon = 0.01;
	cl_mem step_factorsK = NULL;
	int nelr = 97152;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(variables, 485760, areas, 97152, step_factors, 97152, step_factorsC, 97152, nelr);
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

	/* Create command queue for compute_step_factor kernel */
	PRINT_STEP("Creating command queue for \"compute_step_factor\"...");
	queueCompute_Step_Factor = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create compute_step_factor kernel */
	PRINT_STEP("Creating kernel \"compute_step_factor\" from program...");
	kernelCompute_Step_Factor = clCreateKernel(program, "compute_step_factor", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	variablesK = clCreateBuffer(context, CL_MEM_READ_ONLY, 485760 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (variablesK)"));
	areasK = clCreateBuffer(context, CL_MEM_READ_ONLY, 97152 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (areasK)"));
	step_factorsK = clCreateBuffer(context, CL_MEM_READ_WRITE, 97152 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (step_factorsK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for compute_step_factor */
	PRINT_STEP("Setting kernel arguments for \"compute_step_factor\"...");
	fRet = clSetKernelArg(kernelCompute_Step_Factor, 0, sizeof(cl_mem), &variablesK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (variablesK)"));
	fRet = clSetKernelArg(kernelCompute_Step_Factor, 1, sizeof(cl_mem), &areasK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (areasK)"));
	fRet = clSetKernelArg(kernelCompute_Step_Factor, 2, sizeof(cl_mem), &step_factorsK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (step_factorsK)"));
	fRet = clSetKernelArg(kernelCompute_Step_Factor, 3, sizeof(int), &nelr);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (nelr)"));
	PRINT_SUCCESS();

	do {
		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clEnqueueWriteBuffer(queueCompute_Step_Factor, variablesK, CL_TRUE, 0, 485760 * sizeof(float), variables, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (variablesK)"));
		fRet = clEnqueueWriteBuffer(queueCompute_Step_Factor, areasK, CL_TRUE, 0, 97152 * sizeof(float), areas, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (areasK)"));
		fRet = clEnqueueWriteBuffer(queueCompute_Step_Factor, step_factorsK, CL_TRUE, 0, 97152 * sizeof(float), step_factors, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (step_factorsK)"));
		fRet = clSetKernelArg(kernelCompute_Step_Factor, 3, sizeof(int), &nelr);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (nelr)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueCompute_Step_Factor, kernelCompute_Step_Factor, workDimCompute_Step_Factor, NULL, globalSizeCompute_Step_Factor, localSizeCompute_Step_Factor, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueCompute_Step_Factor);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueCompute_Step_Factor, step_factorsK, CL_TRUE, 0, 97152 * sizeof(float), step_factors, 0, NULL, NULL);
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
	for(i = 0; i < 97152; i++) {
		if(TEST_EPSILON(step_factorsC[i],  step_factors[i], step_factorsEpsilon)) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable step_factors[%d]: expected %f got %f (with epsilon).\n", i, step_factorsC[i], step_factors[i]);
		}
	}
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(variablesK)
		clReleaseMemObject(variablesK);
	if(areasK)
		clReleaseMemObject(areasK);
	if(step_factorsK)
		clReleaseMemObject(step_factorsK);

	/* Dealloc variables */
	free(variables);
	free(areas);
	free(step_factors);
	free(step_factorsC);

	/* Dealloc kernels */
	if(kernelCompute_Step_Factor)
		clReleaseKernel(kernelCompute_Step_Factor);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueCompute_Step_Factor)
		clReleaseCommandQueue(queueCompute_Step_Factor);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);


	return rv;
}
