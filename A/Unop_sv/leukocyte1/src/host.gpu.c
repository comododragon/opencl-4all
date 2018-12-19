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
 *            PREAMBLE(grad_m, grad_x, grad_xSz, grad_y, grad_ySz, c_sin_angle, c_sin_angleSz, c_cos_angle, c_cos_angleSz, c_tX, c_tXSz, c_tY, c_tYSz, gicov, gicovSz, gicovC, gicovCSz, width, height);
 *            POSTAMBLE(grad_m, grad_x, grad_xSz, grad_y, grad_ySz, c_sin_angle, c_sin_angleSz, c_cos_angle, c_cos_angleSz, c_tX, c_tXSz, c_tY, c_tYSz, gicov, gicovSz, gicovC, gicovCSz, width, height);
 *            LOOPPREAMBLE(grad_m, grad_x, grad_xSz, grad_y, grad_ySz, c_sin_angle, c_sin_angleSz, c_cos_angle, c_cos_angleSz, c_tX, c_tXSz, c_tY, c_tYSz, gicov, gicovSz, gicovC, gicovCSz, width, height, loopFlag);
 *            LOOPPOSTAMBLE(grad_m, grad_x, grad_xSz, grad_y, grad_ySz, c_sin_angle, c_sin_angleSz, c_cos_angle, c_cos_angleSz, c_tX, c_tXSz, c_tY, c_tYSz, gicov, gicovSz, gicovC, gicovCSz, width, height, loopFlag);
 *            CLEANUP(grad_m, grad_x, grad_xSz, grad_y, grad_ySz, c_sin_angle, c_sin_angleSz, c_cos_angle, c_cos_angleSz, c_tX, c_tXSz, c_tY, c_tYSz, gicov, gicovSz, gicovC, gicovCSz, width, height);
 *        where:
 *            grad_m: variable (int);
 *            grad_x: variable (float *);
 *            grad_xSz: number of members in variable (unsigned int);
 *            grad_y: variable (float *);
 *            grad_ySz: number of members in variable (unsigned int);
 *            c_sin_angle: variable (float *);
 *            c_sin_angleSz: number of members in variable (unsigned int);
 *            c_cos_angle: variable (float *);
 *            c_cos_angleSz: number of members in variable (unsigned int);
 *            c_tX: variable (int *);
 *            c_tXSz: number of members in variable (unsigned int);
 *            c_tY: variable (int *);
 *            c_tYSz: number of members in variable (unsigned int);
 *            gicov: variable (float *);
 *            gicovSz: number of members in variable (unsigned int);
 *            gicovC: variable (float *);
 *            gicovCSz: number of members in variable (unsigned int);
 *            width: variable (int);
 *            height: variable (int);
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
	cl_command_queue queueGicov_Kernel = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelGicov_Kernel = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimGicov_Kernel = 1;
	size_t globalSizeGicov_Kernel[1] = {
		104448
	};
	size_t localSizeGicov_Kernel[1] = {
		256
	};

	/* Input/output variables */
	int grad_m = 219;
	float *grad_x = malloc(140160 * sizeof(float));
	cl_mem grad_xK = NULL;
	float *grad_y = malloc(140160 * sizeof(float));
	cl_mem grad_yK = NULL;
	float *c_sin_angle = malloc(150 * sizeof(float));
	cl_mem c_sin_angleK = NULL;
	float *c_cos_angle = malloc(150 * sizeof(float));
	cl_mem c_cos_angleK = NULL;
	int *c_tX = malloc(1050 * sizeof(int));
	cl_mem c_tXK = NULL;
	int *c_tY = malloc(1050 * sizeof(int));
	cl_mem c_tYK = NULL;
	float *gicov = malloc(140160 * sizeof(float));
	float *gicovC = malloc(140160 * sizeof(float));
	double gicovEpsilon = 0.001;
	cl_mem gicovK = NULL;
	int width = 175;
	int height = 596;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(grad_m, grad_x, 140160, grad_y, 140160, c_sin_angle, 150, c_cos_angle, 150, c_tX, 1050, c_tY, 1050, gicov, 140160, gicovC, 140160, width, height);
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

	/* Create command queue for GICOV_kernel kernel */
	PRINT_STEP("Creating command queue for \"GICOV_kernel\"...");
	queueGicov_Kernel = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create GICOV_kernel kernel */
	PRINT_STEP("Creating kernel \"GICOV_kernel\" from program...");
	kernelGicov_Kernel = clCreateKernel(program, "GICOV_kernel", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	grad_xK = clCreateBuffer(context, CL_MEM_READ_ONLY, 140160 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (grad_xK)"));
	grad_yK = clCreateBuffer(context, CL_MEM_READ_ONLY, 140160 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (grad_yK)"));
	c_sin_angleK = clCreateBuffer(context, CL_MEM_READ_ONLY, 150 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (c_sin_angleK)"));
	c_cos_angleK = clCreateBuffer(context, CL_MEM_READ_ONLY, 150 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (c_cos_angleK)"));
	c_tXK = clCreateBuffer(context, CL_MEM_READ_ONLY, 1050 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (c_tXK)"));
	c_tYK = clCreateBuffer(context, CL_MEM_READ_ONLY, 1050 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (c_tYK)"));
	gicovK = clCreateBuffer(context, CL_MEM_READ_WRITE, 140160 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (gicovK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for GICOV_kernel */
	PRINT_STEP("Setting kernel arguments for \"GICOV_kernel\"...");
	fRet = clSetKernelArg(kernelGicov_Kernel, 0, sizeof(int), &grad_m);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (grad_m)"));
	fRet = clSetKernelArg(kernelGicov_Kernel, 1, sizeof(cl_mem), &grad_xK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (grad_xK)"));
	fRet = clSetKernelArg(kernelGicov_Kernel, 2, sizeof(cl_mem), &grad_yK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (grad_yK)"));
	fRet = clSetKernelArg(kernelGicov_Kernel, 3, sizeof(cl_mem), &c_sin_angleK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (c_sin_angleK)"));
	fRet = clSetKernelArg(kernelGicov_Kernel, 4, sizeof(cl_mem), &c_cos_angleK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (c_cos_angleK)"));
	fRet = clSetKernelArg(kernelGicov_Kernel, 5, sizeof(cl_mem), &c_tXK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (c_tXK)"));
	fRet = clSetKernelArg(kernelGicov_Kernel, 6, sizeof(cl_mem), &c_tYK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (c_tYK)"));
	fRet = clSetKernelArg(kernelGicov_Kernel, 7, sizeof(cl_mem), &gicovK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (gicovK)"));
	fRet = clSetKernelArg(kernelGicov_Kernel, 8, sizeof(int), &width);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (width)"));
	fRet = clSetKernelArg(kernelGicov_Kernel, 9, sizeof(int), &height);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (height)"));
	PRINT_SUCCESS();

	do {
		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clSetKernelArg(kernelGicov_Kernel, 0, sizeof(int), &grad_m);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (grad_m)"));
		fRet = clEnqueueWriteBuffer(queueGicov_Kernel, grad_xK, CL_TRUE, 0, 140160 * sizeof(float), grad_x, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (grad_xK)"));
		fRet = clEnqueueWriteBuffer(queueGicov_Kernel, grad_yK, CL_TRUE, 0, 140160 * sizeof(float), grad_y, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (grad_yK)"));
		fRet = clEnqueueWriteBuffer(queueGicov_Kernel, c_sin_angleK, CL_TRUE, 0, 150 * sizeof(float), c_sin_angle, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (c_sin_angleK)"));
		fRet = clEnqueueWriteBuffer(queueGicov_Kernel, c_cos_angleK, CL_TRUE, 0, 150 * sizeof(float), c_cos_angle, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (c_cos_angleK)"));
		fRet = clEnqueueWriteBuffer(queueGicov_Kernel, c_tXK, CL_TRUE, 0, 1050 * sizeof(int), c_tX, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (c_tXK)"));
		fRet = clEnqueueWriteBuffer(queueGicov_Kernel, c_tYK, CL_TRUE, 0, 1050 * sizeof(int), c_tY, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (c_tYK)"));
		fRet = clEnqueueWriteBuffer(queueGicov_Kernel, gicovK, CL_TRUE, 0, 140160 * sizeof(float), gicov, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (gicovK)"));
		fRet = clSetKernelArg(kernelGicov_Kernel, 8, sizeof(int), &width);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (width)"));
		fRet = clSetKernelArg(kernelGicov_Kernel, 9, sizeof(int), &height);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (height)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueGicov_Kernel, kernelGicov_Kernel, workDimGicov_Kernel, NULL, globalSizeGicov_Kernel, localSizeGicov_Kernel, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueGicov_Kernel);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueGicov_Kernel, gicovK, CL_TRUE, 0, 140160 * sizeof(float), gicov, 0, NULL, NULL);
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
	for(i = 0; i < 140160; i++) {
		if(TEST_EPSILON(gicovC[i],  gicov[i], gicovEpsilon)) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable gicov[%d]: expected %f got %f (with epsilon).\n", i, gicovC[i], gicov[i]);
		}
	}
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(grad_xK)
		clReleaseMemObject(grad_xK);
	if(grad_yK)
		clReleaseMemObject(grad_yK);
	if(c_sin_angleK)
		clReleaseMemObject(c_sin_angleK);
	if(c_cos_angleK)
		clReleaseMemObject(c_cos_angleK);
	if(c_tXK)
		clReleaseMemObject(c_tXK);
	if(c_tYK)
		clReleaseMemObject(c_tYK);
	if(gicovK)
		clReleaseMemObject(gicovK);

	/* Dealloc variables */
	free(grad_x);
	free(grad_y);
	free(c_sin_angle);
	free(c_cos_angle);
	free(c_tX);
	free(c_tY);
	free(gicov);
	free(gicovC);

	/* Dealloc kernels */
	if(kernelGicov_Kernel)
		clReleaseKernel(kernelGicov_Kernel);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueGicov_Kernel)
		clReleaseCommandQueue(queueGicov_Kernel);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);


	return rv;
}
