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
 *            PREAMBLE(img_m, img_n, strel_m, strel_n, c_strel, c_strelSz, img, imgSz, dilated, dilatedSz, dilatedC, dilatedCSz);
 *            POSTAMBLE(img_m, img_n, strel_m, strel_n, c_strel, c_strelSz, img, imgSz, dilated, dilatedSz, dilatedC, dilatedCSz);
 *            LOOPPREAMBLE(img_m, img_n, strel_m, strel_n, c_strel, c_strelSz, img, imgSz, dilated, dilatedSz, dilatedC, dilatedCSz, loopFlag);
 *            LOOPPOSTAMBLE(img_m, img_n, strel_m, strel_n, c_strel, c_strelSz, img, imgSz, dilated, dilatedSz, dilatedC, dilatedCSz, loopFlag);
 *            CLEANUP(img_m, img_n, strel_m, strel_n, c_strel, c_strelSz, img, imgSz, dilated, dilatedSz, dilatedC, dilatedCSz);
 *        where:
 *            img_m: variable (int);
 *            img_n: variable (int);
 *            strel_m: variable (int);
 *            strel_n: variable (int);
 *            c_strel: variable (float *);
 *            c_strelSz: number of members in variable (unsigned int);
 *            img: variable (float *);
 *            imgSz: number of members in variable (unsigned int);
 *            dilated: variable (float *);
 *            dilatedSz: number of members in variable (unsigned int);
 *            dilatedC: variable (float *);
 *            dilatedCSz: number of members in variable (unsigned int);
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
	cl_command_queue queueDilate_Kernel = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelDilate_Kernel = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimDilate_Kernel = 1;
	size_t globalSizeDilate_Kernel[1] = {
		140272
	};
	size_t localSizeDilate_Kernel[1] = {
		176
	};

	/* Input/output variables */
	int img_m = 219;
	int img_n = 640;
	int strel_m = 25;
	int strel_n = 25;
	float *c_strel = malloc(625 * sizeof(float));
	cl_mem c_strelK = NULL;
	float *img = malloc(140160 * sizeof(float));
	cl_mem imgK = NULL;
	float *dilated = malloc(140160 * sizeof(float));
	float *dilatedC = malloc(140160 * sizeof(float));
	double dilatedEpsilon = 0.001;
	cl_mem dilatedK = NULL;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(img_m, img_n, strel_m, strel_n, c_strel, 625, img, 140160, dilated, 140160, dilatedC, 140160);
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

	/* Create command queue for dilate_kernel kernel */
	PRINT_STEP("Creating command queue for \"dilate_kernel\"...");
	queueDilate_Kernel = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create dilate_kernel kernel */
	PRINT_STEP("Creating kernel \"dilate_kernel\" from program...");
	kernelDilate_Kernel = clCreateKernel(program, "dilate_kernel", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	c_strelK = clCreateBuffer(context, CL_MEM_READ_ONLY, 625 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (c_strelK)"));
	imgK = clCreateBuffer(context, CL_MEM_READ_ONLY, 140160 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (imgK)"));
	dilatedK = clCreateBuffer(context, CL_MEM_READ_WRITE, 140160 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (dilatedK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for dilate_kernel */
	PRINT_STEP("Setting kernel arguments for \"dilate_kernel\"...");
	fRet = clSetKernelArg(kernelDilate_Kernel, 0, sizeof(int), &img_m);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (img_m)"));
	fRet = clSetKernelArg(kernelDilate_Kernel, 1, sizeof(int), &img_n);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (img_n)"));
	fRet = clSetKernelArg(kernelDilate_Kernel, 2, sizeof(int), &strel_m);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (strel_m)"));
	fRet = clSetKernelArg(kernelDilate_Kernel, 3, sizeof(int), &strel_n);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (strel_n)"));
	fRet = clSetKernelArg(kernelDilate_Kernel, 4, sizeof(cl_mem), &c_strelK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (c_strelK)"));
	fRet = clSetKernelArg(kernelDilate_Kernel, 5, sizeof(cl_mem), &imgK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (imgK)"));
	fRet = clSetKernelArg(kernelDilate_Kernel, 6, sizeof(cl_mem), &dilatedK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (dilatedK)"));
	PRINT_SUCCESS();

	do {
		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clSetKernelArg(kernelDilate_Kernel, 0, sizeof(int), &img_m);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (img_m)"));
		fRet = clSetKernelArg(kernelDilate_Kernel, 1, sizeof(int), &img_n);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (img_n)"));
		fRet = clSetKernelArg(kernelDilate_Kernel, 2, sizeof(int), &strel_m);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (strel_m)"));
		fRet = clSetKernelArg(kernelDilate_Kernel, 3, sizeof(int), &strel_n);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (strel_n)"));
		fRet = clEnqueueWriteBuffer(queueDilate_Kernel, c_strelK, CL_TRUE, 0, 625 * sizeof(float), c_strel, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (c_strelK)"));
		fRet = clEnqueueWriteBuffer(queueDilate_Kernel, imgK, CL_TRUE, 0, 140160 * sizeof(float), img, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (imgK)"));
		fRet = clEnqueueWriteBuffer(queueDilate_Kernel, dilatedK, CL_TRUE, 0, 140160 * sizeof(float), dilated, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (dilatedK)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueDilate_Kernel, kernelDilate_Kernel, workDimDilate_Kernel, NULL, globalSizeDilate_Kernel, localSizeDilate_Kernel, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueDilate_Kernel);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueDilate_Kernel, dilatedK, CL_TRUE, 0, 140160 * sizeof(float), dilated, 0, NULL, NULL);
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
		if(TEST_EPSILON(dilatedC[i],  dilated[i], dilatedEpsilon)) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable dilated[%d]: expected %f got %f (with epsilon).\n", i, dilatedC[i], dilated[i]);
		}
	}
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(c_strelK)
		clReleaseMemObject(c_strelK);
	if(imgK)
		clReleaseMemObject(imgK);
	if(dilatedK)
		clReleaseMemObject(dilatedK);

	/* Dealloc variables */
	free(c_strel);
	free(img);
	free(dilated);
	free(dilatedC);

	/* Dealloc kernels */
	if(kernelDilate_Kernel)
		clReleaseKernel(kernelDilate_Kernel);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueDilate_Kernel)
		clReleaseCommandQueue(queueDilate_Kernel);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);


	return rv;
}
