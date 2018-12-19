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
 *            PREAMBLE(p, pSz, tIn, tInSz, tOut, tOutSz, tOutC, tOutCSz, sdc, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc);
 *            POSTAMBLE(p, pSz, tIn, tInSz, tOut, tOutSz, tOutC, tOutCSz, sdc, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc);
 *            LOOPPREAMBLE(p, pSz, tIn, tInSz, tOut, tOutSz, tOutC, tOutCSz, sdc, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, loopFlag);
 *            LOOPPOSTAMBLE(p, pSz, tIn, tInSz, tOut, tOutSz, tOutC, tOutCSz, sdc, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, loopFlag);
 *            CLEANUP(p, pSz, tIn, tInSz, tOut, tOutSz, tOutC, tOutCSz, sdc, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc);
 *        where:
 *            p: variable (float *);
 *            pSz: number of members in variable (unsigned int);
 *            tIn: variable (float *);
 *            tInSz: number of members in variable (unsigned int);
 *            tOut: variable (float *);
 *            tOutSz: number of members in variable (unsigned int);
 *            tOutC: variable (float *);
 *            tOutCSz: number of members in variable (unsigned int);
 *            sdc: variable (float);
 *            nx: variable (int);
 *            ny: variable (int);
 *            nz: variable (int);
 *            ce: variable (float);
 *            cw: variable (float);
 *            cn: variable (float);
 *            cs: variable (float);
 *            ct: variable (float);
 *            cb: variable (float);
 *            cc: variable (float);
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
	cl_command_queue queueHotspotopt1 = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelHotspotopt1 = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimHotspotopt1 = 2;
	size_t globalSizeHotspotopt1[2] = {
		512, 512
	};
	size_t localSizeHotspotopt1[2] = {
		64, 4
	};

	/* Input/output variables */
	float *p = malloc(2097152 * sizeof(float));
	cl_mem pK = NULL;
	float *tIn = malloc(2097152 * sizeof(float));
	cl_mem tInK = NULL;
	float *tOut = malloc(2097152 * sizeof(float));
	float *tOutC = malloc(2097152 * sizeof(float));
	double tOutEpsilon = 0.01;
	cl_mem tOutK = NULL;
	float sdc = 0.341333;
	int nx = 512;
	int ny = 512;
	int nz = 8;
	float ce = 0.034133;
	float cw = 0.034133;
	float cn = 0.034133;
	float cs = 0.034133;
	float ct = 0.000533;
	float cb = 0.000533;
	float cc = 0.861867;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(p, 2097152, tIn, 2097152, tOut, 2097152, tOutC, 2097152, sdc, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc);
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

	/* Create command queue for hotspotOpt1 kernel */
	PRINT_STEP("Creating command queue for \"hotspotOpt1\"...");
	queueHotspotopt1 = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create hotspotOpt1 kernel */
	PRINT_STEP("Creating kernel \"hotspotOpt1\" from program...");
	kernelHotspotopt1 = clCreateKernel(program, "hotspotOpt1", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	pK = clCreateBuffer(context, CL_MEM_READ_WRITE, 2097152 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (pK)"));
	tInK = clCreateBuffer(context, CL_MEM_READ_WRITE, 2097152 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (tInK)"));
	tOutK = clCreateBuffer(context, CL_MEM_READ_WRITE, 2097152 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (tOutK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for hotspotOpt1 */
	PRINT_STEP("Setting kernel arguments for \"hotspotOpt1\"...");
	fRet = clSetKernelArg(kernelHotspotopt1, 0, sizeof(cl_mem), &pK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (pK)"));
	fRet = clSetKernelArg(kernelHotspotopt1, 1, sizeof(cl_mem), &tInK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (tInK)"));
	fRet = clSetKernelArg(kernelHotspotopt1, 2, sizeof(cl_mem), &tOutK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (tOutK)"));
	fRet = clSetKernelArg(kernelHotspotopt1, 3, sizeof(float), &sdc);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (sdc)"));
	fRet = clSetKernelArg(kernelHotspotopt1, 4, sizeof(int), &nx);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (nx)"));
	fRet = clSetKernelArg(kernelHotspotopt1, 5, sizeof(int), &ny);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (ny)"));
	fRet = clSetKernelArg(kernelHotspotopt1, 6, sizeof(int), &nz);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (nz)"));
	fRet = clSetKernelArg(kernelHotspotopt1, 7, sizeof(float), &ce);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (ce)"));
	fRet = clSetKernelArg(kernelHotspotopt1, 8, sizeof(float), &cw);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (cw)"));
	fRet = clSetKernelArg(kernelHotspotopt1, 9, sizeof(float), &cn);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (cn)"));
	fRet = clSetKernelArg(kernelHotspotopt1, 10, sizeof(float), &cs);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (cs)"));
	fRet = clSetKernelArg(kernelHotspotopt1, 11, sizeof(float), &ct);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (ct)"));
	fRet = clSetKernelArg(kernelHotspotopt1, 12, sizeof(float), &cb);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (cb)"));
	fRet = clSetKernelArg(kernelHotspotopt1, 13, sizeof(float), &cc);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (cc)"));
	PRINT_SUCCESS();

	do {
		/* Calling loop preamble function */
		PRINT_STEP("[%d] Calling loop preamble function...", i);
		LOOPPREAMBLE(p, 2097152, tIn, 2097152, tOut, 2097152, tOutC, 2097152, sdc, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, loopFlag);
		PRINT_SUCCESS();

		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clEnqueueWriteBuffer(queueHotspotopt1, pK, CL_TRUE, 0, 2097152 * sizeof(float), p, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (pK)"));
		fRet = clEnqueueWriteBuffer(queueHotspotopt1, tInK, CL_TRUE, 0, 2097152 * sizeof(float), tIn, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (tInK)"));
		fRet = clEnqueueWriteBuffer(queueHotspotopt1, tOutK, CL_TRUE, 0, 2097152 * sizeof(float), tOut, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (tOutK)"));
		fRet = clSetKernelArg(kernelHotspotopt1, 3, sizeof(float), &sdc);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (sdc)"));
		fRet = clSetKernelArg(kernelHotspotopt1, 4, sizeof(int), &nx);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (nx)"));
		fRet = clSetKernelArg(kernelHotspotopt1, 5, sizeof(int), &ny);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (ny)"));
		fRet = clSetKernelArg(kernelHotspotopt1, 6, sizeof(int), &nz);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (nz)"));
		fRet = clSetKernelArg(kernelHotspotopt1, 7, sizeof(float), &ce);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (ce)"));
		fRet = clSetKernelArg(kernelHotspotopt1, 8, sizeof(float), &cw);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (cw)"));
		fRet = clSetKernelArg(kernelHotspotopt1, 9, sizeof(float), &cn);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (cn)"));
		fRet = clSetKernelArg(kernelHotspotopt1, 10, sizeof(float), &cs);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (cs)"));
		fRet = clSetKernelArg(kernelHotspotopt1, 11, sizeof(float), &ct);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (ct)"));
		fRet = clSetKernelArg(kernelHotspotopt1, 12, sizeof(float), &cb);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (cb)"));
		fRet = clSetKernelArg(kernelHotspotopt1, 13, sizeof(float), &cc);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (cc)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueHotspotopt1, kernelHotspotopt1, workDimHotspotopt1, NULL, globalSizeHotspotopt1, localSizeHotspotopt1, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueHotspotopt1);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueHotspotopt1, pK, CL_TRUE, 0, 2097152 * sizeof(float), p, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueHotspotopt1, tInK, CL_TRUE, 0, 2097152 * sizeof(float), tIn, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueHotspotopt1, tOutK, CL_TRUE, 0, 2097152 * sizeof(float), tOut, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueReadBuffer"));
		PRINT_SUCCESS();

		/* Calling loop postamble function */
		PRINT_STEP("[%d] Calling loop postamble function...", i);
		LOOPPOSTAMBLE(p, 2097152, tIn, 2097152, tOut, 2097152, tOutC, 2097152, sdc, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, loopFlag);
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
	for(i = 0; i < 2097152; i++) {
		if(TEST_EPSILON(tOutC[i],  tOut[i], tOutEpsilon)) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable tOut[%d]: expected %f got %f (with epsilon).\n", i, tOutC[i], tOut[i]);
		}
	}
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(pK)
		clReleaseMemObject(pK);
	if(tInK)
		clReleaseMemObject(tInK);
	if(tOutK)
		clReleaseMemObject(tOutK);

	/* Dealloc variables */
	free(p);
	free(tIn);
	free(tOut);
	free(tOutC);

	/* Dealloc kernels */
	if(kernelHotspotopt1)
		clReleaseKernel(kernelHotspotopt1);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueHotspotopt1)
		clReleaseCommandQueue(queueHotspotopt1);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);

	/* Calling cleanup function */
	CLEANUP(p, 2097152, tIn, 2097152, tOut, 2097152, tOutC, 2097152, sdc, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc);

	return rv;
}
