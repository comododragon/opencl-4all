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
 *            PREAMBLE(d_lambda, d_Nr, d_Nc, d_Ne, d_iN, d_iNSz, d_iS, d_iSSz, d_jE, d_jESz, d_jW, d_jWSz, d_dN, d_dNSz, d_dS, d_dSSz, d_dE, d_dESz, d_dW, d_dWSz, d_q0sqr, d_c, d_cSz, d_I, d_ISz);
 *            POSTAMBLE(d_lambda, d_Nr, d_Nc, d_Ne, d_iN, d_iNSz, d_iS, d_iSSz, d_jE, d_jESz, d_jW, d_jWSz, d_dN, d_dNSz, d_dS, d_dSSz, d_dE, d_dESz, d_dW, d_dWSz, d_q0sqr, d_c, d_cSz, d_I, d_ISz);
 *            LOOPPREAMBLE(d_lambda, d_Nr, d_Nc, d_Ne, d_iN, d_iNSz, d_iS, d_iSSz, d_jE, d_jESz, d_jW, d_jWSz, d_dN, d_dNSz, d_dS, d_dSSz, d_dE, d_dESz, d_dW, d_dWSz, d_q0sqr, d_c, d_cSz, d_I, d_ISz, loopFlag);
 *            LOOPPOSTAMBLE(d_lambda, d_Nr, d_Nc, d_Ne, d_iN, d_iNSz, d_iS, d_iSSz, d_jE, d_jESz, d_jW, d_jWSz, d_dN, d_dNSz, d_dS, d_dSSz, d_dE, d_dESz, d_dW, d_dWSz, d_q0sqr, d_c, d_cSz, d_I, d_ISz, loopFlag);
 *            CLEANUP(d_lambda, d_Nr, d_Nc, d_Ne, d_iN, d_iNSz, d_iS, d_iSSz, d_jE, d_jESz, d_jW, d_jWSz, d_dN, d_dNSz, d_dS, d_dSSz, d_dE, d_dESz, d_dW, d_dWSz, d_q0sqr, d_c, d_cSz, d_I, d_ISz);
 *        where:
 *            d_lambda: variable (float);
 *            d_Nr: variable (int);
 *            d_Nc: variable (int);
 *            d_Ne: variable (long);
 *            d_iN: variable (int *);
 *            d_iNSz: number of members in variable (unsigned int);
 *            d_iS: variable (int *);
 *            d_iSSz: number of members in variable (unsigned int);
 *            d_jE: variable (int *);
 *            d_jESz: number of members in variable (unsigned int);
 *            d_jW: variable (int *);
 *            d_jWSz: number of members in variable (unsigned int);
 *            d_dN: variable (float *);
 *            d_dNSz: number of members in variable (unsigned int);
 *            d_dS: variable (float *);
 *            d_dSSz: number of members in variable (unsigned int);
 *            d_dE: variable (float *);
 *            d_dESz: number of members in variable (unsigned int);
 *            d_dW: variable (float *);
 *            d_dWSz: number of members in variable (unsigned int);
 *            d_q0sqr: variable (float);
 *            d_c: variable (float *);
 *            d_cSz: number of members in variable (unsigned int);
 *            d_I: variable (float *);
 *            d_ISz: number of members in variable (unsigned int);
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
	cl_command_queue queueSrad_Kernel = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelSrad_Kernel = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimSrad_Kernel = 1;
	size_t globalSizeSrad_Kernel[1] = {
		230144
	};
	size_t localSizeSrad_Kernel[1] = {
		256
	};

	/* Input/output variables */
	float d_lambda = 0.5;
	int d_Nr = 502;
	int d_Nc = 458;
	long d_Ne = 229916;
	int *d_iN = malloc(502 * sizeof(int));
	cl_mem d_iNK = NULL;
	int *d_iS = malloc(502 * sizeof(int));
	cl_mem d_iSK = NULL;
	int *d_jE = malloc(458 * sizeof(int));
	cl_mem d_jEK = NULL;
	int *d_jW = malloc(458 * sizeof(int));
	cl_mem d_jWK = NULL;
	float *d_dN = malloc(229916 * sizeof(float));
	cl_mem d_dNK = NULL;
	float *d_dS = malloc(229916 * sizeof(float));
	cl_mem d_dSK = NULL;
	float *d_dE = malloc(229916 * sizeof(float));
	cl_mem d_dEK = NULL;
	float *d_dW = malloc(229916 * sizeof(float));
	cl_mem d_dWK = NULL;
	float d_q0sqr;
	float *d_c = malloc(229916 * sizeof(float));
	cl_mem d_cK = NULL;
	float *d_I = malloc(229916 * sizeof(float));
	cl_mem d_IK = NULL;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(d_lambda, d_Nr, d_Nc, d_Ne, d_iN, 502, d_iS, 502, d_jE, 458, d_jW, 458, d_dN, 229916, d_dS, 229916, d_dE, 229916, d_dW, 229916, d_q0sqr, d_c, 229916, d_I, 229916);
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

	/* Create command queue for srad_kernel kernel */
	PRINT_STEP("Creating command queue for \"srad_kernel\"...");
	queueSrad_Kernel = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create srad_kernel kernel */
	PRINT_STEP("Creating kernel \"srad_kernel\" from program...");
	kernelSrad_Kernel = clCreateKernel(program, "srad_kernel", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	d_iNK = clCreateBuffer(context, CL_MEM_READ_ONLY, 502 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (d_iNK)"));
	d_iSK = clCreateBuffer(context, CL_MEM_READ_ONLY, 502 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (d_iSK)"));
	d_jEK = clCreateBuffer(context, CL_MEM_READ_ONLY, 458 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (d_jEK)"));
	d_jWK = clCreateBuffer(context, CL_MEM_READ_ONLY, 458 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (d_jWK)"));
	d_dNK = clCreateBuffer(context, CL_MEM_READ_WRITE, 229916 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (d_dNK)"));
	d_dSK = clCreateBuffer(context, CL_MEM_READ_WRITE, 229916 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (d_dSK)"));
	d_dEK = clCreateBuffer(context, CL_MEM_READ_WRITE, 229916 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (d_dEK)"));
	d_dWK = clCreateBuffer(context, CL_MEM_READ_WRITE, 229916 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (d_dWK)"));
	d_cK = clCreateBuffer(context, CL_MEM_READ_WRITE, 229916 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (d_cK)"));
	d_IK = clCreateBuffer(context, CL_MEM_READ_ONLY, 229916 * sizeof(float), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (d_IK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for srad_kernel */
	PRINT_STEP("Setting kernel arguments for \"srad_kernel\"...");
	fRet = clSetKernelArg(kernelSrad_Kernel, 0, sizeof(float), &d_lambda);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_lambda)"));
	fRet = clSetKernelArg(kernelSrad_Kernel, 1, sizeof(int), &d_Nr);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_Nr)"));
	fRet = clSetKernelArg(kernelSrad_Kernel, 2, sizeof(int), &d_Nc);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_Nc)"));
	fRet = clSetKernelArg(kernelSrad_Kernel, 3, sizeof(long), &d_Ne);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_Ne)"));
	fRet = clSetKernelArg(kernelSrad_Kernel, 4, sizeof(cl_mem), &d_iNK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_iNK)"));
	fRet = clSetKernelArg(kernelSrad_Kernel, 5, sizeof(cl_mem), &d_iSK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_iSK)"));
	fRet = clSetKernelArg(kernelSrad_Kernel, 6, sizeof(cl_mem), &d_jEK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_jEK)"));
	fRet = clSetKernelArg(kernelSrad_Kernel, 7, sizeof(cl_mem), &d_jWK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_jWK)"));
	fRet = clSetKernelArg(kernelSrad_Kernel, 8, sizeof(cl_mem), &d_dNK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_dNK)"));
	fRet = clSetKernelArg(kernelSrad_Kernel, 9, sizeof(cl_mem), &d_dSK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_dSK)"));
	fRet = clSetKernelArg(kernelSrad_Kernel, 10, sizeof(cl_mem), &d_dEK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_dEK)"));
	fRet = clSetKernelArg(kernelSrad_Kernel, 11, sizeof(cl_mem), &d_dWK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_dWK)"));
	fRet = clSetKernelArg(kernelSrad_Kernel, 12, sizeof(float), &d_q0sqr);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_q0sqr)"));
	fRet = clSetKernelArg(kernelSrad_Kernel, 13, sizeof(cl_mem), &d_cK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_cK)"));
	fRet = clSetKernelArg(kernelSrad_Kernel, 14, sizeof(cl_mem), &d_IK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_IK)"));
	PRINT_SUCCESS();

	do {
		/* Calling loop preamble function */
		PRINT_STEP("[%d] Calling loop preamble function...", i);
		LOOPPREAMBLE(d_lambda, d_Nr, d_Nc, d_Ne, d_iN, 502, d_iS, 502, d_jE, 458, d_jW, 458, d_dN, 229916, d_dS, 229916, d_dE, 229916, d_dW, 229916, d_q0sqr, d_c, 229916, d_I, 229916, loopFlag);
		PRINT_SUCCESS();

		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clSetKernelArg(kernelSrad_Kernel, 0, sizeof(float), &d_lambda);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_lambda)"));
		fRet = clSetKernelArg(kernelSrad_Kernel, 1, sizeof(int), &d_Nr);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_Nr)"));
		fRet = clSetKernelArg(kernelSrad_Kernel, 2, sizeof(int), &d_Nc);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_Nc)"));
		fRet = clSetKernelArg(kernelSrad_Kernel, 3, sizeof(long), &d_Ne);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_Ne)"));
		fRet = clEnqueueWriteBuffer(queueSrad_Kernel, d_iNK, CL_TRUE, 0, 502 * sizeof(int), d_iN, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (d_iNK)"));
		fRet = clEnqueueWriteBuffer(queueSrad_Kernel, d_iSK, CL_TRUE, 0, 502 * sizeof(int), d_iS, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (d_iSK)"));
		fRet = clEnqueueWriteBuffer(queueSrad_Kernel, d_jEK, CL_TRUE, 0, 458 * sizeof(int), d_jE, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (d_jEK)"));
		fRet = clEnqueueWriteBuffer(queueSrad_Kernel, d_jWK, CL_TRUE, 0, 458 * sizeof(int), d_jW, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (d_jWK)"));
		fRet = clEnqueueWriteBuffer(queueSrad_Kernel, d_dNK, CL_TRUE, 0, 229916 * sizeof(float), d_dN, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (d_dNK)"));
		fRet = clEnqueueWriteBuffer(queueSrad_Kernel, d_dSK, CL_TRUE, 0, 229916 * sizeof(float), d_dS, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (d_dSK)"));
		fRet = clEnqueueWriteBuffer(queueSrad_Kernel, d_dEK, CL_TRUE, 0, 229916 * sizeof(float), d_dE, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (d_dEK)"));
		fRet = clEnqueueWriteBuffer(queueSrad_Kernel, d_dWK, CL_TRUE, 0, 229916 * sizeof(float), d_dW, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (d_dWK)"));
		fRet = clSetKernelArg(kernelSrad_Kernel, 12, sizeof(float), &d_q0sqr);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (d_q0sqr)"));
		fRet = clEnqueueWriteBuffer(queueSrad_Kernel, d_cK, CL_TRUE, 0, 229916 * sizeof(float), d_c, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (d_cK)"));
		fRet = clEnqueueWriteBuffer(queueSrad_Kernel, d_IK, CL_TRUE, 0, 229916 * sizeof(float), d_I, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (d_IK)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueSrad_Kernel, kernelSrad_Kernel, workDimSrad_Kernel, NULL, globalSizeSrad_Kernel, localSizeSrad_Kernel, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueSrad_Kernel);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueSrad_Kernel, d_dNK, CL_TRUE, 0, 229916 * sizeof(float), d_dN, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueSrad_Kernel, d_dSK, CL_TRUE, 0, 229916 * sizeof(float), d_dS, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueSrad_Kernel, d_dEK, CL_TRUE, 0, 229916 * sizeof(float), d_dE, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueSrad_Kernel, d_dWK, CL_TRUE, 0, 229916 * sizeof(float), d_dW, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueSrad_Kernel, d_cK, CL_TRUE, 0, 229916 * sizeof(float), d_c, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueReadBuffer"));
		PRINT_SUCCESS();

		/* Calling loop postamble function */
		PRINT_STEP("[%d] Calling loop postamble function...", i);
		LOOPPOSTAMBLE(d_lambda, d_Nr, d_Nc, d_Ne, d_iN, 502, d_iS, 502, d_jE, 458, d_jW, 458, d_dN, 229916, d_dS, 229916, d_dE, 229916, d_dW, 229916, d_q0sqr, d_c, 229916, d_I, 229916, loopFlag);
		PRINT_SUCCESS();
		timersub(&tNow, &tThen, &tDelta);
		timeradd(&tExecTime, &tDelta, &tExecTime);
		i++;
	} while(loopFlag);

	/* Calling postamble function */
	PRINT_STEP("Calling postamble function...");
	POSTAMBLE(d_lambda, d_Nr, d_Nc, d_Ne, d_iN, 502, d_iS, 502, d_jE, 458, d_jW, 458, d_dN, 229916, d_dS, 229916, d_dE, 229916, d_dW, 229916, d_q0sqr, d_c, 229916, d_I, 229916);
	PRINT_SUCCESS();

	/* Print profiling results */
	long totalTime = (1000000 * tExecTime.tv_sec) + tExecTime.tv_usec;
	printf("Elapsed time spent on kernels: %ld us; Average time per iteration: %lf us.\n", totalTime, totalTime / (double) i);

	/* Validate received data */
	PRINT_STEP("Validating received data...");
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(d_iNK)
		clReleaseMemObject(d_iNK);
	if(d_iSK)
		clReleaseMemObject(d_iSK);
	if(d_jEK)
		clReleaseMemObject(d_jEK);
	if(d_jWK)
		clReleaseMemObject(d_jWK);
	if(d_dNK)
		clReleaseMemObject(d_dNK);
	if(d_dSK)
		clReleaseMemObject(d_dSK);
	if(d_dEK)
		clReleaseMemObject(d_dEK);
	if(d_dWK)
		clReleaseMemObject(d_dWK);
	if(d_cK)
		clReleaseMemObject(d_cK);
	if(d_IK)
		clReleaseMemObject(d_IK);

	/* Dealloc variables */
	free(d_iN);
	free(d_iS);
	free(d_jE);
	free(d_jW);
	free(d_dN);
	free(d_dS);
	free(d_dE);
	free(d_dW);
	free(d_c);
	free(d_I);

	/* Dealloc kernels */
	if(kernelSrad_Kernel)
		clReleaseKernel(kernelSrad_Kernel);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueSrad_Kernel)
		clReleaseCommandQueue(queueSrad_Kernel);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);


	return rv;
}
