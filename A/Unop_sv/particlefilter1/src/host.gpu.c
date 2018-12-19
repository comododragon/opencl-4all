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
 *            PREAMBLE(arrayX, arrayXSz, arrayXC, arrayXCSz, arrayY, arrayYSz, arrayYC, arrayYCSz, xj, xjSz, yj, yjSz, ind, indSz, indC, indCSz, objxy, objxySz, likelihood, likelihoodSz, likelihoodC, likelihoodCSz, I, ISz, weights, weightsSz, weightsC, weightsCSz, Nparticles, countOnes, max_size, k, IszY, Nfr, seed, seedSz, partial_sums, partial_sumsSz, partial_sumsC, partial_sumsCSz);
 *            POSTAMBLE(arrayX, arrayXSz, arrayXC, arrayXCSz, arrayY, arrayYSz, arrayYC, arrayYCSz, xj, xjSz, yj, yjSz, ind, indSz, indC, indCSz, objxy, objxySz, likelihood, likelihoodSz, likelihoodC, likelihoodCSz, I, ISz, weights, weightsSz, weightsC, weightsCSz, Nparticles, countOnes, max_size, k, IszY, Nfr, seed, seedSz, partial_sums, partial_sumsSz, partial_sumsC, partial_sumsCSz);
 *            LOOPPREAMBLE(arrayX, arrayXSz, arrayXC, arrayXCSz, arrayY, arrayYSz, arrayYC, arrayYCSz, xj, xjSz, yj, yjSz, ind, indSz, indC, indCSz, objxy, objxySz, likelihood, likelihoodSz, likelihoodC, likelihoodCSz, I, ISz, weights, weightsSz, weightsC, weightsCSz, Nparticles, countOnes, max_size, k, IszY, Nfr, seed, seedSz, partial_sums, partial_sumsSz, partial_sumsC, partial_sumsCSz, loopFlag);
 *            LOOPPOSTAMBLE(arrayX, arrayXSz, arrayXC, arrayXCSz, arrayY, arrayYSz, arrayYC, arrayYCSz, xj, xjSz, yj, yjSz, ind, indSz, indC, indCSz, objxy, objxySz, likelihood, likelihoodSz, likelihoodC, likelihoodCSz, I, ISz, weights, weightsSz, weightsC, weightsCSz, Nparticles, countOnes, max_size, k, IszY, Nfr, seed, seedSz, partial_sums, partial_sumsSz, partial_sumsC, partial_sumsCSz, loopFlag);
 *            CLEANUP(arrayX, arrayXSz, arrayXC, arrayXCSz, arrayY, arrayYSz, arrayYC, arrayYCSz, xj, xjSz, yj, yjSz, ind, indSz, indC, indCSz, objxy, objxySz, likelihood, likelihoodSz, likelihoodC, likelihoodCSz, I, ISz, weights, weightsSz, weightsC, weightsCSz, Nparticles, countOnes, max_size, k, IszY, Nfr, seed, seedSz, partial_sums, partial_sumsSz, partial_sumsC, partial_sumsCSz);
 *        where:
 *            arrayX: variable (double *);
 *            arrayXSz: number of members in variable (unsigned int);
 *            arrayXC: variable (double *);
 *            arrayXCSz: number of members in variable (unsigned int);
 *            arrayY: variable (double *);
 *            arrayYSz: number of members in variable (unsigned int);
 *            arrayYC: variable (double *);
 *            arrayYCSz: number of members in variable (unsigned int);
 *            xj: variable (double *);
 *            xjSz: number of members in variable (unsigned int);
 *            yj: variable (double *);
 *            yjSz: number of members in variable (unsigned int);
 *            ind: variable (int *);
 *            indSz: number of members in variable (unsigned int);
 *            indC: variable (int *);
 *            indCSz: number of members in variable (unsigned int);
 *            objxy: variable (int *);
 *            objxySz: number of members in variable (unsigned int);
 *            likelihood: variable (double *);
 *            likelihoodSz: number of members in variable (unsigned int);
 *            likelihoodC: variable (double *);
 *            likelihoodCSz: number of members in variable (unsigned int);
 *            I: variable (char *);
 *            ISz: number of members in variable (unsigned int);
 *            weights: variable (double *);
 *            weightsSz: number of members in variable (unsigned int);
 *            weightsC: variable (double *);
 *            weightsCSz: number of members in variable (unsigned int);
 *            Nparticles: variable (int);
 *            countOnes: variable (int);
 *            max_size: variable (int);
 *            k: variable (int);
 *            IszY: variable (int);
 *            Nfr: variable (int);
 *            seed: variable (int *);
 *            seedSz: number of members in variable (unsigned int);
 *            partial_sums: variable (double *);
 *            partial_sumsSz: number of members in variable (unsigned int);
 *            partial_sumsC: variable (double *);
 *            partial_sumsCSz: number of members in variable (unsigned int);
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
	cl_command_queue queueLikelihood_Kernel = NULL;
	FILE *programFile = NULL;
	long programSz;
	char *programContent = NULL;
	cl_int programRet;
	cl_program program = NULL;
	cl_kernel kernelLikelihood_Kernel = NULL;
	bool loopFlag = false;
	bool invalidDataFound = false;
	struct timeval tThen, tNow, tDelta, tExecTime;
	timerclear(&tExecTime);
	cl_uint workDimLikelihood_Kernel = 3;
	size_t globalSizeLikelihood_Kernel[3] = {
		40448, 1, 1
	};
	size_t localSizeLikelihood_Kernel[3] = {
		512, 1, 1
	};

	/* Input/output variables */
	double *arrayX = malloc(40000 * sizeof(double));
	double *arrayXC = malloc(40000 * sizeof(double));
	double arrayXEpsilon = 0.001;
	cl_mem arrayXK = NULL;
	double *arrayY = malloc(40000 * sizeof(double));
	double *arrayYC = malloc(40000 * sizeof(double));
	double arrayYEpsilon = 0.001;
	cl_mem arrayYK = NULL;
	double *xj = malloc(40000 * sizeof(double));
	cl_mem xjK = NULL;
	double *yj = malloc(40000 * sizeof(double));
	cl_mem yjK = NULL;
	int *ind = malloc(2760000 * sizeof(int));
	int *indC = malloc(2760000 * sizeof(int));
	cl_mem indK = NULL;
	int *objxy = malloc(138 * sizeof(int));
	cl_mem objxyK = NULL;
	double *likelihood = malloc(40000 * sizeof(double));
	double *likelihoodC = malloc(40000 * sizeof(double));
	double likelihoodEpsilon = 0.001;
	cl_mem likelihoodK = NULL;
	char *I = malloc(163840 * sizeof(char));
	cl_mem IK = NULL;
	double *weights = malloc(40000 * sizeof(double));
	double *weightsC = malloc(40000 * sizeof(double));
	double weightsEpsilon = 0.1;
	cl_mem weightsK = NULL;
	int Nparticles = 40000;
	int countOnes = 69;
	int max_size = 163840;
	int k = 2;
	int IszY = 128;
	int Nfr = 10;
	int *seed = malloc(40000 * sizeof(int));
	cl_mem seedK = NULL;
	double *partial_sums = malloc(79 * sizeof(double));
	double *partial_sumsC = malloc(79 * sizeof(double));
	double partial_sumsEpsilon = 2;
	cl_mem partial_sumsK = NULL;

	/* Calling preamble function */
	PRINT_STEP("Calling preamble function...");
	PREAMBLE(arrayX, 40000, arrayXC, 40000, arrayY, 40000, arrayYC, 40000, xj, 40000, yj, 40000, ind, 2760000, indC, 2760000, objxy, 138, likelihood, 40000, likelihoodC, 40000, I, 163840, weights, 40000, weightsC, 40000, Nparticles, countOnes, max_size, k, IszY, Nfr, seed, 40000, partial_sums, 79, partial_sumsC, 79);
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

	/* Create command queue for likelihood_kernel kernel */
	PRINT_STEP("Creating command queue for \"likelihood_kernel\"...");
	queueLikelihood_Kernel = clCreateCommandQueue(context, devices[0], 0, &fRet);
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

	/* Create likelihood_kernel kernel */
	PRINT_STEP("Creating kernel \"likelihood_kernel\" from program...");
	kernelLikelihood_Kernel = clCreateKernel(program, "likelihood_kernel", &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateKernel"));
	PRINT_SUCCESS();

	/* Create input and output buffers */
	PRINT_STEP("Creating buffers...");
	arrayXK = clCreateBuffer(context, CL_MEM_READ_WRITE, 40000 * sizeof(double), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (arrayXK)"));
	arrayYK = clCreateBuffer(context, CL_MEM_READ_WRITE, 40000 * sizeof(double), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (arrayYK)"));
	xjK = clCreateBuffer(context, CL_MEM_READ_ONLY, 40000 * sizeof(double), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (xjK)"));
	yjK = clCreateBuffer(context, CL_MEM_READ_ONLY, 40000 * sizeof(double), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (yjK)"));
	indK = clCreateBuffer(context, CL_MEM_READ_WRITE, 2760000 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (indK)"));
	objxyK = clCreateBuffer(context, CL_MEM_READ_ONLY, 138 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (objxyK)"));
	likelihoodK = clCreateBuffer(context, CL_MEM_READ_WRITE, 40000 * sizeof(double), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (likelihoodK)"));
	IK = clCreateBuffer(context, CL_MEM_READ_ONLY, 163840 * sizeof(char), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (IK)"));
	weightsK = clCreateBuffer(context, CL_MEM_READ_WRITE, 40000 * sizeof(double), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (weightsK)"));
	seedK = clCreateBuffer(context, CL_MEM_READ_ONLY, 40000 * sizeof(int), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (seedK)"));
	partial_sumsK = clCreateBuffer(context, CL_MEM_READ_WRITE, 79 * sizeof(double), NULL, &fRet);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clCreateBuffer (partial_sumsK)"));
	PRINT_SUCCESS();

	/* Set kernel arguments for likelihood_kernel */
	PRINT_STEP("Setting kernel arguments for \"likelihood_kernel\"...");
	fRet = clSetKernelArg(kernelLikelihood_Kernel, 0, sizeof(cl_mem), &arrayXK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (arrayXK)"));
	fRet = clSetKernelArg(kernelLikelihood_Kernel, 1, sizeof(cl_mem), &arrayYK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (arrayYK)"));
	fRet = clSetKernelArg(kernelLikelihood_Kernel, 2, sizeof(cl_mem), &xjK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (xjK)"));
	fRet = clSetKernelArg(kernelLikelihood_Kernel, 3, sizeof(cl_mem), &yjK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (yjK)"));
	fRet = clSetKernelArg(kernelLikelihood_Kernel, 4, sizeof(cl_mem), &indK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (indK)"));
	fRet = clSetKernelArg(kernelLikelihood_Kernel, 5, sizeof(cl_mem), &objxyK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (objxyK)"));
	fRet = clSetKernelArg(kernelLikelihood_Kernel, 6, sizeof(cl_mem), &likelihoodK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (likelihoodK)"));
	fRet = clSetKernelArg(kernelLikelihood_Kernel, 7, sizeof(cl_mem), &IK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (IK)"));
	fRet = clSetKernelArg(kernelLikelihood_Kernel, 8, sizeof(cl_mem), &weightsK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (weightsK)"));
	fRet = clSetKernelArg(kernelLikelihood_Kernel, 9, sizeof(int), &Nparticles);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (Nparticles)"));
	fRet = clSetKernelArg(kernelLikelihood_Kernel, 10, sizeof(int), &countOnes);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (countOnes)"));
	fRet = clSetKernelArg(kernelLikelihood_Kernel, 11, sizeof(int), &max_size);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (max_size)"));
	fRet = clSetKernelArg(kernelLikelihood_Kernel, 12, sizeof(int), &k);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (k)"));
	fRet = clSetKernelArg(kernelLikelihood_Kernel, 13, sizeof(int), &IszY);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (IszY)"));
	fRet = clSetKernelArg(kernelLikelihood_Kernel, 14, sizeof(int), &Nfr);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (Nfr)"));
	fRet = clSetKernelArg(kernelLikelihood_Kernel, 15, sizeof(cl_mem), &seedK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (seedK)"));
	fRet = clSetKernelArg(kernelLikelihood_Kernel, 16, sizeof(cl_mem), &partial_sumsK);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (partial_sumsK)"));
	fRet = clSetKernelArg(kernelLikelihood_Kernel, 17, 512 * sizeof(double), NULL);
	ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (__local 17)"));
	PRINT_SUCCESS();

	do {
		/* Setting input and output buffers */
		PRINT_STEP("[%d] Setting buffers...", i);
		fRet = clEnqueueWriteBuffer(queueLikelihood_Kernel, arrayXK, CL_TRUE, 0, 40000 * sizeof(double), arrayX, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (arrayXK)"));
		fRet = clEnqueueWriteBuffer(queueLikelihood_Kernel, arrayYK, CL_TRUE, 0, 40000 * sizeof(double), arrayY, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (arrayYK)"));
		fRet = clEnqueueWriteBuffer(queueLikelihood_Kernel, xjK, CL_TRUE, 0, 40000 * sizeof(double), xj, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (xjK)"));
		fRet = clEnqueueWriteBuffer(queueLikelihood_Kernel, yjK, CL_TRUE, 0, 40000 * sizeof(double), yj, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (yjK)"));
		fRet = clEnqueueWriteBuffer(queueLikelihood_Kernel, indK, CL_TRUE, 0, 2760000 * sizeof(int), ind, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (indK)"));
		fRet = clEnqueueWriteBuffer(queueLikelihood_Kernel, objxyK, CL_TRUE, 0, 138 * sizeof(int), objxy, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (objxyK)"));
		fRet = clEnqueueWriteBuffer(queueLikelihood_Kernel, likelihoodK, CL_TRUE, 0, 40000 * sizeof(double), likelihood, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (likelihoodK)"));
		fRet = clEnqueueWriteBuffer(queueLikelihood_Kernel, IK, CL_TRUE, 0, 163840 * sizeof(char), I, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (IK)"));
		fRet = clEnqueueWriteBuffer(queueLikelihood_Kernel, weightsK, CL_TRUE, 0, 40000 * sizeof(double), weights, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (weightsK)"));
		fRet = clSetKernelArg(kernelLikelihood_Kernel, 9, sizeof(int), &Nparticles);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (Nparticles)"));
		fRet = clSetKernelArg(kernelLikelihood_Kernel, 10, sizeof(int), &countOnes);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (countOnes)"));
		fRet = clSetKernelArg(kernelLikelihood_Kernel, 11, sizeof(int), &max_size);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (max_size)"));
		fRet = clSetKernelArg(kernelLikelihood_Kernel, 12, sizeof(int), &k);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (k)"));
		fRet = clSetKernelArg(kernelLikelihood_Kernel, 13, sizeof(int), &IszY);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (IszY)"));
		fRet = clSetKernelArg(kernelLikelihood_Kernel, 14, sizeof(int), &Nfr);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clSetKernelArg (Nfr)"));
		fRet = clEnqueueWriteBuffer(queueLikelihood_Kernel, seedK, CL_TRUE, 0, 40000 * sizeof(int), seed, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (seedK)"));
		fRet = clEnqueueWriteBuffer(queueLikelihood_Kernel, partial_sumsK, CL_TRUE, 0, 79 * sizeof(double), partial_sums, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueWriteBuffer (partial_sumsK)"));
		PRINT_SUCCESS();

		PRINT_STEP("[%d] Running kernels...", i);
		gettimeofday(&tThen, NULL);
		fRet = clEnqueueNDRangeKernel(queueLikelihood_Kernel, kernelLikelihood_Kernel, workDimLikelihood_Kernel, NULL, globalSizeLikelihood_Kernel, localSizeLikelihood_Kernel, 0, NULL, NULL);
		ASSERT_CALL(CL_SUCCESS == fRet, FUNCTION_ERROR_STATEMENTS("clEnqueueNDRangeKernel"));
		clFinish(queueLikelihood_Kernel);
		gettimeofday(&tNow, NULL);
		PRINT_SUCCESS();

		/* Get output buffers */
		PRINT_STEP("[%d] Getting kernels arguments...", i);
		fRet = clEnqueueReadBuffer(queueLikelihood_Kernel, arrayXK, CL_TRUE, 0, 40000 * sizeof(double), arrayX, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueLikelihood_Kernel, arrayYK, CL_TRUE, 0, 40000 * sizeof(double), arrayY, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueLikelihood_Kernel, indK, CL_TRUE, 0, 2760000 * sizeof(int), ind, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueLikelihood_Kernel, likelihoodK, CL_TRUE, 0, 40000 * sizeof(double), likelihood, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueLikelihood_Kernel, weightsK, CL_TRUE, 0, 40000 * sizeof(double), weights, 0, NULL, NULL);
		fRet = clEnqueueReadBuffer(queueLikelihood_Kernel, partial_sumsK, CL_TRUE, 0, 79 * sizeof(double), partial_sums, 0, NULL, NULL);
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
		if(TEST_EPSILON(arrayXC[i],  arrayX[i], arrayXEpsilon)) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable arrayX[%d]: expected %lf got %lf (with epsilon).\n", i, arrayXC[i], arrayX[i]);
		}
	}
	for(i = 0; i < 40000; i++) {
		if(TEST_EPSILON(arrayYC[i],  arrayY[i], arrayYEpsilon)) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable arrayY[%d]: expected %lf got %lf (with epsilon).\n", i, arrayYC[i], arrayY[i]);
		}
	}
	for(i = 0; i < 2760000; i++) {
		if(indC[i] != ind[i]) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable ind[%d]: expected %d got %d.\n", i, indC[i], ind[i]);
		}
	}
	for(i = 0; i < 40000; i++) {
		if(TEST_EPSILON(likelihoodC[i],  likelihood[i], likelihoodEpsilon)) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable likelihood[%d]: expected %lf got %lf (with epsilon).\n", i, likelihoodC[i], likelihood[i]);
		}
	}
	for(i = 0; i < 40000; i++) {
		if(TEST_EPSILON(weightsC[i],  weights[i], weightsEpsilon)) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable weights[%d]: expected %lf got %lf (with epsilon).\n", i, weightsC[i], weights[i]);
		}
	}
	for(i = 0; i < 79; i++) {
		if(TEST_EPSILON(partial_sumsC[i],  partial_sums[i], partial_sumsEpsilon)) {
			if(!invalidDataFound) {
				PRINT_FAIL();
				invalidDataFound = true;
			}
			printf("Variable partial_sums[%d]: expected %lf got %lf (with epsilon).\n", i, partial_sumsC[i], partial_sums[i]);
		}
	}
	if(!invalidDataFound)
		PRINT_SUCCESS();

_err:

	/* Dealloc buffers */
	if(arrayXK)
		clReleaseMemObject(arrayXK);
	if(arrayYK)
		clReleaseMemObject(arrayYK);
	if(xjK)
		clReleaseMemObject(xjK);
	if(yjK)
		clReleaseMemObject(yjK);
	if(indK)
		clReleaseMemObject(indK);
	if(objxyK)
		clReleaseMemObject(objxyK);
	if(likelihoodK)
		clReleaseMemObject(likelihoodK);
	if(IK)
		clReleaseMemObject(IK);
	if(weightsK)
		clReleaseMemObject(weightsK);
	if(seedK)
		clReleaseMemObject(seedK);
	if(partial_sumsK)
		clReleaseMemObject(partial_sumsK);

	/* Dealloc variables */
	free(arrayX);
	free(arrayXC);
	free(arrayY);
	free(arrayYC);
	free(xj);
	free(yj);
	free(ind);
	free(indC);
	free(objxy);
	free(likelihood);
	free(likelihoodC);
	free(I);
	free(weights);
	free(weightsC);
	free(seed);
	free(partial_sums);
	free(partial_sumsC);

	/* Dealloc kernels */
	if(kernelLikelihood_Kernel)
		clReleaseKernel(kernelLikelihood_Kernel);

	/* Dealloc program */
	if(program)
		clReleaseProgram(program);
	if(programContent)
		free(programContent);
	if(programFile)
		fclose(programFile);

	/* Dealloc queues */
	if(queueLikelihood_Kernel)
		clReleaseCommandQueue(queueLikelihood_Kernel);

	/* Last OpenCL variables */
	if(context)
		clReleaseContext(context);
	if(devices)
		free(devices);
	if(platforms)
		free(platforms);


	return rv;
}
