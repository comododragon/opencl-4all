/* ********************************************************************************************* */
/* * Reed-Solomon Decoder Kernels (for Altera OpenCL)                                          * */
/* * Author: André Bannwart Perina                                                             * */
/* * Deeply based on code available at:                                                        * */
/* *     http://opencores.org/project,bluespec-reedsolomon                                     * */
/* *     Copyright (c) 2008 Abhinav Agarwal, Alfred Man Cheuk Ng                               * */
/* *     Contact: abhiag@gmail.com                                                             * */
/* ********************************************************************************************* */
/* * Copyright (c) 2016 André B. Perina                                                        * */
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

#include "constants.h"
#include "gfa.h"

__attribute__((reqd_work_group_size(256,1,1)))
__kernel void rsd(__global unsigned char *r, __global unsigned char *out, unsigned char loopCount) {
	int gid = get_global_id(0);

	/* Alpha lookup table, loaded as local memory for fast access */
	unsigned char alpha[256] = {ALPHALUT};
	unsigned char gfInvLUT[256] = {GFINVLUT};
	/* Auxiliary variables */
	int i, j, k;
	unsigned short s[2 * T];
	unsigned short c[T + 2];
	unsigned short w[T + 2];
	unsigned short p[T + 2];
	unsigned short a[T + 2];
	unsigned short shiftReg[T + 2];
	unsigned short temp[T + 2];
	unsigned short t1[T + 2];
	unsigned short t2[T + 2];
	unsigned short dStar;
	unsigned short d;
	unsigned short ddStar;
	unsigned short l;
	unsigned short acc;
	unsigned short errCnt;
	unsigned short alphaInv;
	unsigned short alphaInvTmp;
	unsigned short cTmp;
	unsigned short errLocOut[T];
	unsigned short alphaInvOut[T];
	int locIdx;
	unsigned short cVal;
	unsigned short wVal;
	unsigned short cDeriv[T];
	unsigned short errTmp[T];
	unsigned short errOut[K];
	unsigned short tmp;

	if(gid < loopCount) {
		/* Zero variables */
		for(i = 0; i < (2 * T); i++)
			s[i] = 0;

		/* Calculate syndrome */
		for(i = 0; i < N; i++) {
			for(j = 0; j < (2 * T); j++) {
				unsigned short res;
				GFA_MULT(k, res, s[j], alpha[j+1]);
				s[j] = res ^ r[i + (gid * N)];
			}
		}

		/* Initialise values */
		c[0] = 1;
		w[0] = 0;
		p[0] = 1;
		a[0] = 1;
		shiftReg[0] = 0;
		temp[0] = 0;
		dStar = 1;
		d = 0;
		ddStar = 1;
		l = 0;

		for(i = 1; i < (T + 2); i++) {
			c[i] = 0;
			w[i] = 0;
			p[i] = 0;
			a[i] = 0;
			t1[i] = 0;
			t2[i] = 0;
			shiftReg[i] = 0;
			temp[i] = 0;
		}

		for(i = 0; i < (2 * T); i++) {
			for(j = T + 1; j > 0; j--) {
				shiftReg[j] = shiftReg[j-1];
				p[j] = p[j-1];
				a[j] = a[j-1];
			}
			shiftReg[0] = s[i];
			p[0] = 0;
			a[0] = 0;

			/* GF Mult: array-array */
			for(j = 0; j < (T + 2); j++) {
				GFA_MULT(k, temp[j], c[j], shiftReg[j]);
			}

			/* GF Sum: array */
			d = 0;
			for(j = 0; j < (T + 2); j++) {
				GFA_ADD(d, d, temp[j]);
			}

			if(d) {
				GFA_MULT(j, ddStar, d, dStar);

				for(j = 0; j < (T + 2); j++) {
					t1[j] = p[j];
					t2[j] = a[j];
				}

				if((i + 1) > (2 * l)) {
					l = i-l+1;

					for(j = 0; j < (T + 2); j++) {
						p[j] = c[j];
						a[j] = w[j];
					}

					GFA_INV(gfInvLUT, dStar, d);
				}

				/* GF Mult: scalar-array */
				for(j = 0; j < (T + 2); j++) {
					GFA_MULT(k, temp[j], ddStar, t1[j]);
				}
				/* GF Add: array-array */
				for(j = 0; j < (T + 2); j++) {
					GFA_ADD(c[j], c[j], temp[j]);
				}
				/* GF Mult: scalar-array */
				for(j = 0; j < (T + 2); j++) {
					GFA_MULT(k, temp[j], ddStar, t2[j]);
				}
				/* GF Add: array-array */
				for(j = 0; j < (T + 2); j++) {
					GFA_ADD(w[j], w[j], temp[j]);
				}
			}
		}

		acc = 0;
		errCnt = 0;
		alphaInv = 1;

		for(i = (N - 1); i >= 0; i--) {
			for(j = 0; j < T; j++) {
				GFA_MULT(k, cTmp, c[j + 1], alpha[j+1]);
				c[j + 1] = cTmp;
			}

			acc = 1;

			for(j = 0; j < T; j++)
				GFA_ADD(acc, acc, c[j + 1]);

			GFA_MULT(j, alphaInvTmp, alphaInv, 2);
			alphaInv = alphaInvTmp;

			if((i >= (2 * T)) && (i < (K + (2 * T))) && !acc) {
				errLocOut[errCnt] = i - 2 * T;
				alphaInvOut[errCnt] = alphaInv;
				errCnt++;
			}
		}

		locIdx = 0;
		cVal = 0;
		wVal = 0;

		/* Compute deriv */
		for(i = 0; i < T; i++)
	        cDeriv[i] = (i % 2)? 0 : c[i + 1];

		for(i = 0; i < T; i++) {
			/* Poly eval */
			cVal = 0;
			for(j = (T - 1); j >= 0; j--) {
				GFA_MULT(k, tmp, cVal, alphaInvOut[i]);
				GFA_ADD(cVal, tmp, cDeriv[j]);
			}

			/* Poly eval */
			wVal = 0;
			for(j = (T - 1); j >= 0; j--) {
				GFA_MULT(k, tmp, wVal, alphaInvOut[i]);
				GFA_ADD(wVal, tmp, w[j + 1]);
			}

			/* GF Div */
			GFA_MULT(j, errTmp[i], wVal, gfInvLUT[cVal]);
		}

		for(i = 0; i < K; i++) {
			if(((K - 1 - i) == errLocOut[locIdx]) && (locIdx < errCnt)) {
				errOut[i] = errTmp[locIdx];
				locIdx++;
			}
			else {
				errOut[i] = 0;
			}
		}

		for(i = 0; i < K; i++) {
			/* Read input data, make corrections and send to output channel */
			GFA_ADD(out[i + (gid * K)], r[i + (gid * N)], errOut[i]);
		}
	}
}
