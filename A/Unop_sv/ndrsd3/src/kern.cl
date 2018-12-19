#include "constants.h"
#include "gfa.h"

#define MAX_ERR 16

/**
 * @brief Forneys stage kernel.
 *
 * @param loopCount Number of times to process an N bytes block of data.
 */
__attribute__((reqd_work_group_size(256,1,1)))
__kernel void forneys(__global unsigned short *lambda, __global unsigned short *omega, __global unsigned short *errCnt, __global unsigned short *errLoc, __global unsigned short *alphaInv, __global unsigned short *errOut, unsigned char loopCount) {
	int gid = get_global_id(0);

	/* Gallois-field inversion lookup table, loaded as local memory for fast access */
	unsigned char gfInvLUT[256] = {GFINVLUT};
	/* Auxiliary variables */
	int i, j, k;
	int locIdx;
	unsigned short lambdaVal;
	unsigned short omegaVal;
	unsigned short lambdaDeriv[T];
	unsigned short errTmp[T];
	unsigned short tmp;

	if(gid < loopCount) {
		locIdx = 0;
		lambdaVal = 0;
		omegaVal = 0;

		/* Compute deriv */
		for(i = 0; i < T; i++)
	        lambdaDeriv[i] = (i % 2)? 0 : lambda[i + (gid * T)];

		for(i = 0; i < T; i++) {
			/* Poly eval */
			lambdaVal = 0;
			for(j = (T - 1); j >= 0; j--) {
				GFA_MULT(k, tmp, lambdaVal, alphaInv[i + (gid * MAX_ERR)]);
				GFA_ADD(lambdaVal, tmp, lambdaDeriv[j]);
			}

			/* Poly eval */
			omegaVal = 0;
			for(j = (T - 1); j >= 0; j--) {
				GFA_MULT(k, tmp, omegaVal, alphaInv[i + (gid * MAX_ERR)]);
				GFA_ADD(omegaVal, tmp, omega[j + (gid * T)]);
			}

			/* GF Div */
			GFA_MULT(j, errTmp[i], omegaVal, gfInvLUT[lambdaVal]);
		}

		for(i = 0; i < K; i++) {
			if(((K - 1 - i) == errLoc[locIdx + (gid * MAX_ERR)]) && (locIdx < errCnt[gid])) {
				errOut[i + (gid * K)] = errTmp[locIdx];
				locIdx++;
			}
			else {
				errOut[i + (gid * K)] = 0;
			}
		}
	}
}
