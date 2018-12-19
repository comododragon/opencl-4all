#include "constants.h"
#include "gfa.h"

#define MAX_ERR 16

/**
 * @brief Chien stage kernel.
 *
 * @param loopCount Number of times to process an N bytes block of data.
 */
__attribute__((reqd_work_group_size(256,1,1)))
__kernel void chien(__global unsigned short *lA, __global unsigned short *errLocOut, __global unsigned short *alphaInvOut, __global unsigned short *errCnt, unsigned char loopCount) {
	int gid = get_global_id(0);

	/* Alpha lookup table, loaded as local memory for fast access */
	unsigned char alpha[256] = {ALPHALUT};
	/* Auxiliary variables */
	int i, j, k;
	unsigned short acc;
	unsigned short alphaInv;
	unsigned short alphaInvTmp;
	unsigned short lATmp;

	if(gid < loopCount) {
		acc = 0;
		errCnt[gid] = 0;
		alphaInv = 1;

		for(i = (N - 1); i >= 0; i--) {
			for(j = 0; j < T; j++) {
				GFA_MULT(k, lATmp, lA[j + (gid * T)], alpha[j+1]);
				lA[j + (gid * T)] = lATmp;
			}

			acc = 1;

			for(j = 0; j < T; j++)
				GFA_ADD(acc, acc, lA[j + (gid * T)]);

			GFA_MULT(j, alphaInvTmp, alphaInv, 2);
			alphaInv = alphaInvTmp;

			if((i >= (2 * T)) && (i < (K + (2 * T))) && !acc) {
				errLocOut[errCnt[gid] + (gid * MAX_ERR)] = i - 2 * T;
				alphaInvOut[errCnt[gid] + (gid * MAX_ERR)] = alphaInv;
				errCnt[gid]++;
			}
		}
	}
}
