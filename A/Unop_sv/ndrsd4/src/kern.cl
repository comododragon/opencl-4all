#include "constants.h"
#include "gfa.h"

/**
 * @brief Syndrome stage kernel.
 *
 * @param loopCount Number of times to process an N bytes block of data.
 */
__attribute__((reqd_work_group_size(256,1,1)))
__kernel void syndrome(__global unsigned char *r, __global unsigned short *s, unsigned char loopCount) {
	int gid = get_global_id(0);

	/* Alpha lookup table, loaded as local memory for fast access */
	unsigned char alpha[256] = {ALPHALUT};
	/* Auxiliary variables */
	unsigned int i, j, k;

	if(gid < loopCount) {
		/* Zero variables */
		for(i = 0; i < (2 * T); i++)
			s[i + (gid * (2 * T))] = 0;

		/* Calculate syndrome */
		for(i = 0; i < N; i++) {
			for(j = 0; j < (2 * T); j++) {
				unsigned short res;
				GFA_MULT(k, res, s[j + (gid * (2 * T))], alpha[j+1]);
				s[j + (gid * (2 * T))] = res ^ r[i + (gid * N)];
			}
		}
	}
}
