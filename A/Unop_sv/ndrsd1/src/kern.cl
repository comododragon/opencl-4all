#include "constants.h"
#include "gfa.h"

/**
 * @brief Berlekamp stage kernel.
 *
 * @param loopCount Number of times to process an N bytes block of data.
 */
__attribute__((reqd_work_group_size(256,1,1)))
__kernel void berlekamp(__global unsigned short *s, __global unsigned short *c, __global unsigned short *w, unsigned char loopCount) {
	int gid = get_global_id(0);

	/* Gallois-field inversion lookup table, loaded as local memory for fast access */
	unsigned char gfInvLUT[256] = {GFINVLUT};
	/* Auxiliary variables */
	unsigned int i, j, k;
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

	if(gid < loopCount) {
		/* Initialise values */
		c[(gid * (T + 2))] = 1;
		w[(gid * (T + 2))] = 0;
		p[0] = 1;
		a[0] = 1;
		shiftReg[0] = 0;
		temp[0] = 0;
		dStar = 1;
		d = 0;
		ddStar = 1;
		l = 0;

		for(i = 1; i < (T + 2); i++) {
			c[i + (gid * (T + 2))] = 0;
			w[i + (gid * (T + 2))] = 0;
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
			shiftReg[0] = s[i + (2 * T)];
			p[0] = 0;
			a[0] = 0;

			/* GF Mult: array-array */
			for(j = 0; j < (T + 2); j++) {
				GFA_MULT(k, temp[j], c[j + (gid * (T + 2))], shiftReg[j]);
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
						p[j] = c[j + (gid * (T + 2))];
						a[j] = w[j + (gid * (T + 2))];
					}

					GFA_INV(gfInvLUT, dStar, d);
				}

				/* GF Mult: scalar-array */
				for(j = 0; j < (T + 2); j++) {
					GFA_MULT(k, temp[j], ddStar, t1[j]);
				}
				/* GF Add: array-array */
				for(j = 0; j < (T + 2); j++) {
					GFA_ADD(c[j + (gid * (T + 2))], c[j + (gid * (T + 2))], temp[j]);
				}
				/* GF Mult: scalar-array */
				for(j = 0; j < (T + 2); j++) {
					GFA_MULT(k, temp[j], ddStar, t2[j]);
				}
				/* GF Add: array-array */
				for(j = 0; j < (T + 2); j++) {
					GFA_ADD(w[j + (gid * (T + 2))], w[j + (gid * (T + 2))], temp[j]);
				}
			}
		}
	}
}

