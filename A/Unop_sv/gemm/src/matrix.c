/**
 * Copyright (c) 2018 Andre Bannwart Perina
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "matrix.h"

#include <stdlib.h>
#include <string.h>

matrix_t *matrix_create(unsigned int m, unsigned int n) {
	matrix_t *matrix = malloc(sizeof(matrix_t));
	matrix->v = malloc(m * n * sizeof(float));
	matrix->m = m;
	matrix->n = n;

	return matrix;
}

matrix_t *matrix_create_source(unsigned int m, unsigned int n, float *source) {
	matrix_t *matrix = malloc(sizeof(matrix_t));
	matrix->v = malloc(m * n * sizeof(float));
	memcpy(matrix->v, source, m * n * sizeof(float));
	matrix->m = m;
	matrix->n = n;

	return matrix;
}

void matrix_pack(float *dest, matrix_t *M) {
	memcpy(dest, M->v, M->m * M->n * sizeof(float));
}

float matrix_get(matrix_t *M, unsigned int i, unsigned int j) {
	return M->v[(M->n) * i + j];
}

void matrix_set(matrix_t *M, unsigned int i, unsigned int j, float val) {
	M->v[(M->n) * i + j] = val;
}

void matrix_initialise(matrix_t *M, long seed, unsigned int haloWidth, float haloVal, int rowPeriod, int colPeriod) {
	int i, j;

	srand(seed);

	int nTileRows = M->m - 2 * haloWidth;
	if((rowPeriod != -1) && (rowPeriod < nTileRows))
		nTileRows = rowPeriod;

	int nTileCols = M->n - 2 * haloWidth;
	if((colPeriod != -1) && (colPeriod < nTileCols))
		nTileCols = colPeriod;

	for(i = 0; i < nTileRows; i++) {
		for(j = 0; j < nTileCols; j++) {
			M->v[(M->n) * (i + haloWidth) + (j + haloWidth)] = rand() / (float) RAND_MAX;

	if(colPeriod != -1) {
		int nTiles = (M->n - 2 * haloWidth) / colPeriod;
		if((M->n - 2 * haloWidth) % colPeriod)
			nTiles += 1;

		for(i = 1; i < nTiles; i++) {
			for(j = 0; j < nTileRows; j++) {
				memcpy(
					&(M->v[(M->n) * (haloWidth + j) + (haloWidth + i * nTileCols)]),
					&(M->v[(M->n) * (haloWidth + j) + haloWidth]),
					nTileCols * sizeof(float)
				);
			}
		}
	}

	if(rowPeriod != -1) {
		int nTiles = (M->m - 2 * haloWidth) / rowPeriod;
		if((M->m - 2 * haloWidth) % rowPeriod)
			nTiles += 1;

		for(i = 1; i < nTiles; i++) {
			for(j = 0; j < nTileRows; j++) {
				memcpy(
					&(M->v[(M->n) * (haloWidth + i * nTileRows + j) + haloWidth]),
					&(M->v[(M->n) * (haloWidth + j) + haloWidth]),
					(M->n - 2 * haloWidth) * sizeof(float)
				);
			}
		}
	}

	for(i = 0; i < M->m; i++) {
		for(j = 0; j < M->n; j++) {
			bool inHalo = false;

			if((i < haloWidth) || (i > M->m - 1 - haloWidth))
				inHalo = true;
			else if((j < haloWidth) || (j > M->n - 1 - haloWidth))
				inHalo = true;

			if(inHalo)
				M->v[(M->n) * i + j] = haloVal;
		}
	}
}

unsigned int matrix_get_m(matrix_t *M) {
	return M->m;
}

unsigned int matrix_get_n(matrix_t *M) {
	return M->n;
}

matrix_t *matrix_add(matrix_t *A, matrix_t *B) {
	if(A->m != B->m && A->n != B->n)
		return NULL;

	int i, j;
	matrix_t *C = matrix_create(A->m, A->n);

	for(i = 0; i < A->m; i++)
		for(j = 0; j < A->n; j++)
			C->v[(C->n) * i + j] = A->v[(A->n) * i + j] + B->v[(A->n) * i + j];
			
	return C;
}

matrix_t *matrix_mult_mm(matrix_t *A, matrix_t *B) {
	if(A->n != B->m)
		return NULL;

	int i, j, k;
	matrix_t *C = matrix_create(A->m, B->n);

	for(i = 0; i < A->m; i++) {
		for(j = 0; j < B->n; j++) {
			float val = 0;
			for(k = 0; k < A->n; k++)
				val += A->v[(A->n) * i + k] * B->v[(B->n) * k + j];

			C->v[(C->n) * i + j] = val;
			}
	}
			
	return C;
}

matrix_t *matrix_mult_ms(matrix_t *A, float val) {
	int i;
	matrix_t *C = matrix_create(A->m, A->n);

	for(i = 0; i < (C->m * C->n); i++)
		C->v[i] = A->v[i] * val;

	return C;
}

void matrix_destroy(matrix_t **M) {
	free((*M)->v);
	free(*M);
}
