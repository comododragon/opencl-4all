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

#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
	float *v;
	unsigned int n;
	unsigned int m;
} matrix_t;

matrix_t *matrix_create(unsigned int m, unsigned int n);
matrix_t *matrix_create_source(unsigned int m, unsigned int n, float *source);
void matrix_pack(float *dest, matrix_t *M);
float matrix_get(matrix_t *M, unsigned int i, unsigned int j);
void matrix_set(matrix_t *M, unsigned int i, unsigned int j, float val);
void matrix_initialise(matrix_t *M, long seed, unsigned int haloWidth, float haloVal, int rowPeriod, int colPeriod);
unsigned int matrix_get_m(matrix_t *M);
unsigned int matrix_get_n(matrix_t *M);
matrix_t *matrix_add(matrix_t *A, matrix_t *B);
matrix_t *matrix_mult_mm(matrix_t *A, matrix_t *B);
matrix_t *matrix_mult_ms(matrix_t *A, float val);
void matrix_destroy(matrix_t **M);

#endif
