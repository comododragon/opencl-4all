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
