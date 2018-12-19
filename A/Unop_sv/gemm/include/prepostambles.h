#include <stdlib.h>
#include <time.h>

#define PREAMBLE(A, ASz, lda, B, BSz, ldb, C, CSz, ldc, k, alpha, beta) {\
	int _i, _j;\
\
	srand(time(NULL));\
\
	for(_i = 0; _i < lda; _i++)\
		for(_j = 0; _j < lda; _j++)\
			A[_i * lda + _j] = (rand() / (float) RAND_MAX) * 1.5 + 0.5;\
\
	for(_i = 0; _i < ldb; _i++)\
		for(_j = 0; _j < ldb; _j++)\
			B[_i * ldb + _j] = (rand() / (float) RAND_MAX) * 1.5 + 0.5;\
\
	for(_i = 0; _i < ldc; _i++)\
		for(_j = 0; _j < ldc; _j++)\
			C[_i * ldc + _j] = 0;\
}
