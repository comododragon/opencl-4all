#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 64

#define PREAMBLE(A, ASz, B, BSz, result, resultSz, resultC, resultCSz, size) {\
	int _i, _j, _k;\
	srand(time(NULL));\
\
	for(_i = 0; _i < ASz; _i++) {\
		A[_i].s0 = (rand() % 8) / 3.0;\
		A[_i].s1 = (rand() % 8) / 3.0;\
		A[_i].s2 = (rand() % 8) / 3.0;\
		A[_i].s3 = (rand() % 8) / 3.0;\
		A[_i].s4 = (rand() % 8) / 3.0;\
		A[_i].s5 = (rand() % 8) / 3.0;\
		A[_i].s6 = (rand() % 8) / 3.0;\
		A[_i].s7 = (rand() % 8) / 3.0;\
		A[_i].s8 = (rand() % 8) / 3.0;\
		A[_i].s9 = (rand() % 8) / 3.0;\
		A[_i].sa = (rand() % 8) / 3.0;\
		A[_i].sb = (rand() % 8) / 3.0;\
		A[_i].sc = (rand() % 8) / 3.0;\
		A[_i].sd = (rand() % 8) / 3.0;\
		A[_i].se = (rand() % 8) / 3.0;\
		A[_i].sf = (rand() % 8) / 3.0;\
	}\
\
	for(_i = 0; _i < BSz; _i++) {\
		B[_i].s0 = (rand() % 8) / 3.0;\
		B[_i].s1 = (rand() % 8) / 3.0;\
		B[_i].s2 = (rand() % 8) / 3.0;\
		B[_i].s3 = (rand() % 8) / 3.0;\
		B[_i].s4 = (rand() % 8) / 3.0;\
		B[_i].s5 = (rand() % 8) / 3.0;\
		B[_i].s6 = (rand() % 8) / 3.0;\
		B[_i].s7 = (rand() % 8) / 3.0;\
		B[_i].s8 = (rand() % 8) / 3.0;\
		B[_i].s9 = (rand() % 8) / 3.0;\
		B[_i].sa = (rand() % 8) / 3.0;\
		B[_i].sb = (rand() % 8) / 3.0;\
		B[_i].sc = (rand() % 8) / 3.0;\
		B[_i].sd = (rand() % 8) / 3.0;\
		B[_i].se = (rand() % 8) / 3.0;\
		B[_i].sf = (rand() % 8) / 3.0;\
	}\
\
	for(_i = 0; _i < resultCSz; _i++) {\
    	float _intermed[BLOCK_SIZE];\
\
		for(_j = 0; _j < BLOCK_SIZE; _j++) {\
			_intermed[_j] = 0;\
			_intermed[_j] += A[(_i * BLOCK_SIZE) + _j].s0 * B[(_i * BLOCK_SIZE) + _j].s0;\
			_intermed[_j] += A[(_i * BLOCK_SIZE) + _j].s1 * B[(_i * BLOCK_SIZE) + _j].s1;\
			_intermed[_j] += A[(_i * BLOCK_SIZE) + _j].s2 * B[(_i * BLOCK_SIZE) + _j].s2;\
			_intermed[_j] += A[(_i * BLOCK_SIZE) + _j].s3 * B[(_i * BLOCK_SIZE) + _j].s3;\
			_intermed[_j] += A[(_i * BLOCK_SIZE) + _j].s4 * B[(_i * BLOCK_SIZE) + _j].s4;\
			_intermed[_j] += A[(_i * BLOCK_SIZE) + _j].s5 * B[(_i * BLOCK_SIZE) + _j].s5;\
			_intermed[_j] += A[(_i * BLOCK_SIZE) + _j].s6 * B[(_i * BLOCK_SIZE) + _j].s6;\
			_intermed[_j] += A[(_i * BLOCK_SIZE) + _j].s7 * B[(_i * BLOCK_SIZE) + _j].s7;\
			_intermed[_j] += A[(_i * BLOCK_SIZE) + _j].s8 * B[(_i * BLOCK_SIZE) + _j].s8;\
			_intermed[_j] += A[(_i * BLOCK_SIZE) + _j].s9 * B[(_i * BLOCK_SIZE) + _j].s9;\
			_intermed[_j] += A[(_i * BLOCK_SIZE) + _j].sa * B[(_i * BLOCK_SIZE) + _j].sa;\
			_intermed[_j] += A[(_i * BLOCK_SIZE) + _j].sb * B[(_i * BLOCK_SIZE) + _j].sb;\
			_intermed[_j] += A[(_i * BLOCK_SIZE) + _j].sc * B[(_i * BLOCK_SIZE) + _j].sc;\
			_intermed[_j] += A[(_i * BLOCK_SIZE) + _j].sd * B[(_i * BLOCK_SIZE) + _j].sd;\
			_intermed[_j] += A[(_i * BLOCK_SIZE) + _j].se * B[(_i * BLOCK_SIZE) + _j].se;\
			_intermed[_j] += A[(_i * BLOCK_SIZE) + _j].sf * B[(_i * BLOCK_SIZE) + _j].sf;\
		}\
\
		for(_j = BLOCK_SIZE / 2; _j > 0; _j /= 2) {\
			for(_k = 0; _k < BLOCK_SIZE; _k++) {\
				if(_k < _j)\
					_intermed[_k] += _intermed[_k + _j];\
			}\
		}\
\
        resultC[_i] = _intermed[0];\
	}\
}
