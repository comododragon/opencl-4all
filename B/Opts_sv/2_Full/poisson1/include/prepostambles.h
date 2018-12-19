#include <stdlib.h>
#include <time.h>

#define PREAMBLE(x, xSz, y, ySz, z, zSz, zC, zCSz) {\
	int _i, _j;\
	float _alpha = 3.2;\
	srand(time(NULL));\
\
	for(_i = 0; _i < xSz; _i++) {\
		x[_i].s0 = (rand() % 2048) / 16.0;\
		x[_i].s1 = (rand() % 2048) / 16.0;\
		x[_i].s2 = (rand() % 2048) / 16.0;\
		x[_i].s3 = (rand() % 2048) / 16.0;\
		x[_i].s4 = (rand() % 2048) / 16.0;\
		x[_i].s5 = (rand() % 2048) / 16.0;\
		x[_i].s6 = (rand() % 2048) / 16.0;\
		x[_i].s7 = (rand() % 2048) / 16.0;\
		x[_i].s8 = (rand() % 2048) / 16.0;\
		x[_i].s9 = (rand() % 2048) / 16.0;\
		x[_i].sa = (rand() % 2048) / 16.0;\
		x[_i].sb = (rand() % 2048) / 16.0;\
		x[_i].sc = (rand() % 2048) / 16.0;\
		x[_i].sd = (rand() % 2048) / 16.0;\
		x[_i].se = (rand() % 2048) / 16.0;\
		x[_i].sf = (rand() % 2048) / 16.0;\
	}\
\
	for(_i = 0; _i < ySz; _i++) {\
		y[_i].s0 = (rand() % 2048) / 16.0;\
		y[_i].s1 = (rand() % 2048) / 16.0;\
		y[_i].s2 = (rand() % 2048) / 16.0;\
		y[_i].s3 = (rand() % 2048) / 16.0;\
		y[_i].s4 = (rand() % 2048) / 16.0;\
		y[_i].s5 = (rand() % 2048) / 16.0;\
		y[_i].s6 = (rand() % 2048) / 16.0;\
		y[_i].s7 = (rand() % 2048) / 16.0;\
		y[_i].s8 = (rand() % 2048) / 16.0;\
		y[_i].s9 = (rand() % 2048) / 16.0;\
		y[_i].sa = (rand() % 2048) / 16.0;\
		y[_i].sb = (rand() % 2048) / 16.0;\
		y[_i].sc = (rand() % 2048) / 16.0;\
		y[_i].sd = (rand() % 2048) / 16.0;\
		y[_i].se = (rand() % 2048) / 16.0;\
		y[_i].sf = (rand() % 2048) / 16.0;\
	}\
\
	for(_i = 0; _i < zCSz; _i++) {\
		zC[_i].s0 = _alpha * x[_i].s0 + y[_i].s0;\
		zC[_i].s1 = _alpha * x[_i].s1 + y[_i].s1;\
		zC[_i].s2 = _alpha * x[_i].s2 + y[_i].s2;\
		zC[_i].s3 = _alpha * x[_i].s3 + y[_i].s3;\
		zC[_i].s4 = _alpha * x[_i].s4 + y[_i].s4;\
		zC[_i].s5 = _alpha * x[_i].s5 + y[_i].s5;\
		zC[_i].s6 = _alpha * x[_i].s6 + y[_i].s6;\
		zC[_i].s7 = _alpha * x[_i].s7 + y[_i].s7;\
		zC[_i].s8 = _alpha * x[_i].s8 + y[_i].s8;\
		zC[_i].s9 = _alpha * x[_i].s9 + y[_i].s9;\
		zC[_i].sa = _alpha * x[_i].sa + y[_i].sa;\
		zC[_i].sb = _alpha * x[_i].sb + y[_i].sb;\
		zC[_i].sc = _alpha * x[_i].sc + y[_i].sc;\
		zC[_i].sd = _alpha * x[_i].sd + y[_i].sd;\
		zC[_i].se = _alpha * x[_i].se + y[_i].se;\
		zC[_i].sf = _alpha * x[_i].sf + y[_i].sf;\
	}\
}
