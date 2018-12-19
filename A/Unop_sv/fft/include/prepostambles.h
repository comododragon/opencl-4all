#include "stdlib.h"
#include "time.h"

#define PREAMBLE(work, workSz) {\
	int _i;\
	int _hWorkSz = workSz >> 1;\
\
	srand(time(NULL));\
\
	for(_i = 0; _i < _hWorkSz; _i++) {\
		work[_i].x = (rand() / (float) RAND_MAX) * 2 - 1;\
		work[_i].y = (rand() / (float) RAND_MAX) * 2 - 1;\
		work[_i + _hWorkSz].x = work[_i].x;\
		work[_i + _hWorkSz].y = work[_i].y;\
	}\
}
