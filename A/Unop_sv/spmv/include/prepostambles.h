#include <stdlib.h>
#include <time.h>

#define PREAMBLE(val, valSz, vec, vecSz, cols, colsSz, rowDelimiters, rowDelimitersSz, dim, vecWidth, out, outSz, outC, outCSz) {\
	int _i, _j;\
	int _nnzAssigned = 0;\
	double _prob = valSz / ((double) (vecSz * vecSz));\
	bool _fill = false;\
\
	srand(time(NULL));\
\
	for(_i = 0; _i < valSz; _i++)\
		val[_i] = ((double) 10 * (rand() / (RAND_MAX + 1.0)));\
\
	for(_i = 0; _i < vecSz; _i++)\
		vec[_i] = ((double) 10 * (rand() / (RAND_MAX + 1.0)));\
\
	for(_i = 0; _i < vecSz; _i++) {\
		rowDelimiters[_i] = _nnzAssigned;\
		for(_j = 0; _j < vecSz; _j++) {\
			int _numEntriesLeft = (vecSz * vecSz) - ((_i * vecSz) + _j);\
			int _needToAssign = valSz - _nnzAssigned;\
\
			if(_numEntriesLeft <= _needToAssign)\
				_fill = true;\
\
			double _rand = rand() / (double) RAND_MAX;\
			if((_nnzAssigned < valSz && _rand <= _prob) || _fill) {\
				cols[_nnzAssigned] = _j;\
				_nnzAssigned++;\
			}\
		}\
	}\
\
	rowDelimiters[rowDelimitersSz - 1] = valSz;\
\
	for(_i = 0; _i < outCSz; _i++) {\
		double _t = 0;\
		for(_j = rowDelimiters[_i]; _j < rowDelimiters[_i + 1]; _j++)\
			_t += val[_j] * vec[cols[_j]];\
		outC[_i] = _t;\
	}\
}
