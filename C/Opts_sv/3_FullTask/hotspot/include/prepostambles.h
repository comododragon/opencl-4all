#include "constants.h"

#define MIN(a, b) (((a) <= (b))? (a) : (b))

int iters = 0;

#define PREAMBLE(power, powerSz, src, srcSz, dst, dstSz, grid_cols, grid_rows, sdc, Rx_1, Ry_1, Rz_1, comp_exit) {\
	int _i, _j;\
	unsigned int _vars = 2;\
	char *_fileNames[] = {\
		"inputPower",\
		"inputTempSrc"\
	};\
	void *_varsPointers[] = {\
		power,\
		src\
	};\
	unsigned int _varsSizes[] = {\
		powerSz,\
		srcSz\
	};\
	unsigned int _varsTypeSizes[] = {\
		sizeof(float),\
		sizeof(float)\
	};\
\
	for(_i = 0; _i < _vars; _i++) {\
		FILE *ipf = fopen(_fileNames[_i], "rb");\
		fread(_varsPointers[_i], _varsTypeSizes[_i], _varsSizes[_i], ipf);\
		fclose(ipf);\
	}\
\
	float _gridHeight = CHIP_HEIGHT / grid_rows;\
	float _gridWidth = CHIP_WIDTH / grid_cols;\
\
	float _Rx = _gridWidth / (2.0 * K_SI * T_CHIP * _gridHeight);\
	float _Ry = _gridHeight / (2.0 * K_SI * T_CHIP * _gridWidth);\
	float _Rz = T_CHIP / (K_SI * _gridHeight * _gridWidth);\
	Rx_1 = 1 / _Rx;\
	Ry_1 = 1 / _Ry;\
	Rz_1 = 1 / _Rz;\
\
	float _cap = FACTOR_CHIP * SPEC_HEAT_SI * T_CHIP * _gridWidth * _gridHeight;\
	float _maxSlope = MAX_PD / (FACTOR_CHIP * T_CHIP * SPEC_HEAT_SI);\
	float _step = PRECISION / _maxSlope;\
	sdc = _step / _cap;\
\
	int _compBSize = BLOCK_X - 2;\
	int _lastCol = (grid_cols % _compBSize == 0) ? grid_cols + 0 : grid_cols + _compBSize - grid_cols % _compBSize;\
	int _colBlocks = _lastCol / _compBSize;\
	comp_exit = BLOCK_X * _colBlocks * (grid_rows + 1) / SSIZE;\
}

#define LOOPPOSTAMBLE(power, powerSz, src, srcSz, dst, dstSz, grid_cols, grid_rows, sdc, Rx_1, Ry_1, Rz_1, comp_exit, loopFlag) {\
	int _i;\
	float tmp[262144];\
\
	memcpy(tmp, src, 262144 * sizeof(float));\
	memcpy(src, dst, 262144 * sizeof(float));\
	memcpy(dst, tmp, 262144 * sizeof(float));\
\
	iters++;\
\
	loopFlag = (iters < TOTAL_ITERATIONS);\
}
