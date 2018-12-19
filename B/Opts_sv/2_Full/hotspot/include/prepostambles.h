#include "constants.h"

#define MIN(a, b) (((a) <= (b))? (a) : (b))

int iters = 0;

#define PREAMBLE(iteration, power, powerSz, temp_src, temp_srcSz, temp_dst, temp_dstSz, grid_cols, grid_rows, pyramid_height, step_div_cap, Rx, Ry, Rz, small_block_rows, small_block_cols) {\
	int _i, _j;\
	unsigned int _vars = 2;\
	char *_fileNames[] = {\
		"inputPower",\
		"inputTempSrc"\
	};\
	void *_varsPointers[] = {\
		power,\
		temp_src\
	};\
	unsigned int _varsSizes[] = {\
		powerSz,\
		temp_srcSz\
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
	pyramid_height = PYRAMID_HEIGHT;\
	float _gridHeight = CHIP_HEIGHT / grid_rows;\
	float _gridWidth = CHIP_WIDTH / grid_cols;\
\
	Rx = _gridWidth / (2.0 * K_SI * T_CHIP * _gridHeight);\
	Ry = _gridHeight / (2.0 * K_SI * T_CHIP * _gridWidth);\
	Rz = T_CHIP / (K_SI * _gridHeight * _gridWidth);\
\
	float _cap = FACTOR_CHIP * SPEC_HEAT_SI * T_CHIP * _gridWidth * _gridHeight;\
	float _maxSlope = MAX_PD / (FACTOR_CHIP * T_CHIP * SPEC_HEAT_SI);\
	float _step = PRECISION / _maxSlope;\
	step_div_cap = _step / _cap;\
}

#define LOOPPREAMBLE(iteration, power, powerSz, temp_src, temp_srcSz, temp_dst, temp_dstSz, grid_cols, grid_rows, pyramid_height, step_div_cap, Rx, Ry, Rz, small_block_rows, small_block_cols, loopFlag) {\
	iteration = MIN(PYRAMID_HEIGHT, TOTAL_ITERATIONS - iters);\
	small_block_rows = BLOCK_Y - iteration * 2;\
	small_block_cols = BLOCK_X - iteration * 2;\
}

#define LOOPPOSTAMBLE(iteration, power, powerSz, temp_src, temp_srcSz, temp_dst, temp_dstSz, grid_cols, grid_rows, pyramid_height, step_div_cap, Rx, Ry, Rz, small_block_rows, small_block_cols, loopFlag) {\
	int _i;\
	float tmp[262144];\
\
	memcpy(tmp, temp_src, 262144 * sizeof(float));\
	memcpy(temp_src, temp_dst, 262144 * sizeof(float));\
	memcpy(temp_dst, tmp, 262144 * sizeof(float));\
\
	iters += PYRAMID_HEIGHT;\
\
	loopFlag = (iters < TOTAL_ITERATIONS);\
}
