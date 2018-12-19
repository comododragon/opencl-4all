#include "constants.h"

#define MIN(a, b) (((a) <= (b))? (a) : (b))

int writeOutput(float *v, int gridRows, int gridCols, char *filename) {
	int i, j, index = 0;
	FILE *fp;
	char str[256];

	if(!(fp = fopen(filename, "w")))
		return -1;

	for(i = 0; i < gridRows; i++) {
		for(j = 0; j < gridCols; j++) {
			sprintf(str, "%d\t%g\n", index, v[i * gridCols + j]);
			fputs(str, fp);
			index++;
		}
	}

	fclose(fp);

	return 0;
}

int readInput(float *v, int gridRows, int gridCols, char *filename) {
	int i, j;
	FILE *fp;
	char str[256];
	float val;

	if(!(fp = fopen(filename, "r")))
		return -1;

	for(i = 0; i < gridRows; i++) {
		for(j = 0; j < gridCols; j++) {
			if(!fgets(str, 256, fp))
				return -2;

			if(sscanf(str, "%f", &val) != 1)
				return -3;

			v[i * gridCols + j] = val;
		}
	}

	fclose(fp);

	return 0;
}

int iters = 0;

#define PREAMBLE(iteration, power, powerSz, temp_src, temp_srcSz, temp_dst, temp_dstSz,\
		grid_cols, grid_rows, border_cols, border_rows, Cap, Rx, Ry, Rz, step) {\
	int _size = grid_rows * grid_cols;\
	char _tFile[] = "temp512";\
	char _pFile[] = "power512";\
\
	border_cols = PYRAMID_HEIGHT;\
	border_rows = PYRAMID_HEIGHT;\
	int _smallBlockCol = BLOCK_SIZE - PYRAMID_HEIGHT * EXPAND_RATE;\
	int _smallBlockRow = BLOCK_SIZE - PYRAMID_HEIGHT * EXPAND_RATE;\
	int _blockCols = grid_cols / _smallBlockCol + ((grid_cols % _smallBlockCol)? 1 : 0);\
	int _blockRows = grid_rows / _smallBlockRow + ((grid_rows % _smallBlockRow)? 1 : 0);\
\
	ASSERT_CALL(!readInput(temp_src, grid_rows, grid_cols, _tFile), rv = EXIT_FAILURE);\
	ASSERT_CALL(!readInput(power, grid_rows, grid_cols, _pFile), rv = EXIT_FAILURE);\
\
	float _gridHeight = CHIP_HEIGHT / grid_rows;\
	float _gridWidth = CHIP_WIDTH / grid_cols;\
\
	Cap = FACTOR_CHIP * SPEC_HEAT_SI * T_CHIP * _gridWidth * _gridHeight;\
	Rx = _gridWidth / (2.0 * K_SI * T_CHIP * _gridHeight);\
	Ry = _gridHeight / (2.0 * K_SI * T_CHIP * _gridWidth);\
	Rz = T_CHIP / (K_SI * _gridHeight * _gridWidth);\
\
	float _maxSlope = MAX_PD / (FACTOR_CHIP * T_CHIP * SPEC_HEAT_SI);\
	step = PRECISION / _maxSlope;\
}

#define LOOPPREAMBLE(iteration, power, powerSz, temp_src, temp_srcSz, temp_dst, temp_dstSz,\
		grid_cols, grid_rows, border_cols, border_rows, Cap, Rx, Ry, Rz, step, loopFlag) {\
	iteration = MIN(PYRAMID_HEIGHT, TOTAL_ITERATIONS - iters);\
}

#define LOOPPOSTAMBLE(iteration, power, powerSz, temp_src, temp_srcSz, temp_dst, temp_dstSz,\
		grid_cols, grid_rows, border_cols, border_rows, Cap, Rx, Ry, Rz, step, loopFlag) {\
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

#define POSTAMBLE(iteration, power, powerSz, temp_src, temp_srcSz, temp_dst, temp_dstSz,\
		grid_cols, grid_rows, border_cols, border_rows, Cap, Rx, Ry, Rz, step) {\
	char _oFile[] = "out512";\
\
	ASSERT_CALL(!writeOutput((i % 2)? temp_src : temp_dst, grid_rows, grid_cols, _oFile), rv = EXIT_FAILURE);\
}
