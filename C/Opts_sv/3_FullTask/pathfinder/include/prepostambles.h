#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "constants.h"

#define MIN(a, b) ((a)<=(b) ? (a) : (b))

int *gData = NULL;
int gIndex = 0;

#define PREAMBLE(wall, wallSz, src, srcSz, dst, dstSz, cols, rem_rows, starting_row, comp_exit) {\
	int _i, _j;\
\
	srand(time(NULL));\
\
	gData = malloc(ROWS * COLS * sizeof(int));\
	for(_i = 0; _i < ROWS; _i++)\
		for(_j = 0; _j < COLS; _j++)\
			gData[_i * COLS + _j] = rand() % 10;\
\
	memcpy(wall, &gData[COLS], ((ROWS * COLS) - COLS) * sizeof(int));\
	memcpy(src, gData, COLS * sizeof(int));\
\
	/* We are using another logic to break loop */\
	loopFlag = true;\
}

#define LOOPPREAMBLE(wall, wallSz, src, srcSz, dst, dstSz, cols, rem_rows, starting_row, comp_exit, loopFlag) {\
	if(gIndex >= ROWS - 1) {\
		PRINT_SUCCESS();\
		break;\
	}\
\
	rem_rows = MIN(PYRAMID_HEIGHT, ROWS - gIndex - 1);\
	starting_row = gIndex;\
	int _compBSize = BSIZE - 2 * rem_rows;\
	int _lastCol = (0 == COLS % _compBSize)? COLS : COLS + _compBSize - COLS % _compBSize;\
	int _colBlocks = _lastCol / _compBSize;\
	comp_exit = BSIZE * _colBlocks * (rem_rows + 1) / SSIZE;\
}

#define LOOPPOSTAMBLE(wall, wallSz, src, srcSz, dst, dstSz, cols, rem_rows, starting_row, comp_exit, loopFlag) {\
	memcpy(src, dst, COLS * sizeof(int));\
	gIndex += PYRAMID_HEIGHT;\
}

#define POSTAMBLE(wall, wallSz, src, srcSz, dst, dstSz, cols, rem_rows, starting_row, comp_exit) {\
	int _i;\
	FILE *opf = fopen("outputResults", "w");\
\
	for(_i = 0; _i < COLS; _i++)\
		fprintf(opf, "%d\n", dst[_i]);\
\
	fclose(opf);\
\
	opf = fopen("outputBuffer", "w");\
\
	fclose(opf);\
}

#define CLEANUP(wall, wallSz, src, srcSz, dst, dstSz, cols, rem_rows, starting_row, comp_exit) {\
	if(gData)\
		free(gData);\
}
