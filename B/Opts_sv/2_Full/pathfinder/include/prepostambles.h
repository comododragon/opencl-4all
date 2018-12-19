#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MIN(a, b) ((a)<=(b) ? (a) : (b))

#define ROWS 100
#define COLS 10000
#define PYRAMID_HEIGHT 92
#define HALO_VAL 1

int *gData = NULL;
int gIndex = 0;

#define PREAMBLE(iteration, gpuWall, gpuWallSz, gpuSrc, gpuSrcSz, gpuResults, gpuResultsSz, cols, startStep, border) {\
	int _i, _j;\
\
	srand(time(NULL));\
\
	gData = malloc(ROWS * COLS * sizeof(int));\
	for(_i = 0; _i < ROWS; _i++)\
		for(_j = 0; _j < COLS; _j++)\
			gData[_i * COLS + _j] = rand() % 10;\
\
	memcpy(gpuWall, &gData[COLS], ((ROWS * COLS) - COLS) * sizeof(int));\
	memcpy(gpuSrc, gData, COLS * sizeof(int));\
\
	border = PYRAMID_HEIGHT * HALO_VAL;\
	/* We are using another logic to break loop */\
	loopFlag = true;\
}

#define LOOPPREAMBLE(iteration, gpuWall, gpuWallSz, gpuSrc, gpuSrcSz, gpuResults, gpuResultsSz, cols, startStep, border, loopFlag) {\
	if(gIndex >= ROWS - 1) {\
		PRINT_SUCCESS();\
		break;\
	}\
\
	iteration = MIN(PYRAMID_HEIGHT, ROWS - gIndex - 1);\
	startStep = gIndex;\
}

#define LOOPPOSTAMBLE(iteration, gpuWall, gpuWallSz, gpuSrc, gpuSrcSz, gpuResults, gpuResultsSz, cols, startStep, border, loopFlag) {\
	memcpy(gpuSrc, gpuResults, COLS * sizeof(int));\
	gIndex += PYRAMID_HEIGHT;\
}

#define POSTAMBLE(iteration, gpuWall, gpuWallSz, gpuSrc, gpuSrcSz, gpuResults, gpuResultsSz, cols, startStep, border) {\
	int _i;\
	FILE *opf = fopen("outputResults", "w");\
\
	for(_i = 0; _i < COLS; _i++)\
		fprintf(opf, "%d\n", gpuResults[_i]);\
\
	fclose(opf);\
\
	opf = fopen("outputBuffer", "w");\
\
	fclose(opf);\
}

#define CLEANUP(iteration, gpuWall, gpuWallSz, gpuSrc, gpuSrcSz, gpuResults, gpuResultsSz, cols, startStep, border) {\
	if(gData)\
		free(gData);\
}
