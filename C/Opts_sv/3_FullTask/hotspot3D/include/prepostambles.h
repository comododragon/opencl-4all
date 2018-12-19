#include "constants.h"

float *gTemp = NULL;


#define PREAMBLE(p, pSz, tIn, tInSz, tOut, tOutSz, tOutC, tOutCSz, sdc, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, last_col, comp_exit) {\
	int _i, _j;\
	unsigned int _vars = 3;\
	char *_fileNames[] = {\
		"inputP",\
		"inputTIn",\
		"outputTOut"\
	};\
	float *_fVars[] = {\
		p,\
		tIn,\
		tOutC\
	};\
	int *_iVars[] = {\
		NULL,\
		NULL,\
		NULL\
	};\
	char *_cVars[] = {\
		NULL,\
		NULL,\
		NULL\
	};\
	unsigned int _varsSizes[] = {\
		pSz,\
		tInSz,\
		tOutCSz\
	};\
\
	for(_i = 0; _i < _vars; _i++) {\
		FILE *ipf = fopen(_fileNames[_i], "r");\
		if(_fVars[_i]) {\
			for(_j = 0; _j < _varsSizes[_i]; _j++)\
				fscanf(ipf, "%f", &(_fVars[_i])[_j]);\
		}\
		else if(_iVars[_i]) {\
			for(_j = 0; _j < _varsSizes[_i]; _j++)\
				fscanf(ipf, "%d", &(_iVars[_i])[_j]);\
		}\
		else if(_cVars[_i]) {\
			for(_j = 0; _j < _varsSizes[_i]; _j++)\
				fscanf(ipf, "%c", &(_cVars[_i])[_j]);\
		}\
		fclose(ipf);\
	}\
\
	/* We are using another logic to break loop */\
	loopFlag = true;\
\
	gTemp = malloc(tOutSz * sizeof(float));\
\
	int _compBSizeX = BLOCK_X - 2;\
	int _compBSizeY = BLOCK_Y - 2;\
	last_col = ((0 == NUM_COLS % _compBSizeX)? NUM_COLS : NUM_COLS + _compBSizeX - NUM_COLS % _compBSizeX) - _compBSizeX;\
	int _lastRow = ((0 == NUM_ROWS % _compBSizeY)? NUM_ROWS : NUM_ROWS + _compBSizeY - NUM_ROWS % _compBSizeY) - _compBSizeY;\
	int _colBlocks = (last_col / _compBSizeX) + 1;\
	int _rowBlocks = (_lastRow / _compBSizeY) + 1;\
	comp_exit = (BLOCK_X * _colBlocks * BLOCK_Y * _rowBlocks * (NUM_LAYERS + 1)) / SSIZE;\
}

#define LOOPPREAMBLE(p, pSz, tIn, tInSz, tOut, tOutSz, tOutC, tOutCSz, sdc, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, last_col, comp_exit, loopFlag) {\
	if(i >= MAX_ITERS) {\
		PRINT_SUCCESS();\
		break;\
	}\
}

#define LOOPPOSTAMBLE(p, pSz, tIn, tInSz, tOut, tOutSz, tOutC, tOutCSz, sdc, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, last_col, comp_exit, loopFlag) {\
	memcpy(gTemp, tIn, tInSz * sizeof(float));\
	memcpy(tIn, tOut, tInSz * sizeof(float));\
	memcpy(tOut, gTemp, tInSz * sizeof(float));\
}

#define CLEANUP(p, pSz, tIn, tInSz, tOut, tOutSz, tOutC, tOutCSz, sdc, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, last_col, comp_exit) {\
	if(gTemp)\
		free(gTemp);\
}
