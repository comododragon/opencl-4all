#define MAX_COLS 2048
#define MAX_ROWS 2048
#define MAX_COLS_P (MAX_COLS + 1)
#define MAX_ROWS_P (MAX_ROWS + 1)
#define BLOCK_SIZE 4096
#define BLOCK_SIZE_M (BLOCK_SIZE - 1)
#define PAR 64

int _bx;
int _numBlocks;

#define PREAMBLE(reference, referenceSz, data, dataSz, dataC, dataCSz, input_v, input_vSz, dim, penalty, loop_exit, block_offset) {\
	int _i, _j;\
	unsigned int _vars = 4;\
	char *_fileNames[] = {\
		"inputReference",\
		"inputData",\
		"outputData",\
		"inputInputV"\
	};\
	void *_varsPointers[] = {\
		reference,\
		data,\
		dataC,\
		input_v\
	};\
	unsigned int _varsSizes[] = {\
		referenceSz,\
		dataSz,\
		dataCSz,\
		input_vSz\
	};\
	unsigned int _varsTypeSizes[] = {\
		sizeof(int),\
		sizeof(int),\
		sizeof(int),\
		sizeof(int)\
	};\
\
	for(_i = 0; _i < _vars; _i++) {\
		FILE *ipf = fopen(_fileNames[_i], "rb");\
		fread(_varsPointers[_i], _varsTypeSizes[_i], _varsSizes[_i], ipf);\
		fclose(ipf);\
	}\
\
	int _cols = MAX_COLS - 1 + PAR;\
	int _exitCol = (0 == _cols % PAR)? _cols : _cols + PAR - (_cols % PAR);\
	loop_exit = _exitCol * (BLOCK_SIZE / PAR);\
\
	int _lastDiag = (0 == MAX_ROWS % BLOCK_SIZE_M)? MAX_ROWS : MAX_ROWS + BLOCK_SIZE_M - (MAX_ROWS % BLOCK_SIZE_M);\
	_numBlocks = _lastDiag / BLOCK_SIZE_M;\
	_bx = 0;\
\
	/* We are using another logic to break loop */\
	loopFlag = true;\
}

#define LOOPPREAMBLE(reference, referenceSz, data, dataSz, dataC, dataCSz, input_v, input_vSz, dim, penalty, loop_exit, block_offset, loopFlag) {\
	if(_bx >= _numBlocks) {\
		PRINT_SUCCESS();\
		break;\
	}\
\
	block_offset = _bx / BLOCK_SIZE_M;\
}

#define LOOPPOSTAMBLE(reference, referenceSz, data, dataSz, dataC, dataCSz, input_v, input_vSz, dim, penalty, loop_exit, block_offset, loopFlag) {\
	_bx++;\
}
