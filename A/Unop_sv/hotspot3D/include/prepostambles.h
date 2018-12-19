#define MAX_ITERS 3

float *gTemp = NULL;

#define PREAMBLE(p, pSz, tIn, tInSz, tOut, tOutSz, tOutC, tOutCSz, sdc, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc) {\
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
}

#define LOOPPREAMBLE(p, pSz, tIn, tInSz, tOut, tOutSz, tOutC, tOutCSz, sdc, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, loopFlag) {\
	if(i >= MAX_ITERS) {\
		PRINT_SUCCESS();\
		break;\
	}\
}

#define LOOPPOSTAMBLE(p, pSz, tIn, tInSz, tOut, tOutSz, tOutC, tOutCSz, sdc, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc, loopFlag) {\
	memcpy(gTemp, tIn, tInSz * sizeof(float));\
	memcpy(tIn, tOut, tInSz * sizeof(float));\
	memcpy(tOut, gTemp, tInSz * sizeof(float));\
}

#define CLEANUP(p, pSz, tIn, tInSz, tOut, tOutSz, tOutC, tOutCSz, sdc, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc) {\
	if(gTemp)\
		free(gTemp);\
}
