#define PREAMBLE(delta, deltaSz, hid, ly, lySz, lyC, lyCSz, in, w, wSz, wC, wCSz, oldw, oldwSz) {\
	int _i, _j;\
	unsigned int _vars = 4;\
	char *_fileNames[] = {\
		"inputLY",\
		"inputW",\
		"outputLY",\
		"outputW"\
	};\
	float *_fVars[] = {\
		ly,\
		w,\
		lyC,\
		wC,\
	};\
	int *_iVars[] = {\
		NULL,\
		NULL,\
		NULL,\
		NULL\
	};\
	char *_cVars[] = {\
		NULL,\
		NULL,\
		NULL,\
		NULL\
	};\
	unsigned int _varsSizes[] = {\
		lySz,\
		wSz,\
		lyCSz,\
		wCSz\
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
	for(_i = 0; _i < deltaSz; _i++)\
		delta[_i] = 0;\
}
