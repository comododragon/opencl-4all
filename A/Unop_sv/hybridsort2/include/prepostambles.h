#define PREAMBLE(input, inputSz, indice, indiceSz, output, outputSz, outputC, outputCSz, size, d_prefixoffsets, d_prefixoffsetsSz, l_offsets, l_offsetsSz) {\
	int _i, _j;\
	unsigned int _vars = 5;\
	char *_fileNames[] = {\
		"inputInput",\
		"inputIndice",\
		"outputOutput",\
		"inputDPrefixoffsets",\
		"inputLOffsets"\
	};\
	float *_fVars[] = {\
		input,\
		NULL,\
		outputC,\
		NULL,\
		NULL\
	};\
	int *_iVars[] = {\
		NULL,\
		indice,\
		NULL,\
		d_prefixoffsets,\
		l_offsets\
	};\
	char *_cVars[] = {\
		NULL,\
		NULL,\
		NULL,\
		NULL,\
		NULL\
	};\
	unsigned int _varsSizes[] = {\
		inputSz,\
		indiceSz,\
		outputCSz,\
		d_prefixoffsetsSz,\
		l_offsetsSz\
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
}
