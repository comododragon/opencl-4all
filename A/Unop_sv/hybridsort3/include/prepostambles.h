#define PREAMBLE(input, inputSz, result, resultSz, resultC, resultCSz, nrElems, threadsPerDiv, constStartAddr, constStartAddrSz) {\
	int _i, _j;\
	unsigned int _vars = 3;\
	char *_fileNames[] = {\
		"inputInput",\
		"inputConstStartAddr",\
		"outputResult"\
	};\
	float *_fVars[] = {\
		NULL,\
		NULL,\
		NULL\
	};\
	cl_float4 *_f4Vars[] = {\
		input,\
		NULL,\
		resultC\
	};\
	int *_iVars[] = {\
		NULL,\
		constStartAddr,\
		NULL\
	};\
	char *_cVars[] = {\
		NULL,\
		NULL,\
		NULL\
	};\
	unsigned int _varsSizes[] = {\
		inputSz,\
		constStartAddrSz,\
		resultCSz\
	};\
\
	for(_i = 0; _i < _vars; _i++) {\
		FILE *ipf = fopen(_fileNames[_i], "r");\
		if(_fVars[_i]) {\
			for(_j = 0; _j < _varsSizes[_i]; _j++)\
				fscanf(ipf, "%f", &(_fVars[_i])[_j]);\
		}\
		else if(_f4Vars[_i]) {\
			for(_j = 0; _j < _varsSizes[_i]; _j++) {\
				fscanf(ipf, "%f", &(_f4Vars[_i])[_j].s[0]);\
				fscanf(ipf, "%f", &(_f4Vars[_i])[_j].s[1]);\
				fscanf(ipf, "%f", &(_f4Vars[_i])[_j].s[2]);\
				fscanf(ipf, "%f", &(_f4Vars[_i])[_j].s[3]);\
			}\
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
