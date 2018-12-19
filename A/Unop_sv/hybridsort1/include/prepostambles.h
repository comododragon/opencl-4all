#define PREAMBLE(d_Result, d_ResultSz, d_ResultC, d_ResultCSz, d_Data, d_DataSz, minimum, maximum, dataCount) {\
	int _i, _j;\
	unsigned int _vars = 3;\
	char *_fileNames[] = {\
		"inputDData",\
		"inputDResult",\
		"outputDResult"\
	};\
	float *_fVars[] = {\
		d_Data,\
		NULL,\
		NULL\
	};\
	int *_iVars[] = {\
		NULL,\
		d_Result,\
		d_ResultC\
	};\
	char *_cVars[] = {\
		NULL,\
		NULL,\
		NULL\
	};\
	unsigned int _varsSizes[] = {\
		d_DataSz,\
		d_ResultSz,\
		d_ResultCSz\
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
