#define PREAMBLE(weights, weightsSz, weightsC, weightsCSz, Nparticles, partial_sums, partial_sumsSz, CDF, CDFSz, u, uSz, uC, uCSz, seed, seedSz) {\
	int _i, _j;\
	unsigned int _vars = 5;\
	char *_fileNames[] = {\
		"inputWeights",\
		"outputWeights",\
		"inputPartialSums",\
		"outputU",\
		"inputSeed",\
	};\
	double *_dVars[] = {\
		weights,\
		weightsC,\
		partial_sums,\
		uC,\
		NULL\
	};\
	int *_iVars[] = {\
		NULL,\
		NULL,\
		NULL,\
		NULL,\
		seed\
	};\
	char *_cVars[] = {\
		NULL,\
		NULL,\
		NULL,\
		NULL,\
		NULL\
	};\
	unsigned int _varsSizes[] = {\
		weightsSz,\
		weightsCSz,\
		partial_sumsSz,\
		uCSz,\
		seedSz\
	};\
\
	for(_i = 0; _i < _vars; _i++) {\
		FILE *ipf = fopen(_fileNames[_i], "r");\
		if(_dVars[_i]) {\
			for(_j = 0; _j < _varsSizes[_i]; _j++)\
				fscanf(ipf, "%lf", &(_dVars[_i])[_j]);\
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
