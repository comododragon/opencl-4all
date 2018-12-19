#define PREAMBLE(arrayX, arrayXSz, arrayXC, arrayXCSz, arrayY, arrayYSz, arrayYC, arrayYCSz, xj, xjSz, yj, yjSz, ind, indSz, indC, indCSz, objxy, objxySz, likelihood, likelihoodSz, likelihoodC, likelihoodCSz, I, ISz, weights, weightsSz, weightsC, weightsCSz, Nparticles, countOnes, max_size, k, IszY, Nfr, seed, seedSz, partial_sums, partial_sumsSz, partial_sumsC, partial_sumsCSz) {\
	int _i, _j;\
	unsigned int _vars = 11;\
	char *_fileNames[] = {\
		"outputArrayX",\
		"outputArrayY",\
		"inputXj",\
		"inputYj",\
		"outputInd",\
		"inputObjxy",\
		"outputLikelihood",\
		"inputI",\
		"outputWeights",\
		"inputSeed",\
		"outputPartialSums"\
	};\
	double *_dVars[] = {\
		arrayXC,\
		arrayYC,\
		xj,\
		yj,\
		NULL,\
		NULL,\
		likelihoodC,\
		NULL,\
		weightsC,\
		NULL,\
		partial_sumsC\
	};\
	int *_iVars[] = {\
		NULL,\
		NULL,\
		NULL,\
		NULL,\
		indC,\
		objxy,\
		NULL,\
		NULL,\
		NULL,\
		seed,\
		NULL\
	};\
	char *_cVars[] = {\
		NULL,\
		NULL,\
		NULL,\
		NULL,\
		NULL,\
		NULL,\
		NULL,\
		I,\
		NULL,\
		NULL,\
		NULL\
	};\
	unsigned int _varsSizes[] = {\
		arrayXCSz,\
		arrayYCSz,\
		xjSz,\
		yjSz,\
		indCSz,\
		objxySz,\
		likelihoodCSz,\
		ISz,\
		weightsCSz,\
		seedSz,\
		partial_sumsCSz\
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
