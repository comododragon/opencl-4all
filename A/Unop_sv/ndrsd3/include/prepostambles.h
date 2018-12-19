#define PREAMBLE(lambda, lambdaSz, omega, omegaSz, errCnt, errCntSz, errLoc, errLocSz, alphaInv, alphaInvSz, errOut, errOutSz, errOutC, errOutCSz, loopCount) {\
	int _i, _j;\
	unsigned int _vars = 6;\
	char *_fileNames[] = {\
		"inputLambda",\
		"inputOmega",\
		"inputErrCnt",\
		"inputErrLoc",\
		"inputAlphaInv",\
		"outputErrOut"\
	};\
	void *_varsPointers[] = {\
		lambda,\
		omega,\
		errCnt,\
		errLoc,\
		alphaInv,\
		errOutC\
	};\
	unsigned int _varsSizes[] = {\
		lambdaSz,\
		omegaSz,\
		errCntSz,\
		errLocSz,\
		alphaInvSz,\
		errOutCSz\
	};\
	unsigned int _varsTypeSizes[] = {\
		sizeof(short),\
		sizeof(short),\
		sizeof(short),\
		sizeof(short),\
		sizeof(short),\
		sizeof(short)\
	};\
\
	for(_i = 0; _i < _vars; _i++) {\
		FILE *ipf = fopen(_fileNames[_i], "rb");\
		fread(_varsPointers[_i], _varsTypeSizes[_i], _varsSizes[_i], ipf);\
		fclose(ipf);\
	}\
}
