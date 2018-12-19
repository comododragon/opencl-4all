#define PREAMBLE(lA, lASz, errLocOut, errLocOutSz, errLocOutC, errLocOutCSz, alphaInvOut, alphaInvOutSz, alphaInvOutC, alphaInvOutCSz, errCnt, errCntSz, errCntC, errCntCSz, loopCount) {\
	int _i, _j;\
	unsigned int _vars = 4;\
	char *_fileNames[] = {\
		"inputLA",\
		"outputErrLocOut",\
		"outputAlphaInvOut",\
		"outputErrCnt"\
	};\
	void *_varsPointers[] = {\
		lA,\
		errLocOutC,\
		alphaInvOutC,\
		errCntC\
	};\
	unsigned int _varsSizes[] = {\
		lASz,\
		errLocOutCSz,\
		alphaInvOutCSz,\
		errCntCSz\
	};\
	unsigned int _varsTypeSizes[] = {\
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
