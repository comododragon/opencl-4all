#define PREAMBLE(s, sSz, c, cSz, cC, cCSz, w, wSz, wC, wCSz, loopCount) {\
	int _i, _j;\
	unsigned int _vars = 3;\
	char *_fileNames[] = {\
		"inputS",\
		"outputC",\
		"outputW"\
	};\
	void *_varsPointers[] = {\
		s,\
		cC,\
		wC\
	};\
	unsigned int _varsSizes[] = {\
		sSz,\
		cCSz,\
		wCSz\
	};\
	unsigned int _varsTypeSizes[] = {\
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
