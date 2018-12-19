#define PREAMBLE(m, mSz, mC, mCSz, matrix_dim, offset) {\
	int _i, _j;\
	unsigned int _vars = 2;\
	char *_fileNames[] = {\
		"inputM",\
		"outputM"\
	};\
	void *_varsPointers[] = {\
		m,\
		mC\
	};\
	unsigned int _varsSizes[] = {\
		mSz,\
		mCSz\
	};\
	unsigned int _varsTypeSizes[] = {\
		sizeof(float),\
		sizeof(float)\
	};\
\
	for(_i = 0; _i < _vars; _i++) {\
		FILE *ipf = fopen(_fileNames[_i], "rb");\
		fread(_varsPointers[_i], _varsTypeSizes[_i], _varsSizes[_i], ipf);\
		fclose(ipf);\
	}\
}
