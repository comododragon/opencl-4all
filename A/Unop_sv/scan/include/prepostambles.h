#define PREAMBLE(in, inSz, isums, isumsSz, out, outSz, outC, outCSz, n) {\
	int _i, _j;\
	unsigned int _vars = 3;\
	char *_fileNames[] = {\
		"inputIn",\
		"inputIsums",\
		"outputOut"\
	};\
	void *_varsPointers[] = {\
		in,\
		isums,\
		outC\
	};\
	unsigned int _varsSizes[] = {\
		inSz,\
		isumsSz,\
		outCSz\
	};\
	unsigned int _varsTypeSizes[] = {\
		sizeof(float),\
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
