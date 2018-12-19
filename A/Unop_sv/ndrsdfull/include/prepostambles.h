#define PREAMBLE(r, rSz, out, outSz, outC, outCSz, loopCount) {\
	int _i, _j;\
	unsigned int _vars = 2;\
	char *_fileNames[] = {\
		"inputR",\
		"outputOut"\
	};\
	void *_varsPointers[] = {\
		r,\
		outC\
	};\
	unsigned int _varsSizes[] = {\
		rSz,\
		outCSz\
	};\
	unsigned int _varsTypeSizes[] = {\
		1,\
		1\
	};\
\
	for(_i = 0; _i < _vars; _i++) {\
		FILE *ipf = fopen(_fileNames[_i], "rb");\
		fread(_varsPointers[_i], _varsTypeSizes[_i], _varsSizes[_i], ipf);\
		fclose(ipf);\
	}\
}
