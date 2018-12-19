#define PREAMBLE(data, dataSz, newData, newDataSz, newDataC, newDataCSz, alignment, wCenter, wCardinal, wDiagonal) {\
	int _i, _j;\
	unsigned int _vars = 2;\
	char *_fileNames[] = {\
		"inputData",\
		"outputNewData"\
	};\
	void *_varsPointers[] = {\
		data,\
		newDataC\
	};\
	unsigned int _varsSizes[] = {\
		dataSz,\
		newDataCSz\
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
