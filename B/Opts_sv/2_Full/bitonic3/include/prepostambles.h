#define PREAMBLE(d_DstKey, d_DstKeySz, d_DstKeyC, d_DstKeyCSz, d_DstVal, d_DstValSz, d_DstValC, d_DstValCSz, d_SrcKey, d_SrcKeySz, d_SrcVal, d_SrcValSz, arrayLength, stride, size, sortDir) {\
	int _i, _j;\
	unsigned int _vars = 4;\
	char *_fileNames[] = {\
		"outputDDstKey",\
		"outputDDstVal",\
		"inputDSrcKey",\
		"inputDSrcVal"\
	};\
	void *_varsPointers[] = {\
		d_DstKeyC,\
		d_DstValC,\
		d_SrcKey,\
		d_SrcVal\
	};\
	unsigned int _varsSizes[] = {\
		d_DstKeyCSz,\
		d_DstValCSz,\
		d_SrcKeySz,\
		d_SrcValSz\
	};\
	unsigned int _varsTypeSizes[] = {\
		sizeof(unsigned int),\
		sizeof(unsigned int),\
		sizeof(unsigned int),\
		sizeof(unsigned int)\
	};\
\
	for(_i = 0; _i < _vars; _i++) {\
		FILE *ipf = fopen(_fileNames[_i], "rb");\
		fread(_varsPointers[_i], _varsTypeSizes[_i], _varsSizes[_i], ipf);\
		fclose(ipf);\
	}\
}

