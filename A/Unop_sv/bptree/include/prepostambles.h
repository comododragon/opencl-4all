#define PREAMBLE(height, knodesDLocation, knodesDLocationSz, knodesDIndices, knodesDIndicesSz, knodesDKeys, knodesDKeysSz, knodesDIsLeaf, knodesDIsLeafSz, knodesDNumKeys, knodesDNumKeysSz, knodes_elem, recordsD, recordsDSz, currKnodeD, currKnodeDSz, currKnodeDC, currKnodeDCSz, offsetD, offsetDSz, offsetDC, offsetDCSz, keysD, keysDSz, ansD, ansDSz, ansDC, ansDCSz) {\
	int _i, _j;\
	unsigned int _vars = 10;\
	char *_fileNames[] = {\
		"inputKnodesDLocation",\
		"inputKnodesDIndices",\
		"inputKnodesDKeys",\
		"inputKnodesDIsLeaf",\
		"inputKnodesDNumKeys",\
		"inputRecordsD",\
		"outputCurrKnodeD",\
		"outputOffsetD",\
		"inputKeysD",\
		"outputAnsD"\
	};\
	void *_varsPointers[] = {\
		knodesDLocation,\
		knodesDIndices,\
		knodesDKeys,\
		knodesDIsLeaf,\
		knodesDNumKeys,\
		recordsD,\
		currKnodeDC,\
		offsetDC,\
		keysD,\
		ansDC\
	};\
	unsigned int _varsSizes[] = {\
		knodesDLocationSz,\
		knodesDIndicesSz,\
		knodesDKeysSz,\
		knodesDIsLeafSz,\
		knodesDNumKeysSz,\
		recordsDSz,\
		currKnodeDCSz,\
		offsetDCSz,\
		keysDSz,\
		ansDCSz\
	};\
	unsigned int _varsTypeSizes[] = {\
		sizeof(int),\
		sizeof(int),\
		sizeof(int),\
		sizeof(bool),\
		sizeof(int),\
		sizeof(int),\
		sizeof(long),\
		sizeof(long),\
		sizeof(int),\
		sizeof(int)\
	};\
\
	for(_i = 0; _i < _vars; _i++) {\
		FILE *ipf = fopen(_fileNames[_i], "rb");\
		fread(_varsPointers[_i], _varsTypeSizes[_i], _varsSizes[_i], ipf);\
		fclose(ipf);\
	}\
}
