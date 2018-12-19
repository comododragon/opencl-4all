#define PREAMBLE(variables, variablesSz, areas, areasSz, step_factors, step_factorsSz, step_factorsC, step_factorsCSz, nelr) {\
	int _i, _j;\
	unsigned int _vars = 3;\
	char *_fileNames[] = {\
		"inputVariables",\
		"inputAreas",\
		"outputStepFactors"\
	};\
	void *_varsPointers[] = {\
		variables,\
		areas,\
		step_factorsC\
	};\
	unsigned int _varsSizes[] = {\
		variablesSz,\
		areasSz,\
		step_factorsCSz\
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
