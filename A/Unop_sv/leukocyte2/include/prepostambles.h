#define PREAMBLE(img_m, img_n, strel_m, strel_n, c_strel, c_strelSz, img, imgSz, dilated, dilatedSz, dilatedC, dilatedCSz) {\
	int _i, _j;\
	unsigned int _vars = 3;\
	char *_fileNames[] = {\
		"inputCStrel",\
		"inputImg",\
		"outputDilated"\
	};\
	float *_fVars[] = {\
		c_strel,\
		img,\
		dilatedC,\
	};\
	int *_iVars[] = {\
		NULL,\
		NULL,\
		NULL\
	};\
	char *_cVars[] = {\
		NULL,\
		NULL,\
		NULL\
	};\
	unsigned int _varsSizes[] = {\
		c_strelSz,\
		imgSz,\
		dilatedCSz\
	};\
\
	for(_i = 0; _i < _vars; _i++) {\
		FILE *ipf = fopen(_fileNames[_i], "r");\
		if(_fVars[_i]) {\
			for(_j = 0; _j < _varsSizes[_i]; _j++)\
				fscanf(ipf, "%f", &(_fVars[_i])[_j]);\
		}\
		else if(_iVars[_i]) {\
			for(_j = 0; _j < _varsSizes[_i]; _j++)\
				fscanf(ipf, "%d", &(_iVars[_i])[_j]);\
		}\
		else if(_cVars[_i]) {\
			for(_j = 0; _j < _varsSizes[_i]; _j++)\
				fscanf(ipf, "%c", &(_cVars[_i])[_j]);\
		}\
		fclose(ipf);\
	}\
}
