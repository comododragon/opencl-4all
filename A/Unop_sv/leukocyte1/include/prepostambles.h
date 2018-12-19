#define PREAMBLE(grad_m, grad_x, grad_xSz, grad_y, grad_ySz, c_sin_angle, c_sin_angleSz, c_cos_angle, c_cos_angleSz, c_tX, c_tXSz, c_tY, c_tYSz, gicov, gicovSz, gicovC, gicovCSz, width, height) {\
	int _i, _j;\
	unsigned int _vars = 7;\
	char *_fileNames[] = {\
		"inputGradX",\
		"inputGradY",\
		"inputCSinAngle",\
		"inputCCosAngle",\
		"inputCTX",\
		"inputCTY",\
		"outputGicov",\
	};\
	float *_fVars[] = {\
		grad_x,\
		grad_y,\
		c_sin_angle,\
		c_cos_angle,\
		NULL,\
		NULL,\
		gicovC\
	};\
	int *_iVars[] = {\
		NULL,\
		NULL,\
		NULL,\
		NULL,\
		c_tX,\
		c_tY,\
		NULL\
	};\
	char *_cVars[] = {\
		NULL,\
		NULL,\
		NULL,\
		NULL,\
		NULL,\
		NULL,\
		NULL\
	};\
	unsigned int _varsSizes[] = {\
		grad_xSz,\
		grad_ySz,\
		c_sin_angleSz,\
		c_cos_angleSz,\
		c_tXSz,\
		c_tYSz,\
		gicovCSz\
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
