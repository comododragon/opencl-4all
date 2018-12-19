#define PREAMBLE(p_weight, p_weightSz, p_assign, p_assignSz, p_cost, p_costSz, coord_d, coord_dSz, work_mem_d, work_mem_dSz, work_mem_dC, work_mem_dCSz, center_table_d, center_table_dSz, switch_membership_d, switch_membership_dSz, switch_membership_dC, switch_membership_dCSz, dim, x, K) {\
	int _i, _j;\
	unsigned int _vars = 7;\
	char *_fileNames[] = {\
		"inputPWeight",\
		"inputPAssign",\
		"inputPCost",\
		"inputCoordD",\
		"outputWorkMemD",\
		"inputCenterTableD",\
		"outputSwitchMembershipD"\
	};\
	float *_fVars[] = {\
		p_weight,\
		NULL,\
		p_cost,\
		coord_d,\
		work_mem_dC,\
		NULL,\
		NULL\
	};\
	int *_iVars[] = {\
		NULL,\
		NULL,\
		NULL,\
		NULL,\
		NULL,\
		center_table_d,\
		NULL\
	};\
	long *_lVars[] = {\
		NULL,\
		p_assign,\
		NULL,\
		NULL,\
		NULL,\
		NULL,\
		NULL\
	};\
	char *_cVars[] = {\
		NULL,\
		NULL,\
		NULL,\
		NULL,\
		NULL,\
		NULL,\
		switch_membership_dC\
	};\
	unsigned int _varsSizes[] = {\
		p_weightSz,\
		p_assignSz,\
		p_costSz,\
		coord_dSz,\
		work_mem_dSz,\
		center_table_dSz,\
		switch_membership_dSz\
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
		else if(_lVars[_i]) {\
			for(_j = 0; _j < _varsSizes[_i]; _j++)\
				fscanf(ipf, "%ld", &(_lVars[_i])[_j]);\
		}\
		else if(_cVars[_i]) {\
			for(_j = 0; _j < _varsSizes[_i]; _j++)\
				fscanf(ipf, "%c", &(_cVars[_i])[_j]);\
		}\
		fclose(ipf);\
	}\
}
