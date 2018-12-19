#define BLOCK_SIZE 96

#define PREAMBLE(m, mSz, mC, mCSz, matrix_dim, offset) {\
	int _i;\
\
	FILE *ipf = fopen("inputM", "r");\
	for(_i = 0; _i < mSz; _i++)\
		fscanf(ipf, "%f", &m[_i]);\
	fclose(ipf);\
\
	ipf = fopen("outputM", "r");\
	for(_i = 0; _i < mCSz; _i++)\
		fscanf(ipf, "%f", &mC[_i]);\
	fclose(ipf);\
}

#if 0
#define PREAMBLE(m, mSz, mC, mCSz, matrix_dim, offset) {\
	int _i;\
\
	FILE *ipf = fopen("inputM", "r");\
	for(_i = 0; _i < mSz; _i++)\
		fscanf(ipf, "%f", &m[_i]);\
	fclose(ipf);\
\
	ipf = fopen("outputM", "r");\
	for(_i = 0; _i < mCSz; _i++)\
		fscanf(ipf, "%f", &mC[_i]);\
	fclose(ipf);\
\
	offset = 0;\
\
	/* We are using another logic to break loop */\
	loopFlag = true;\
}

#define LOOPPREAMBLE(m, mSz, mC, mCSz, matrix_dim, offset, loopFlag) {\
	if(offset >= (matrix_dim - BLOCK_SIZE))\
		break;\
}

#define LOOPPOSTAMBLE(m, mSz, mC, mCSz, matrix_dim, offset, loopFlag) {\
	offset += BLOCK_SIZE;\
}
#endif
