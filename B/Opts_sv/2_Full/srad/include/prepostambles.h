#define NR 502
#define NC 458
#define NE 229916
#define NITER 10
#define GLOBAL 230144
#define LOCAL 256

void srad2_soft(int d_Nr, long d_Ne,
		int *d_iN, int *d_iS, int *d_jE, int *d_jW,
		float *d_dN, float *d_dS, float *d_dE, float *d_dW,
		float *d_c, float *d_I
	) {
	int i;
	float d_lambda = 0.5;
	int d_Nc = 458;

	for(i = 0; i < 230144; i++) {
		int bx = i / 256;
		int tx = i % 256;
		int ei = bx * LOCAL + tx;
		int row;
		int col;

		float d_cN, d_cS, d_cW, d_cE;
		float d_D;

		row = (ei + 1) % d_Nr - 1;
		col = (ei + 1) / d_Nr + 1 - 1;
		if(!((ei + 1) % d_Nr)) {
			row = d_Nr - 1;
			col--;
		}

		if(ei < d_Ne) {
			d_cN = d_c[ei];
			d_cS = d_c[d_iS[row] + d_Nr * col];
			d_cW = d_c[ei];
			d_cE = d_c[row + d_Nr * d_jE[col]];

			d_D = d_cN * d_dN[ei] + d_cS * d_dS[ei] + d_cW * d_dW[ei] + d_cE * d_dE[ei];

			d_I[ei] = d_I[ei] + 0.25 * d_lambda * d_D;
		}
	}
}

int gIndex = 0;
float gQ0sqr[NITER];

#define PREAMBLE(d_Nr, d_Ne, d_iN, d_iNSz, d_iS, d_iSSz, d_jE, d_jESz, d_jW, d_jWSz, d_dN, d_dNSz, d_dS, d_dSSz, d_dE, d_dESz, d_dW, d_dWSz, d_q0sqr, d_c, d_cSz, d_I, d_ISz) {\
	int _i;\
	FILE *ipf;\
\
	for(_i = 0; _i < d_Nr; _i++) {\
		d_iN[_i] = _i - 1;\
		d_iS[_i] = _i + 1;\
	}\
\
	/* d_jW and d_jE buffers are swapped to the srad1 and srad2 kernels in original implementation. Perhaps a bug? */\
	/* We are maintaining the same logic here (i.e. d_jW and d_jE are assigned swapped) */\
	for(_i = 0; _i < 458; _i++) {\
		/*d_jW[_i] = _i - 1;*/\
		/*d_jE[_i] = _i + 1;*/\
		d_jE[_i] = _i - 1;\
		d_jW[_i] = _i + 1;\
	}\
\
	d_iN[0] = 0;\
	d_iS[NR - 1] = NR - 1;\
	/*d_jW[0] = 0;*/\
	/*d_jE[NC - 1] = NC - 1;*/\
	d_jE[0] = 0;\
	d_jW[NC - 1] = NC - 1;\
\
	ipf = fopen("inI", "r");\
	for(_i = 0; _i < NE; _i++) {\
		fscanf(ipf, "%f", &d_I[_i]);\
		fgetc(ipf);\
	}\
	fclose(ipf);\
\
	ipf = fopen("inQ0sqr", "r");\
	for(_i = 0; _i < NITER; _i++) {\
		fscanf(ipf, "%f", &gQ0sqr[_i]);\
		fgetc(ipf);\
	}\
	fclose(ipf);\
\
	/* We are using another logic to break loop */\
	loopFlag = true;\
}

#define LOOPPREAMBLE(d_Nr, d_Ne, d_iN, d_iNSz, d_iS, d_iSSz, d_jE, d_jESz, d_jW, d_jWSz, d_dN, d_dNSz, d_dS, d_dSSz, d_dE, d_dESz, d_dW, d_dWSz, d_q0sqr, d_c, d_cSz, d_I, d_ISz, loopFlag) {\
	if(gIndex >= NITER) {\
		PRINT_SUCCESS();\
		break;\
	}\
\
	d_q0sqr = gQ0sqr[gIndex];\
}

#define LOOPPOSTAMBLE(d_Nr, d_Ne, d_iN, d_iNSz, d_iS, d_iSSz, d_jE, d_jESz, d_jW, d_jWSz, d_dN, d_dNSz, d_dS, d_dSSz, d_dE, d_dESz, d_dW, d_dWSz, d_q0sqr, d_c, d_cSz, d_I, d_ISz, loopFlag) {\
	/* Software version of srad2_kernel */\
	srad2_soft(d_Nr, d_Ne, d_iN, d_iS, d_jE, d_jW, d_dN, d_dS, d_dE, d_dW, d_c, d_I);\
\
	gIndex++;\
}

#define POSTAMBLE(d_Nr, d_Ne, d_iN, d_iNSz, d_iS, d_iSSz, d_jE, d_jESz, d_jW, d_jWSz, d_dN, d_dNSz, d_dS, d_dSSz, d_dE, d_dESz, d_dW, d_dWSz, d_q0sqr, d_c, d_cSz, d_I, d_ISz) {\
	int _i;\
	FILE *opf;\
\
	opf = fopen("outN", "w");\
	for(_i = 0; _i < NE; _i++)\
		fprintf(opf, "%f\n", d_dN[_i]);\
	fclose(opf);\
\
	opf = fopen("outS", "w");\
	for(_i = 0; _i < NE; _i++)\
		fprintf(opf, "%f\n", d_dS[_i]);\
	fclose(opf);\
\
	opf = fopen("outE", "w");\
	for(_i = 0; _i < NE; _i++)\
		fprintf(opf, "%f\n", d_dE[_i]);\
	fclose(opf);\
\
	opf = fopen("outW", "w");\
	for(_i = 0; _i < NE; _i++)\
		fprintf(opf, "%f\n", d_dW[_i]);\
	fclose(opf);\
\
	opf = fopen("outC", "w");\
	for(_i = 0; _i < NE; _i++)\
		fprintf(opf, "%f\n", d_c[_i]);\
	fclose(opf);\
}
