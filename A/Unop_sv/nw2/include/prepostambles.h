#define MAX_COLS 2048
#define MAX_ROWS 2048
#define MAX_COLS_P (MAX_COLS + 1)
#define MAX_ROWS_P (MAX_ROWS + 1)
#define BLOCK_SIZE 16
#define WORKSIZE 2048

#define PREAMBLE(reference_d, reference_dSz, input_itemsets_d, input_itemsets_dSz,\
		cols, penalty, blk, block_width, worksize, offset_r, offset_c\
	) {\
	int _i, _j;\
	FILE *ipf = fopen("inputReference", "r");\
\
	for(_i = 0; _i < MAX_COLS_P; _i++) {\
		for(_j = 0; _j < MAX_ROWS_P; _j++) {\
			fscanf(ipf, "%d", &reference_d[_i * MAX_COLS_P + _j]);\
			fgetc(ipf);\
		}\
	}\
\
	fclose(ipf);\
	ipf = fopen("inputInputItemsets", "r");\
\
	for(_i = 0; _i < MAX_COLS_P; _i++) {\
		for(_j = 0; _j < MAX_ROWS_P; _j++) {\
			fscanf(ipf, "%d", &input_itemsets_d[_i * MAX_COLS_P + _j]);\
			fgetc(ipf);\
		}\
	}\
\
	fclose(ipf);\
\
	/* We are using another logic to break loop */\
	loopFlag = true;\
}

#define LOOPPREAMBLE(reference_d, reference_dSz, input_itemsets_d, input_itemsets_dSz,\
		cols, penalty, blk, block_width, worksize, offset_r, offset_c,\
		loopFlag\
	) {\
	if(blk < 1) {\
		PRINT_SUCCESS();\
		break;\
	}\
\
	globalSizeNw_Kernel2[0] = BLOCK_SIZE * blk;\
}

#define LOOPPOSTAMBLE(reference_d, reference_dSz, input_itemsets_d, input_itemsets_dSz,\
		cols, penalty, blk, block_width, worksize, offset_r, offset_c,\
		loopFlag\
	) {\
	blk--;\
}

#define POSTAMBLE(reference_d, reference_dSz, input_itemsets_d, input_itemsets_dSz,\
		cols, penalty, blk, block_width, worksize, offset_r, offset_c\
	) {\
	int _i, _j;\
	FILE *opf = fopen("output", "w");\
\
	for(_i = 0; _i < MAX_COLS_P; _i++)\
		for(_j = 0; _j < MAX_ROWS_P; _j++)\
			fprintf(opf, "%d\n", input_itemsets_d[_i * MAX_COLS_P + _j]);\
\
	fclose(opf);\
}
