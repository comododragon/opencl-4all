#include <stdlib.h>
#include <time.h>

#define MAX_COLS 2048
#define MAX_ROWS 2048
#define MAX_COLS_P (MAX_COLS + 1)
#define MAX_ROWS_P (MAX_ROWS + 1)
#define BLOCK_SIZE 16
#define WORKSIZE 2048

int gBlosum62[24][24] = {
	{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
	{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
	{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
	{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
	{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
	{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
	{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
	{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
	{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
	{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
	{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
	{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
	{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
	{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
	{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
	{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
	{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
	{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
	{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
	{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
	{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
	{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
	{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
	{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

#define PREAMBLE(reference_d, reference_dSz, input_itemsets_d, input_itemsets_dSz,\
		cols, penalty, blk, block_width, worksize, offset_r, offset_c\
	) {\
	int _i, _j;\
\
	srand(time(NULL));\
\
	for(_i = 0; _i < MAX_COLS_P; _i++)\
		for(_j = 0; _j < MAX_ROWS_P; _j++)\
			input_itemsets_d[_i * MAX_COLS_P + _j] = 0;\
\
	for(_i = 1; _i < MAX_ROWS_P; _i++)\
		input_itemsets_d[_i * MAX_COLS_P] = rand() % 10 + 1;\
\
	for(_i = 1; _i < MAX_COLS_P; _i++)\
		input_itemsets_d[_i] = rand() % 10 + 1;\
\
	for(_i = 1; _i < MAX_COLS_P; _i++)\
		for(_j = 1; _j < MAX_ROWS_P; _j++)\
			reference_d[_i * MAX_COLS_P + _j] = gBlosum62[input_itemsets_d[_i * MAX_COLS_P]][input_itemsets_d[_j]];\
\
	for(_i = 1; _i < MAX_ROWS_P; _i++)\
		input_itemsets_d[_i * MAX_COLS_P] = -_i * penalty;\
\
	for(_i = 1; _i < MAX_COLS_P; _i++)\
		input_itemsets_d[_i] = -_i * penalty;\
\
	/* We are using another logic to break loop */\
	loopFlag = true;\
}

#define LOOPPREAMBLE(reference_d, reference_dSz, input_itemsets_d, input_itemsets_dSz,\
		cols, penalty, blk, block_width, worksize, offset_r, offset_c,\
		loopFlag\
	) {\
	if(blk > WORKSIZE / BLOCK_SIZE) {\
		PRINT_SUCCESS();\
		break;\
	}\
\
	globalSizeNw_Kernel1[0] = BLOCK_SIZE * blk;\
}

#define LOOPPOSTAMBLE(reference_d, reference_dSz, input_itemsets_d, input_itemsets_dSz,\
		cols, penalty, blk, block_width, worksize, offset_r, offset_c,\
		loopFlag\
	) {\
	blk++;\
}

#define POSTAMBLE(reference_d, reference_dSz, input_itemsets_d, input_itemsets_dSz,\
		cols, penalty, blk, block_width, worksize, offset_r, offset_c\
	) {\
	int _i, _j;\
	FILE *opf = fopen("outputReference", "w");\
\
	for(_i = 0; _i < MAX_COLS_P; _i++)\
		for(_j = 0; _j < MAX_ROWS_P; _j++)\
			fprintf(opf, "%d\n", reference_d[_i * MAX_COLS_P + _j]);\
\
	fclose(opf);\
	opf = fopen("outputInputItemsets", "w");\
\
	for(_i = 0; _i < MAX_COLS_P; _i++)\
		for(_j = 0; _j < MAX_ROWS_P; _j++)\
			fprintf(opf, "%d\n", input_itemsets_d[_i * MAX_COLS_P + _j]);\
\
	fclose(opf);\
}
