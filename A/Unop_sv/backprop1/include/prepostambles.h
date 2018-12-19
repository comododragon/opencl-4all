#include "bpnn.h"

BPNN *gNet;
float gOutErr, gHidErr;

#define PREAMBLE(input_cuda, input_cudaSz, input_hidden_cuda, input_hidden_cudaSz, hidden_partial_sum, hidden_partial_sumSz, hidden_partial_sumC, hidden_partial_sumCSz, in, hid) {\
	int _i, _j;\
\
	srand(7);\
	gNet = bpnn_create(in, hid, 1);\
	bpnn_load(in, gNet);\
\
	for(_i = 0; _i < input_cudaSz; _i++) {\
		input_cuda[_i] = gNet->input_units[_i];\
\
		for(_j = 0; _j < (hid + 1); _j++)\
			input_hidden_cuda[(_i * (hid + 1)) + _j] = gNet->input_weights[_i][_j];\
	}\
\
	FILE *ipf = fopen("outputHiddenPartialSum", "r");\
\
	for(_i = 0; _i < hidden_partial_sumCSz; _i++)\
		fscanf(ipf, "%f", &hidden_partial_sumC[_i]);\
\
	fclose(ipf);\
}

#define CLEANUP(input_cuda, input_cudaSz, input_hidden_cuda, input_hidden_cudaSz, hidden_partial_sum, hidden_partial_sumSz, hidden_partial_sumC, hidden_partial_sumCSz, in, hid) {\
	bpnn_free(gNet);\
}
