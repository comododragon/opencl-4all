#include <stdlib.h>
#include <time.h>

#define BOXES1D 10
#define ALPHA 0.5
#define NUMBER_BOXES BOXES1D * BOXES1D * BOXES1D
#define NUMBER_PAR_PER_BOX 100
#define SPACE_ELEM NUMBER_BOXES * NUMBER_PAR_PER_BOX
#define SPACE_MEM SPACE_ELEM * sizeof(cl_float4)
#define SPACE_MEM2 SPACE_ELEM * sizeof(float)
#define BOX_MEM NUMBER_BOXES * sizeof(box_str)

typedef struct {
	cl_int3 xyz; // unused
	int number;
	long offset; //unused
} nei_str;

typedef struct {
	cl_int3 xyz; //unused
	int number; //unused
	long offset;

	int nn;
	nei_str nei[26];
} box_str;

#define PREAMBLE(d_par_gpu_alpha, d_dim_gpu_number_boxes,\
		d_box_gpu_offset, d_box_gpu_offsetSz, d_box_gpu_nn, d_box_gpu_nnSz, d_box_gpu_nei_number, d_box_gpu_nei_numberSz,\
		d_rv_gpu, d_rv_gpuSz, d_qv_gpu, d_qv_gpuSz, d_fv_gpu, d_fv_gpuSz\
	) {\
	int _i, _j, _k, _l, _m, _n;\
	box_str _boxCpu[NUMBER_BOXES];\
	int _nh = 0;\
\
	d_par_gpu_alpha = ALPHA;\
\
	for(_i = 0; _i < BOXES1D; _i++) {\
		for(_j = 0; _j < BOXES1D; _j++) {\
			for(_k = 0; _k < BOXES1D; _k++) {\
				_boxCpu[_nh].xyz.x = _k;\
				_boxCpu[_nh].xyz.y = _j;\
				_boxCpu[_nh].xyz.z = _i;\
				_boxCpu[_nh].number = _nh;\
				_boxCpu[_nh].offset = _nh * NUMBER_PAR_PER_BOX;\
\
				_boxCpu[_nh].nn = 0;\
\
				for(_l = -1; _l < 2; _l++) {\
					for(_m = -1; _m < 2; _m++) {\
						for(_n = -1; _n < 2; _n++) {\
							if(((_i + _l >= 0 && _j + _m >= 0 && _k + _n >= 0) && (_i + _l < BOXES1D && _j + _m < BOXES1D && _k + _n < BOXES1D)) && !(!_l && !_m && !_n)) {\
								_boxCpu[_nh].nei[_boxCpu[_nh].nn].xyz.x = _k + _n;\
								_boxCpu[_nh].nei[_boxCpu[_nh].nn].xyz.y = _j + _m;\
								_boxCpu[_nh].nei[_boxCpu[_nh].nn].xyz.z = _i + _l;\
								_boxCpu[_nh].nei[_boxCpu[_nh].nn].number =\
									(_boxCpu[_nh].nei[_boxCpu[_nh].nn].xyz.z * BOXES1D * BOXES1D) +\
									(_boxCpu[_nh].nei[_boxCpu[_nh].nn].xyz.y * BOXES1D) +\
									_boxCpu[_nh].nei[_boxCpu[_nh].nn].xyz.x;\
								_boxCpu[_nh].nei[_boxCpu[_nh].nn].offset =\
									_boxCpu[_nh].nei[_boxCpu[_nh].nn].number * NUMBER_PAR_PER_BOX;\
\
								(_boxCpu[_nh].nn)++;\
							}\
						}\
					}\
				}\
\
				_nh++;\
			}\
		}\
	}\
\
	for(_i = 0; _i < NUMBER_BOXES; _i++) {\
		d_box_gpu_offset[_i] = _boxCpu[_i].offset;\
		d_box_gpu_nn[_i] = _boxCpu[_i].nn;\
\
		for(_j = 0; _j < 26; _j++)\
			d_box_gpu_nei_number[_i * 26 + _j] = _boxCpu[_i].nei[_j].number;\
	}\
\
	srand(0);\
\
	for(_i = 0; _i < SPACE_ELEM; _i++) {\
		d_rv_gpu[_i].w = (rand() % 10 + 1) / 10.0;\
		d_rv_gpu[_i].x = (rand() % 10 + 1) / 10.0;\
		d_rv_gpu[_i].y = (rand() % 10 + 1) / 10.0;\
		d_rv_gpu[_i].z = (rand() % 10 + 1) / 10.0;\
	}\
\
	for(_i = 0; _i < SPACE_ELEM; _i++)\
		d_qv_gpu[_i] = (rand() % 10 + 1) / 10.0;\
\
	for(_i = 0; _i < SPACE_ELEM; _i++) {\
		d_fv_gpu[_i].w = 0;\
		d_fv_gpu[_i].x = 0;\
		d_fv_gpu[_i].y = 0;\
		d_fv_gpu[_i].z = 0;\
	}\
}

#define POSTAMBLE(d_par_gpu_alpha, d_dim_gpu_number_boxes,\
		d_box_gpu_offset, d_box_gpu_offsetSz, d_box_gpu_nn, d_box_gpu_nnSz, d_box_gpu_nei_number, d_box_gpu_nei_numberSz,\
		d_rv_gpu, d_rv_gpuSz, d_qv_gpu, d_qv_gpuSz, d_fv_gpu, d_fv_gpuSz\
	) {\
	int _i;\
\
	FILE *_opf = fopen("result", "w");\
	for(_i = 0; _i < SPACE_ELEM; _i++)\
		fprintf(_opf, "%.2f, %.2f, %.2f, %.2f\n", d_fv_gpu[_i].w, d_fv_gpu[_i].x, d_fv_gpu[_i].y, d_fv_gpu[_i].z);\
\
	fclose(_opf);\
}
