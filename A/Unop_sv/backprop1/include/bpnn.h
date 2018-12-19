#ifndef BPNN_H
#define BPNN_H

#define BIGRND 0x7fffffff
#define THREADS 256
#define WIDTH 16
#define HEIGHT 16
#define BLOCK_SIZE 16

#define ETA 0.3
#define MOMENTUM 0.3
#define NUM_THREAD 4

typedef struct {
	/* number of input units */
	int input_n;
	/* number of hidden units */
	int hidden_n;
	/* number of output units */
	int output_n;

	/* the input units */
	float *input_units;
	/* the hidden units */
	float *hidden_units;
	/* the output units */
	float *output_units;

	/* storage for hidden unit error */
	float *hidden_delta;
	/* storage for output unit error */
	float *output_delta;

	/* storage for target vector */
	float *target;

	/* weights from input to hidden layer */
	float **input_weights;
	/* weights from hidden to output layer */
	float **hidden_weights;

	/*** The next two are for momentum ***/

	/* previous change on input to hidden wgt */
	float **input_prev_weights;
	/* previous change on hidden to output wgt */
	float **hidden_prev_weights;
} BPNN;

/*** User-level functions ***/

void bpnn_initialize(int seed);
BPNN *bpnn_create(int n_in, int n_hidden, int n_out);
void bpnn_free(BPNN *net);
void bpnn_train(BPNN *net, float *eo, float *eh);
void bpnn_feedforward(BPNN *net);
void bpnn_save(BPNN *net, char *filename);
BPNN *bpnn_read(char *filename);
void bpnn_load(int layer_size, BPNN *net);
int bpnn_train_kernel(BPNN *net, float *eo, float *eh);
void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);
void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);
void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err); 
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);

#endif
