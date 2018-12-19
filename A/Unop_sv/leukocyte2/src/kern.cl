// The number of sample points in each ellipse (stencil)
#define NPOINTS 150
// The maximum radius of a sample ellipse
#define MAX_RAD 20
// The total number of sample ellipses
#define NCIRCLES 7
// The size of the structuring element used in dilation
#define STREL_SIZE (12 * 2 + 1)

// Kernel to compute the dilation of the GICOV matrix produced by the GICOV kernel
// Each element (i, j) of the output matrix is set equal to the maximal value in
//  the neighborhood surrounding element (i, j) in the input matrix
// Here the neighborhood is defined by the structuring element (c_strel)
__attribute__((reqd_work_group_size(176,1,1)))
__kernel void dilate_kernel(int img_m, int img_n, int strel_m, int strel_n, __constant float *c_strel,
                            __global float *img, __global float *dilated) {
	// Find the center of the structuring element
	int el_center_i = strel_m / 2;
	int el_center_j = strel_n / 2;

	// Determine this thread's location in the matrix
	int thread_id = get_global_id(0); //(blockIdx.x * blockDim.x) + threadIdx.x;
	int i = thread_id % img_m;
	int j = thread_id / img_m;

	if(j > img_n) return;

	// Initialize the maximum GICOV score seen so far to zero
	float max = 0.0f;
	
	// Iterate across the structuring element in one dimension
	int el_i, el_j, x, y;
	// Lingjie Zhang modificated at 11/06/2015

    if (j < img_n){
        for (el_i = 0; el_i < strel_m; el_i++) {
	    	y = i - el_center_i + el_i;
	    	// Make sure we have not gone off the edge of the matrix
	    	if ( (y >= 0) && (y < img_m) ) {
	    		// Iterate across the structuring element in the other dimension
	    		for (el_j = 0; el_j < strel_n; el_j++) {
	    			x = j - el_center_j + el_j;
	    			// Make sure we have not gone off the edge of the matrix
	    			//  and that the current structuring element value is not zero
	    			if ( (x >= 0) &&
	    				 (x < img_n) &&
	    				 (c_strel[(el_i * strel_n) + el_j] != 0) ) {
	    					// Determine if this is the maximal value seen so far
	    					int addr = (x * img_m) + y;
	    					float temp = img[addr];
	    					if (temp > max) max = temp;
	    			}
	    		}
	    	}
	    }
	    
	    // Store the maximum value found
	    dilated[(i * img_n) + j] = max;
    }
    // end of Lingjie Zhang's modification
}
