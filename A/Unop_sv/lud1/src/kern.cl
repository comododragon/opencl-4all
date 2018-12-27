/**
 * Copyright (c) 2018 Andre Bannwart Perina and others
 *
 * Adapted from
 * rodinia_3.1/opencl/lud/ocl/lud_kernel.cl
 * Different licensing may apply, please check Rodinia documentation.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#define BLOCK_SIZE 16

__attribute__((reqd_work_group_size(16,1,1)))
__kernel void 
lud_diagonal(__global float *m, 
			 __local  float *shadow,
			 int   matrix_dim, 
			 int   offset)
{ 
	int i,j;
	int tx = get_local_id(0);

	int array_offset = offset*matrix_dim+offset;
	for(i=0; i < BLOCK_SIZE; i++){
		shadow[i * BLOCK_SIZE + tx]=m[array_offset + tx];
		array_offset += matrix_dim;
	}
  
	barrier(CLK_LOCAL_MEM_FENCE);
  
	for(i=0; i < BLOCK_SIZE-1; i++) {

    if (tx>i){
      for(j=0; j < i; j++)
        shadow[tx * BLOCK_SIZE + i] -= shadow[tx * BLOCK_SIZE + j] * shadow[j * BLOCK_SIZE + i];
		shadow[tx * BLOCK_SIZE + i] /= shadow[i * BLOCK_SIZE + i];
    }

	barrier(CLK_LOCAL_MEM_FENCE);
    if (tx>i){

      for(j=0; j < i+1; j++)
        shadow[(i+1) * BLOCK_SIZE + tx] -= shadow[(i+1) * BLOCK_SIZE + j]*shadow[j * BLOCK_SIZE + tx];
    }
    
	barrier(CLK_LOCAL_MEM_FENCE);
    }

    array_offset = (offset+1)*matrix_dim+offset;
    for(i=1; i < BLOCK_SIZE; i++){
      m[array_offset+tx]=shadow[i * BLOCK_SIZE + tx];
      array_offset += matrix_dim;
    }
  
}
