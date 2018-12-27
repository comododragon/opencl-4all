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

__attribute__((reqd_work_group_size(16,16,1)))
__kernel void
lud_internal(__global float *m, 
			 __local  float *peri_row,
			 __local  float *peri_col,
			int matrix_dim, 
			int offset)
{
  
  int  bx = get_group_id(0);	
  int  by = get_group_id(1);	
  
  int  tx = get_local_id(0);
  int  ty = get_local_id(1);

  int i;
  float sum;

  int global_row_id = offset + (by+1)*BLOCK_SIZE;
  int global_col_id = offset + (bx+1)*BLOCK_SIZE;

  peri_row[ty * BLOCK_SIZE + tx] = m[(offset+ty)*matrix_dim+global_col_id+tx];
  peri_col[ty * BLOCK_SIZE + tx] = m[(global_row_id+ty)*matrix_dim+offset+tx];

  barrier(CLK_LOCAL_MEM_FENCE);

  sum = 0;
  for (i=0; i < BLOCK_SIZE; i++)
    sum += peri_col[ty * BLOCK_SIZE + i] * peri_row[i * BLOCK_SIZE + tx];
  m[(global_row_id+ty)*matrix_dim+global_col_id+tx] -= sum;


}
