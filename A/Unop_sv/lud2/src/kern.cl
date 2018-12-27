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

__attribute__((reqd_work_group_size(32,1,1)))
__kernel void
lud_perimeter(__global float *m, 
			  __local  float *dia,
			  __local  float *peri_row,
			  __local  float *peri_col,
			  int matrix_dim, 
			  int offset)
{
    int i,j, array_offset;
    int idx;

    int  bx = get_group_id(0);	
    int  tx = get_local_id(0);

    if (tx < BLOCK_SIZE) {
      idx = tx;
      array_offset = offset*matrix_dim+offset;
      for (i=0; i < BLOCK_SIZE/2; i++){
      dia[i * BLOCK_SIZE + idx]=m[array_offset+idx];
      array_offset += matrix_dim;
      }
    
    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_row[i * BLOCK_SIZE+ idx]=m[array_offset+(bx+1)*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }

    } else {
    idx = tx-BLOCK_SIZE;
    
    array_offset = (offset+BLOCK_SIZE/2)*matrix_dim+offset;
    for (i=BLOCK_SIZE/2; i < BLOCK_SIZE; i++){
      dia[i * BLOCK_SIZE + idx]=m[array_offset+idx];
      array_offset += matrix_dim;
    }
    
    array_offset = (offset+(bx+1)*BLOCK_SIZE)*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_col[i * BLOCK_SIZE + idx] = m[array_offset+idx];
      array_offset += matrix_dim;
    }
  
   }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tx < BLOCK_SIZE) { //peri-row
     idx=tx;
      for(i=1; i < BLOCK_SIZE; i++){
      for (j=0; j < i; j++)
        peri_row[i * BLOCK_SIZE + idx]-=dia[i * BLOCK_SIZE+ j]*peri_row[j * BLOCK_SIZE + idx];
    }
    } else { //peri-col
     idx=tx - BLOCK_SIZE;
     for(i=0; i < BLOCK_SIZE; i++){
      for(j=0; j < i; j++)
        peri_col[idx * BLOCK_SIZE + i]-=peri_col[idx * BLOCK_SIZE+ j]*dia[j * BLOCK_SIZE + i];
      peri_col[idx * BLOCK_SIZE + i] /= dia[i * BLOCK_SIZE+ i];
     }
   }

	barrier(CLK_LOCAL_MEM_FENCE);
    
  if (tx < BLOCK_SIZE) { //peri-row
    idx=tx;
    array_offset = (offset+1)*matrix_dim+offset;
    for(i=1; i < BLOCK_SIZE; i++){
      m[array_offset+(bx+1)*BLOCK_SIZE+idx] = peri_row[i*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }
  } else { //peri-col
    idx=tx - BLOCK_SIZE;
    array_offset = (offset+(bx+1)*BLOCK_SIZE)*matrix_dim+offset;
    for(i=0; i < BLOCK_SIZE; i++){
      m[array_offset+idx] =  peri_col[i*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }
  }

}
