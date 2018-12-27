/**
 * Copyright (c) 2018 Andre Bannwart Perina and others
 *
 * Adapted from
 * https://github.com/fpga-opencl-benchmarks/rodinia_fpga
 * Different licensing may apply, please check the repository documentation.
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

#define INT_SIMD 2
// XXX: INT_CU must be 1 for correct simulation (seems like a simulation bug when local memory and CUs are used)
#define INT_CU 1
#define BSIZE 96

__attribute__((num_compute_units(INT_CU)))
__attribute__((num_simd_work_items(INT_SIMD)))
__attribute__((reqd_work_group_size(BSIZE,BSIZE,1)))
__kernel void lud_internal(__global float* restrict m,
                                    int             matrix_dim,
                                    int             offset)
{
	__local float peri_row[BSIZE * BSIZE], peri_col[BSIZE * BSIZE];

	int bx = get_group_id(0);
	int by = get_group_id(1);
  
	int tx = get_local_id(0);
	int ty = get_local_id(1);

	int global_row_id = (by + 1) * BSIZE;
	int global_col_id = (bx + 1) * BSIZE;

	peri_row[ty * BSIZE + tx] = m[offset + (ty) * matrix_dim + global_col_id + tx];
	peri_col[ty * BSIZE + tx] = m[offset + (global_row_id + ty) * matrix_dim + tx];

	barrier(CLK_LOCAL_MEM_FENCE);

	float sum = 0;
	#pragma unroll
	for (int i = 0; i < BSIZE; i++)
	{
		sum += peri_col[ty * BSIZE + i] * peri_row[i * BSIZE + tx];
	}
	m[offset + (global_row_id + ty) * matrix_dim + global_col_id + tx] -= sum;
}

