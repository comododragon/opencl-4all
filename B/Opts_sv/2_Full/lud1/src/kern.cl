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

#define DIA_UNROLL 4
#define BSIZE 96

__attribute__((reqd_work_group_size(BSIZE,1,1)))
__kernel void lud_diagonal(__global float* restrict m, 
                                    int             matrix_dim,
                                    int             offset)
{ 
	int tx = get_local_id(0);
	__local float __attribute__((memory, numbanks(1), bankwidth(4*DIA_UNROLL), doublepump, numreadports(3), numwriteports(1))) shadow_row[BSIZE * BSIZE];
	__local float __attribute__((memory, numbanks(1), bankwidth(4*DIA_UNROLL), doublepump, numreadports(3), numwriteports(1))) shadow_col[BSIZE * BSIZE];

	int array_offset = offset;
	for(int i = 0; i < BSIZE; i++)
	{
		shadow_row[i * BSIZE + tx] = m[array_offset + tx];
		shadow_col[tx * BSIZE + i] = m[array_offset + tx];
		array_offset += matrix_dim;
	}
  
	barrier(CLK_LOCAL_MEM_FENCE);

	array_offset = offset + matrix_dim;
	for(int i = 0; i < BSIZE - 1; i++)
	{
		if (tx > i)
		{
			float sum = 0.0f;
			#pragma unroll DIA_UNROLL
			for(int j = 0; j < i; j++)
			{
				sum += shadow_row[tx * BSIZE + j] * shadow_col[i * BSIZE + j];
			}
			shadow_row[tx * BSIZE + i] = (shadow_row[tx * BSIZE + i] - sum) / shadow_col[i * BSIZE + i];
			//shadow_col[i * BSIZE + tx] = shadow_row[tx * BSIZE + i];		// commented out since it is not actually required and output is correct either way
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if (tx > i)
		{
			float sum = 0.0f;
			#pragma unroll DIA_UNROLL
			for(int j = 0; j < i + 1; j++)
			{
				sum += shadow_row[(i + 1) * BSIZE + j] * shadow_col[tx * BSIZE + j];
			}
			shadow_row[(i + 1) * BSIZE + tx] -= sum;
			shadow_col[tx * BSIZE + (i + 1)] = shadow_row[(i + 1) * BSIZE + tx];
		}
		m[array_offset + tx] = shadow_row[(i + 1) * BSIZE + tx];
		array_offset += matrix_dim;
	}
}
