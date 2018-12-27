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

#define PERI_UNROLL	8
// XXX: PERI_CU must be 1 for correct simulation (seems like a simulation bug when local memory and CUs are used)
#define PERI_CU	2
#define BSIZE 96

__attribute__((num_compute_units(PERI_CU)))
__attribute__((reqd_work_group_size(BSIZE * 2,1,1)))
__kernel void lud_perimeter(__global float* restrict m,
                                     int             matrix_dim,
                                     int             offset)
{
	__local float dia_row[BSIZE * BSIZE], dia_col[BSIZE * BSIZE], peri_row[BSIZE * BSIZE];
	__local float __attribute__((memory, numbanks(1), bankwidth(4*PERI_UNROLL), doublepump, numreadports(3), numwriteports(2))) peri_col[BSIZE * BSIZE];

	int bx = get_group_id(0);
	int tx = get_local_id(0);

	int idx = tx % BSIZE;
	int txg = tx / BSIZE;

	int constant_1 = txg * matrix_dim;
	int constant_2 = (bx + 1) * BSIZE;
	int constant_3 = (bx + 1) * BSIZE * matrix_dim;

	int array_offset_1 = offset + constant_1;
	int array_offset_2 = offset + constant_1 + constant_2;
	int array_offset_3 = offset + constant_1 + constant_3;

	// two block rows are read per iteration
	for (int i = 0; i < BSIZE; i = i + 2)
	{
		dia_row[(i + txg) * BSIZE + idx]  = m[array_offset_1 + idx];
		dia_col[idx * BSIZE + (i + txg)]  = m[array_offset_1 + idx];
		peri_row[idx * BSIZE + (i + txg)] = m[array_offset_2 + idx];
		peri_col[(i + txg) * BSIZE + idx] = m[array_offset_3 + idx];

		array_offset_1 += 2 * matrix_dim;
		array_offset_2 += 2 * matrix_dim;
		array_offset_3 += 2 * matrix_dim;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (tx < BSIZE)
	{ //peri-row
		int idx = tx;
		int peri_row_array_offset = offset + constant_2;
		for(int i = 0; i < BSIZE; i++)
		{
			float sum = 0.0f;
			#pragma unroll PERI_UNROLL
			for (int j = 0; j < i; j++)
			{
				sum += dia_row[i * BSIZE + j] * peri_row[idx * BSIZE + j];
			}
			peri_row[idx * BSIZE + i] -= sum;

			// write-back is done here since it removes one extra read from the peri_row buffer
			// and accesses to external memory are consecutive based on work-group ID anyway
			m[peri_row_array_offset + idx] = peri_row[idx * BSIZE + i];
			peri_row_array_offset += matrix_dim;
		}
	}
	else
	{ //peri-col
		int idx = tx - BSIZE;
		for(int i = 0; i < BSIZE; i++)
		{
			float sum = 0.0f;
			#pragma unroll PERI_UNROLL
			for(int j = 0; j < i; j++)
			{
				sum += dia_col[i * BSIZE + j] * peri_col[idx * BSIZE + j];
			}
			peri_col[idx * BSIZE + i] = (peri_col[idx * BSIZE + i] - sum) / dia_col[i * BSIZE + i];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	int peri_col_array_offset = offset + constant_1 + constant_3;
	// two block rows are written per iteration, disable compiler auto unrolling
	#pragma unroll 1
	for(int i = 0; i < BSIZE; i = i + 2)
	{
		// even though this could also be merged into the compute loop like the other write-back,
		// it was avoided since it would have resulted in accesses that are not consecutive based
		// on work-group ID and lowered performance
		m[peri_col_array_offset + idx] = peri_col[(i + txg) * BSIZE + idx];
		peri_col_array_offset += 2 * matrix_dim;
	}
}
