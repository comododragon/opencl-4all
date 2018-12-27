/**
 * Copyright (c) 2018 Andre Bannwart Perina and others
 *
 * Adapted from
 * shoc/src/opencl/level1/spmv/spmv.cl
 * Different licensing may apply, please check SHOC documentation.
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

#define FPTYPE float

// ****************************************************************************
// Function: spmv_csr_vector_kernel
//
// Purpose:
//   Computes sparse matrix - vector multiplication on the GPU using
//   the CSR data storage format, using a warp per row of the sparse
//   matrix; based on Bell (SC09) and Baskaran (IBM Tech Report)
//
// Arguments:
//   val: array holding the non-zero values for the matrix
//   vec: dense vector for multiplication
//   cols: array of column indices for each element of the sparse matrix
//   rowDelimiters: array of size dim+1 holding indices to rows of the matrix
//                  last element is the index one past the last
//                  element of the matrix
//   dim: number of rows in the matrix
//   vecWidth: preferred simd width to use
//   out: output - result from the spmv calculation
//
// Returns:  nothing
//           out indirectly through a pointer
//
// Programmer: Lukasz Wesolowski
// Creation: June 28, 2010
//
// Modifications:
//
// ****************************************************************************
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void
spmv_csr_vector_kernel(__global const FPTYPE * restrict val,
                       __global const FPTYPE * restrict vec,
                       __global const int * restrict cols,
                       __global const int * restrict rowDelimiters,
                       const int dim, const int vecWidth, __global FPTYPE * restrict out)
{
    // Thread ID in block
    int t = get_local_id(0);
    // Thread ID within warp
    int id = t & (vecWidth-1);
    // One row per warp
    int vecsPerBlock = get_local_size(0) / vecWidth;
    int myRow = (get_group_id(0) * vecsPerBlock) + (t / vecWidth);

    __local volatile FPTYPE partialSums[128];
    partialSums[t] = 0;

    if (myRow < dim)
    {
        int vecStart = rowDelimiters[myRow];
        int vecEnd = rowDelimiters[myRow+1];
        FPTYPE mySum = 0;
        for (int j= vecStart + id; j < vecEnd;
             j+=vecWidth)
        {
            int col = cols[j];
            mySum += val[j] * vec[col];
        }

        partialSums[t] = mySum;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Reduce partial sums
	int bar = vecWidth / 2;
	while(bar > 0)
	{
	    if (id < bar) partialSums[t] += partialSums[t+ bar];
	    barrier(CLK_LOCAL_MEM_FENCE);
	    bar = bar / 2;
	}

        // Write result
        if (id == 0)
        {
            out[myRow] = partialSums[t];
        }
    }
}
