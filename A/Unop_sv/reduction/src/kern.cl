/**
 * Copyright (c) 2018 Andre Bannwart Perina and others
 *
 * Adapted from
 * shoc/src/opencl/level1/reduction/reduction.cl
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

__attribute__((reqd_work_group_size(256,1,1)))
__kernel void
reduce(__global const FPTYPE *g_idata, __global FPTYPE *g_odata,
       __local FPTYPE* sdata, const unsigned int n)
{
    const unsigned int tid = get_local_id(0);
    unsigned int i = (get_group_id(0)*(get_local_size(0)*2)) + tid;
    const unsigned int gridSize = get_local_size(0)*2*get_num_groups(0);
    const unsigned int blockSize = get_local_size(0);

    sdata[tid] = 0;

    // Reduce multiple elements per thread, strided by grid size
    while (i < n)
    {
        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        i += gridSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
    for (unsigned int s = blockSize / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result back to global memory
    if (tid == 0)
    {
        g_odata[get_group_id(0)] = sdata[0];
    }
}
