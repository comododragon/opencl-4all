/**
 * Copyright (c) 2018 Andre Bannwart Perina and others
 *
 * Adapted from
 * shoc/src/opencl/level1/scan/scan.cl
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

// This kernel scans the contents of local memory using a work
// inefficient, but highly parallel Kogge-Stone style scan.
// Set exclusive to 1 for an exclusive scan or 0 for an inclusive scan
inline float scanLocalMem(float val, __local float* lmem, int exclusive)
{
    // Set first half of local memory to zero to make room for scanning
    int idx = get_local_id(0);
    lmem[idx] = 0.0f;

    // Set second half to block sums from global memory, but don't go out
    // of bounds
    idx += get_local_size(0);
    lmem[idx] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Now, perform Kogge-Stone scan
    float t;
    for (int i = 1; i < get_local_size(0); i *= 2)
    {
        t = lmem[idx -  i]; barrier(CLK_LOCAL_MEM_FENCE);
        lmem[idx] += t;     barrier(CLK_LOCAL_MEM_FENCE);
    }
    return lmem[idx-exclusive];
}

__attribute__((reqd_work_group_size(256,1,1)))
__kernel void
bottom_scan(__global const float * in,
            __global const float * isums,
            __global float * out,
            const int n,
            __local float * lmem)
{
    __local float s_seed;
    s_seed = 0;

    // Prepare for reading 4-element vectors
    // Assume n is divisible by 4
    __global float4 *in4  = (__global float4*) in;
    __global float4 *out4 = (__global float4*) out;
    int n4 = n / 4; //vector type is 4 wide

    int region_size = n4 / get_num_groups(0);
    int block_start = get_group_id(0) * region_size;
    // Give the last block any extra elements
    int block_stop  = (get_group_id(0) == get_num_groups(0) - 1) ?
        n4 : block_start + region_size;

    // Calculate starting index for this thread/work item
    int i = block_start + get_local_id(0);
    unsigned int window = block_start;

    // Seed the bottom scan with the results from the top scan (i.e. load the per
    // block sums from the previous kernel)
    float seed = isums[get_group_id(0)];

    // Scan multiple elements per thread
    while (window < block_stop) {
        float4 val_4;
        if (i < block_stop) {
            val_4 = in4[i];
        } else {
            val_4.x = 0.0f;
            val_4.y = 0.0f;
            val_4.z = 0.0f;
            val_4.w = 0.0f;
        }

        // Serial scan in registers
        val_4.y += val_4.x;
        val_4.z += val_4.y;
        val_4.w += val_4.z;

        // ExScan sums in local memory
        float res = scanLocalMem(val_4.w, lmem, 1);

        // Update and write out to global memory
        val_4.x += res + seed;
        val_4.y += res + seed;
        val_4.z += res + seed;
        val_4.w += res + seed;

        if (i < block_stop)
        {
            out4[i] = val_4;
        }

        // Next seed will be the last value
        // Last thread puts seed into smem.
        barrier(CLK_LOCAL_MEM_FENCE);
        if (get_local_id(0) == get_local_size(0)-1) {
              s_seed = val_4.w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Broadcast seed to other threads
        seed = s_seed;

        // Advance window
        window += get_local_size(0);
        i += get_local_size(0);
    }
}



