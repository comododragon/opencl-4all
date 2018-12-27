/**
 * Copyright (c) 2018 Andre Bannwart Perina and others
 *
 * Adapted from
 * rodinia_3.1/opencl/hybridsort/bucketsort_kernels.cl
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

#define DIVISIONS               (1 << 10)
#define LOG_DIVISIONS	(10)
#define BUCKET_WARP_LOG_SIZE	(5)
#define BUCKET_WARP_N			(1)
#ifdef BUCKET_WG_SIZE_1
#define BUCKET_THREAD_N BUCKET_WG_SIZE_1
#else
#define BUCKET_THREAD_N			(BUCKET_WARP_N << BUCKET_WARP_LOG_SIZE)
#endif
#define BUCKET_BLOCK_MEMORY		(DIVISIONS * BUCKET_WARP_N)
#define BUCKET_BAND				(128)

__attribute__((reqd_work_group_size(32,1,1)))
__kernel void
bucketsort(global float *input, global int *indice, __global float *output, const int size, global uint *d_prefixoffsets,
		   global uint *l_offsets)
{
	volatile __local unsigned int s_offset[BUCKET_BLOCK_MEMORY];
    
	int prefixBase = get_group_id(0) * BUCKET_BLOCK_MEMORY;
    const int warpBase = (get_local_id(0) >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = get_global_size(0);
    
	for (int i = get_local_id(0); i < BUCKET_BLOCK_MEMORY; i += get_local_size(0)){
		s_offset[i] = l_offsets[i & (DIVISIONS - 1)] + d_prefixoffsets[prefixBase + i];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	for (int tid = get_global_id(0); tid < size; tid += numThreads) {
       
		float elem = input[tid];
		int id = indice[tid];
		output[s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >> LOG_DIVISIONS)] = elem;
        int test = s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >> LOG_DIVISIONS);
//        if(test == 2) {
//            printf("EDLLAWD %f", elem);
//        }
	}
}

