/**
 * Copyright (c) 2018 Andre Bannwart Perina and others
 *
 * Adapted from
 * https://github.com/fahadmuslim/Bitonic-Sorting
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

#define LOCAL_SIZE_LIMIT 64

inline void ComparatorLocal(
	__local uint *keyA,
	__local uint *valA,
	__local uint *keyB,
	__local uint *valB,
	uint arrowDir
){
	if((*keyA > *keyB) == arrowDir) {
		uint t;
		t = *keyA; *keyA = *keyB; *keyB = t;
		t = *valA; *valA = *valB; *valB = t;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Bitonic sort kernel for large arrays (not fitting into local memory)
////////////////////////////////////////////////////////////////////////////////
//Bottom-level bitonic sort
//Almost the same as bitonicSortLocal with the only exception
//of even / odd subarrays (of LOCAL_SIZE_LIMIT points) being
//sorted in opposite directions
__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE_LIMIT / 2, 1, 1)))
void bitonicSortLocal1(
	__global uint *d_DstKey,
	__global uint *d_DstVal,
	__global uint *d_SrcKey,
	__global uint *d_SrcVal
){
	__attribute__((numbanks(LOCAL_SIZE_LIMIT), bankwidth(4)))
	__local uint l_key[LOCAL_SIZE_LIMIT];
	__attribute__((numbanks(LOCAL_SIZE_LIMIT), bankwidth(4)))
	__local uint l_val[LOCAL_SIZE_LIMIT];

	if(get_local_id(0) == 0) {
		for(int i = 0; i < LOCAL_SIZE_LIMIT; i++) {
			l_key[i] = d_SrcKey[get_group_id(0) * LOCAL_SIZE_LIMIT + i];
			l_val[i] = d_SrcVal[get_group_id(0) * LOCAL_SIZE_LIMIT + i];
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////
	barrier(CLK_LOCAL_MEM_FENCE);

	uint comparatorI = get_global_id(0) & ((LOCAL_SIZE_LIMIT / 2) - 1);

#pragma unroll
	for(uint size = 2; size < LOCAL_SIZE_LIMIT; size <<= 1) {
		//Bitonic merge
		uint dir = (comparatorI & (size / 2)) != 0;
		for(uint stride = size / 2; stride > 0; stride >>= 1) {
			barrier(CLK_LOCAL_MEM_FENCE);
			uint pos = 2 * get_local_id(0) - (get_local_id(0) & (stride - 1));
			ComparatorLocal(
				&l_key[pos +	  0], &l_val[pos +	  0],
				&l_key[pos + stride], &l_val[pos + stride],
				dir
			);
		}
	}

	//Odd / even arrays of LOCAL_SIZE_LIMIT elements
	//sorted in opposite directions
	{
		uint dir = (get_group_id(0) & 1);
#pragma unroll
		for_local1_last: for(uint stride = LOCAL_SIZE_LIMIT / 2; stride > 0; stride >>= 1) {
			barrier(CLK_LOCAL_MEM_FENCE);
			uint pos = 2 * get_local_id(0) - (get_local_id(0) & (stride - 1));
			ComparatorLocal(
				&l_key[pos +	  0], &l_val[pos +	  0],
				&l_key[pos + stride], &l_val[pos + stride],
			   dir
			);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if(get_local_id(0) == 0) {
		for(int i = 0; i < LOCAL_SIZE_LIMIT; i++) {
			d_DstKey[get_group_id(0) * LOCAL_SIZE_LIMIT + i] = l_key[i];
			d_DstVal[get_group_id(0) * LOCAL_SIZE_LIMIT + i] = l_val[i];
		}
	}

///////////////////////////////////////////////////////////////////////////////////////
}
