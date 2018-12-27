/**
 * Copyright (c) 2018 Andre Bannwart Perina and others
 *
 * Adapted from
 * rodinia_3.1/opencl/pathfinder/kernels.cl
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

#define IN_RANGE(x, min, max) ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__attribute__((reqd_work_group_size(4000,1,1)))
__kernel void dynproc_kernel (int iteration,
                              __global int* gpuWall,
                              __global int* gpuSrc,
                              __global int* gpuResults,
                              int cols,
                              int rows,
                              int startStep,
                              int border,
                              int HALO,
                              __local int* prev,
                              __local int* result,
                              __global int* outputBuffer)
{
	int BLOCK_SIZE = get_local_size(0);
	int bx = get_group_id(0);
	int tx = get_local_id(0);

	// Each block finally computes result for a small block
	// after N iterations.
	// it is the non-overlapping small blocks that cover
	// all the input data

	// calculate the small block size.
	int small_block_cols = BLOCK_SIZE - (iteration*HALO*2);

	// calculate the boundary for the block according to
	// the boundary of its small block
	int blkX = (small_block_cols*bx) - border;
	int blkXmax = blkX+BLOCK_SIZE-1;

	// calculate the global thread coordination
	int xidx = blkX+tx;

	// effective range within this block that falls within
	// the valid range of the input data
	// used to rule out computation outside the boundary.
	int validXmin = (blkX < 0) ? -blkX : 0;
	int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;
	
	int W = tx-1;
	int E = tx+1;

	W = (W < validXmin) ? validXmin : W;
	E = (E > validXmax) ? validXmax : E;

	bool isValid = IN_RANGE(tx, validXmin, validXmax);

	if(IN_RANGE(xidx, 0, cols-1))
	{
		prev[tx] = gpuSrc[xidx];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

	bool computed;
	for (int i = 0; i < iteration; i++)
	{
		computed = false;
		
		if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) && isValid )
		{
			computed = true;
			int left = prev[W];
			int up = prev[tx];
			int right = prev[E];
			int shortest = MIN(left, up);
			shortest = MIN(shortest, right);
			
			int index = cols*(startStep+i)+xidx;
			result[tx] = shortest + gpuWall[index];
			
			// ===================================================================
			// add debugging info to the debug output buffer...
			if (tx==11 && i==0)
			{
				// set bufIndex to what value/range of values you want to know.
				int bufIndex = gpuSrc[xidx];
				// dont touch the line below.
				outputBuffer[bufIndex] = 1;
			}
			// ===================================================================
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if(i==iteration-1)
		{
			// we are on the last iteration, and thus don't need to 
			// compute for the next step.
			break;
		}

		if(computed)
		{
			//Assign the computation range
			prev[tx] = result[tx];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// update the global memory
	// after the last iteration, only threads coordinated within the
	// small block perform the calculation and switch on "computed"
	if (computed)
	{
		gpuResults[xidx] = result[tx];
	}
}




