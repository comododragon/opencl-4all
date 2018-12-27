/**
 * Copyright (c) 2018 Andre Bannwart Perina and others
 *
 * Adapted from
 * rodinia_3.1/opencl/nn/nearestNeighbor_kernel.cl
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

//#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void NearestNeighbor(__global float2 *d_locations,
							  __global float *d_distances,
							  const int numRecords,
							  const float lat,
							  const float lng) {
	 int globalId = get_global_id(0);
							  
     if (globalId < numRecords) {
         __global float2 *latLong = d_locations+globalId;
    
         __global float *dist=d_distances+globalId;
         *dist = (float)sqrt((lat-latLong->x)*(lat-latLong->x)+(lng-latLong->y)*(lng-latLong->y));
	 }
}
