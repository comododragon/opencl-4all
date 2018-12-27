/**
 * Copyright (c) 2018 Andre Bannwart Perina and others
 *
 * Adapted from
 * rodinia_3.1/opencl/hybridsort/kmeans.cl
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

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

__attribute__((reqd_work_group_size(256,1,1)))
__kernel void
kmeans_kernel_c(__global float  *feature,   
			  __global float  *clusters,
			  __global int    *membership,
			    int     npoints,
				int     nclusters,
				int     nfeatures,
				int		offset,
				int		size
			  ) 
{
	unsigned int point_id = get_global_id(0);
    int index = 0;
    //const unsigned int point_id = get_global_id(0);
		if (point_id < npoints)
		{
			float min_dist=FLT_MAX;
			for (int i=0; i < nclusters; i++) {
				
				float dist = 0;
				float ans  = 0;
				for (int l=0; l<nfeatures; l++){
						ans += (feature[l * npoints + point_id]-clusters[i*nfeatures+l])* 
							   (feature[l * npoints + point_id]-clusters[i*nfeatures+l]);
				}

				dist = ans;
				if (dist < min_dist) {
					min_dist = dist;
					index    = i;
					
				}
			}
		  //printf("%d\n", index);
		  membership[point_id] = index;
		}	
	
	return;
}
