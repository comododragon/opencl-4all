/**
 * Copyright (c) 2018 Andre Bannwart Perina and others
 *
 * Adapted from
 * rodinia_3.1/opencl/hybridsort/mergesort.cl
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

#define DIVISIONS (1024)
float4 sortElem(float4 r) {
	float4 nr;
    
	nr.x = (r.x > r.y) ? r.y : r.x;
	nr.y = (r.y > r.x) ? r.y : r.x;
	nr.z = (r.z > r.w) ? r.w : r.z;
	nr.w = (r.w > r.z) ? r.w : r.z;
    
	r.x = (nr.x > nr.z) ? nr.z : nr.x;
	r.y = (nr.y > nr.w) ? nr.w : nr.y;
	r.z = (nr.z > nr.x) ? nr.z : nr.x;
	r.w = (nr.w > nr.y) ? nr.w : nr.y;
    
	nr.x = r.x;
	nr.y = (r.y > r.z) ? r.z : r.y;
	nr.z = (r.z > r.y) ? r.z : r.y;
	nr.w = r.w;
	return nr;
}

float4 getLowest(float4 a, float4 b)
{
	a.x = (a.x < b.w) ? a.x : b.w;
	a.y = (a.y < b.z) ? a.y : b.z;
	a.z = (a.z < b.y) ? a.z : b.y;
	a.w = (a.w < b.x) ? a.w : b.x;
	return a;
}

float4 getHighest(float4 a, float4 b)
{
	b.x = (a.w >= b.x) ? a.w : b.x;
	b.y = (a.z >= b.y) ? a.z : b.y;
	b.z = (a.y >= b.z) ? a.y : b.z;
	b.w = (a.x >= b.w) ? a.x : b.w;
	return b;
}

__attribute__((reqd_work_group_size(208,1,1)))
__kernel void
mergeSortPass(__global float4 *input, __global float4 *result,const int nrElems,int threadsPerDiv, __global int *constStartAddr)
{

	int gid = get_global_id(0);
	// The division to work on
	int division = gid / threadsPerDiv;
	if(division >= DIVISIONS) return;
	// The block within the division
	int int_gid = gid - division * threadsPerDiv;
	int Astart = constStartAddr[division] + int_gid * nrElems;
    
	int Bstart = Astart + nrElems/2;
	global float4 *resStart;
    resStart= &(result[Astart]);
    
	if(Astart >= constStartAddr[division + 1])
		return;
	if(Bstart >= constStartAddr[division + 1]){
		for(int i=0; i<(constStartAddr[division + 1] - Astart); i++)
		{
			resStart[i] = input[Astart + i];
		}
		return;
	}
    
	int aidx = 0;
	int bidx = 0;
	int outidx = 0;
	float4 a, b;
	a = input[Astart + aidx];
	b = input[Bstart + bidx];
	
	while(true)//aidx < nrElems/2)// || (bidx < nrElems/2  && (Bstart + bidx < constEndAddr[division])))
	{
		/**
		 * For some reason, it's faster to do the texture fetches here than
		 * after the merge
		 */
		float4 nextA = input[Astart + aidx + 1];
		float4 nextB = input[Bstart + bidx + 1];
        
		float4 na = getLowest(a,b);
		float4 nb = getHighest(a,b);
		a = sortElem(na);
		b = sortElem(nb);
		// Now, a contains the lowest four elements, sorted
		resStart[outidx++] = a;
        
		bool elemsLeftInA;
		bool elemsLeftInB;
        
		elemsLeftInA = (aidx + 1 < nrElems/2); // Astart + aidx + 1 is allways less than division border
		elemsLeftInB = (bidx + 1 < nrElems/2) && (Bstart + bidx + 1 < constStartAddr[division + 1]);
        
		if(elemsLeftInA){
			if(elemsLeftInB){
				if(nextA.x < nextB.x) { aidx += 1; a = nextA; }
				else { bidx += 1;  a = nextB; }
			}
			else {
				aidx += 1; a = nextA;
			}
		}
		else {
			if(elemsLeftInB){
				bidx += 1;  a = nextB;
			}
			else {
				break;
			}
		}
        
	}
	resStart[outidx++] = b;
}
