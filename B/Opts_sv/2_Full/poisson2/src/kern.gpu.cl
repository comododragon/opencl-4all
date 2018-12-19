///////////// Copyright © 2016 Fabian Oboril. All rights reserved. /////////
//
//   Project     : Altera OpenCL Kernels
//   File        : cdot.cl
//   Description :
//      vector scalar product c = x*y
//
//   Created On: 09.09.2016
//   Created By: Fabian Oboril
////////////////////////////////////////////////////////////////////////////

#define BLOCK_SIZE 64
__attribute__((num_compute_units(4)))
__attribute__((reqd_work_group_size(64,1,1)))
__kernel void cdot_advanced(__global const float16 * restrict A, __global const float16 * restrict B,
                            __global float * result, int size)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group = get_group_id(0);

    int nl= get_local_size(0);

    __local float intermed[BLOCK_SIZE];

    // perform local scalar product on float16
    intermed[lid] = 0;
	intermed[lid] += A[gid].s0 * B[gid].s0;
	intermed[lid] += A[gid].s1 * B[gid].s1;
	intermed[lid] += A[gid].s2 * B[gid].s2;
	intermed[lid] += A[gid].s3 * B[gid].s3;
	intermed[lid] += A[gid].s4 * B[gid].s4;
	intermed[lid] += A[gid].s5 * B[gid].s5;
	intermed[lid] += A[gid].s6 * B[gid].s6;
	intermed[lid] += A[gid].s7 * B[gid].s7;
	intermed[lid] += A[gid].s8 * B[gid].s8;
	intermed[lid] += A[gid].s9 * B[gid].s9;
	intermed[lid] += A[gid].sa * B[gid].sa;
	intermed[lid] += A[gid].sb * B[gid].sb;
	intermed[lid] += A[gid].sc * B[gid].sc;
	intermed[lid] += A[gid].sd * B[gid].sd;
	intermed[lid] += A[gid].se * B[gid].se;
	intermed[lid] += A[gid].sf * B[gid].sf;

    // perform reduction operation
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = (min(BLOCK_SIZE,nl))/2; i > 0; i /= 2) {
        if (lid < i) {
            intermed[lid] += intermed[lid + i];

        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
        barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        result[group] = intermed[lid];
    }
}
