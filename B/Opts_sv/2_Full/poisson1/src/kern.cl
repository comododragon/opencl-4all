///////////// Copyright © 2016 Fabian Oboril. All rights reserved. /////////
//
//   Project     : Altera OpenCL Kernels
//   File        : vectorAdd.cl
//   Description :
//      vector addition z = alpha*x+y
//
//   Created On: 09.09.2016
//   Created By: Fabian Oboril
////////////////////////////////////////////////////////////////////////////

__kernel void vectorAdd(__global const float16 *x, 
                        __global const float16 *y, 
                        __global float16 *restrict z)
{
    // get index of the work item
    int index = get_global_id(0);

    __local float16 x_local;
    __local float16 y_local;
    __local float16 z_local;

    float alpha = 3.2;

    // load data to local memory
    x_local = x[index];
    y_local = y[index];

    // perform operation locally
    z_local = alpha*x_local + y_local;

    // copy data back
    z[index] = z_local;
}