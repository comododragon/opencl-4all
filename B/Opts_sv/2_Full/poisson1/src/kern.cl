/**
 * Copyright (c) 2018 Andre Bannwart Perina and others
 *
 * Adapted from
 * http://cdnc.itec.kit.edu/OpenCLFPGA.php
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