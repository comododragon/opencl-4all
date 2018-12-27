/**
 * Copyright (c) 2018 Andre Bannwart Perina and others
 *
 * Adapted from
 * https://github.com/fpga-opencl-benchmarks/rodinia_fpga
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

#define AMB_TEMP (80.0f)
#define SSIZE 4

__attribute__((num_simd_work_items(SSIZE)))
__attribute__((reqd_work_group_size(64,4,1)))
__kernel void hotspotOpt1(__global float* restrict p,
                          __global float* restrict tIn,
                          __global float* restrict tOut,
                                   float           sdc,
                                   int             nx,
                                   int             ny,
                                   int             nz,
                                   float           ce,
                                   float           cw, 
                                   float           cn,
                                   float           cs,
                                   float           ct,
                                   float           cb, 
                                   float           cc)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int c = i + j * nx;
  int xy = nx * ny;

  int W = (i == 0)        ? c : c - 1;
  int E = (i == nx-1)     ? c : c + 1;
  int N = (j == 0)        ? c : c - nx;
  int S = (j == ny-1)     ? c : c + nx;

  float temp1, temp2, temp3;
  temp1 = temp2 = tIn[c];
  temp3 = tIn[c+xy];
  tOut[c] = cc * temp2 + cn * tIn[N] + cs * tIn[S] + ce * tIn[E]
    + cw * tIn[W] + ct * temp3 + cb * temp1 + sdc * p[c] + ct * AMB_TEMP;
  c += xy;
  W += xy;
  E += xy;
  N += xy;
  S += xy;

  for (int k = 1; k < nz-1; ++k) {
      temp1 = temp2;
      temp2 = temp3;
      temp3 = tIn[c+xy];
      tOut[c] = cc * temp2 + cn * tIn[N] + cs * tIn[S] + ce * tIn[E]
        + cw * tIn[W] + ct * temp3 + cb * temp1 + sdc * p[c] + ct * AMB_TEMP;
      c += xy;
      W += xy;
      E += xy;
      N += xy;
      S += xy;
  }
  temp1 = temp2;
  temp2 = temp3;
  tOut[c] = cc * temp2 + cn * tIn[N] + cs * tIn[S] + ce * tIn[E]
    + cw * tIn[W] + ct * temp3 + cb * temp1 + sdc * p[c] + ct * AMB_TEMP;
  return;
}


