/**
 * Copyright (c) 2018 Andre Bannwart Perina and others
 *
 * Adapted from
 * rodinia_3.1/opencl/cfd/Kernels.cl
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

#define GAMMA (1.4f)

#define NDIM 3

#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)

inline void compute_velocity(float  density, float3 momentum, float3* velocity){
	velocity->x = momentum.x / density;
	velocity->y = momentum.y / density;
	velocity->z = momentum.z / density;
}
	
inline float compute_speed_sqd(float3 velocity){
	return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}

inline float compute_pressure(float density, float density_energy, float speed_sqd){
	return ((float)(GAMMA) - (float)(1.0f))*(density_energy - (float)(0.5f)*density*speed_sqd);
}
inline float compute_speed_of_sound(float density, float pressure){
	//return sqrtf(float(GAMMA)*pressure/density);
	return sqrt((float)(GAMMA)*pressure/density);
}

__attribute__((reqd_work_group_size(192,1,1)))
__kernel void compute_step_factor(__global float* variables, 
							__global float* areas, 
							__global float* step_factors,
							int nelr){
	//const int i = (blockDim.x*blockIdx.x + threadIdx.x);
	const int i = get_global_id(0);
	if( i >= nelr) return;

	float density = variables[i + VAR_DENSITY*nelr];
	float3 momentum;
	momentum.x = variables[i + (VAR_MOMENTUM+0)*nelr];
	momentum.y = variables[i + (VAR_MOMENTUM+1)*nelr];
	momentum.z = variables[i + (VAR_MOMENTUM+2)*nelr];
	
	float density_energy = variables[i + VAR_DENSITY_ENERGY*nelr];
	
	float3 velocity;       compute_velocity(density, momentum, &velocity);
	float speed_sqd      = compute_speed_sqd(velocity);
	//float speed_sqd;
	//compute_speed_sqd(velocity, speed_sqd);
	float pressure       = compute_pressure(density, density_energy, speed_sqd);
	float speed_of_sound = compute_speed_of_sound(density, pressure);

	// dt = float(0.5f) * sqrtf(areas[i]) /  (||v|| + c).... but when we do time stepping, this later would need to be divided by the area, so we just do it all at once
	//step_factors[i] = (float)(0.5f) / (sqrtf(areas[i]) * (sqrtf(speed_sqd) + speed_of_sound));
	step_factors[i] = (float)(0.5f) / (sqrt(areas[i]) * (sqrt(speed_sqd) + speed_of_sound));
}
