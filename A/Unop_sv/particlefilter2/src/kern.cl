#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/*****************************
* RANDU
* GENERATES A UNIFORM DISTRIBUTION
* returns a double representing a randomily generated number from a uniform distribution with range [0, 1)
******************************/
double d_randu(__global int * seed, int index)
{

	int M = INT_MAX;
	int A = 1103515245;
	int C = 12345;
	int num = A*seed[index] + C;
	seed[index] = num % M;
	return fabs(seed[index] / ((double) M));
}

/****************************
CDF CALCULATE
CALCULATES CDF
param1 CDF
param2 weights
param3 Nparticles
*****************************/
void cdfCalc(__global double * CDF, __global double * weights, int Nparticles){
	int x;
	CDF[0] = weights[0];
	for(x = 1; x < Nparticles; x++){
		CDF[x] = weights[x] + CDF[x-1];
	}
}

__attribute__((reqd_work_group_size(512,1,1)))
__kernel void normalize_weights_kernel(__global double * weights, int Nparticles, __global double * partial_sums, __global double * CDF, __global double * u, __global int * seed)
{
	int i = get_global_id(0);
	int local_id = get_local_id(0);
	__local double u1;
	__local double sumWeights;

	if(0 == local_id)
		sumWeights = partial_sums[0];

	barrier(CLK_LOCAL_MEM_FENCE);

	if(i < Nparticles) {
		weights[i] = weights[i]/sumWeights;
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if(i == 0) {
		cdfCalc(CDF, weights, Nparticles);
		u[0] = (1/((double)(Nparticles))) * d_randu(seed, i); // do this to allow all threads in all blocks to use the same u1
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if(0 == local_id)
		u1 = u[0];

	barrier(CLK_LOCAL_MEM_FENCE);

	if(i < Nparticles)
	{
		u[i] = u1 + i/((double)(Nparticles));
	}
}
