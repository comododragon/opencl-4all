/**
 * Copyright (c) 2018 Andre Bannwart Perina and others
 *
 * Adapted from
 * rodinia_3.1/opencl/particlefilter/particle_single.cl
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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/** added this function. was missing in original double version.
 * Takes in a double and returns an integer that approximates to that double
 * @return if the mantissa < .5 => return value < input value; else return value > input value
 */
double dev_round_double(double value) {
    int newValue = (int) (value);
    if (value - newValue < .5f)
        return newValue;
    else
        return newValue++;
}

/********************************
* CALC LIKELIHOOD SUM
* DETERMINES THE LIKELIHOOD SUM BASED ON THE FORMULA: SUM( (IK[IND] - 100)^2 - (IK[IND] - 228)^2)/ 100
* param 1 I 3D matrix
* param 2 current ind array
* param 3 length of ind array
* returns a double representing the sum
********************************/
double calcLikelihoodSum(__global unsigned char * I, __global int * ind, int numOnes, int index){
	double likelihoodSum = 0.0;
	int x;
	for(x = 0; x < numOnes; x++)
		likelihoodSum += (pow((double)(I[ind[index*numOnes + x]] - 100),2) - pow((double)(I[ind[index*numOnes + x]]-228),2))/50.0;
	return likelihoodSum;
}

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

/**
* Generates a normally distributed random number using the Box-Muller transformation
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a double representing random number generated using the Box-Muller algorithm
* @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
*/
double d_randn(__global int * seed, int index){
	//Box-Muller algortihm
	double pi = 3.14159265358979323846;
	double u = d_randu(seed, index);
	double v = d_randu(seed, index);
	double cosine = cos(2*pi*v);
	double rt = -2*log(u);
	return sqrt(rt)*cosine;
}

/*****************************
* OpenCL Likelihood Kernel Function to replace FindIndex
* param1: arrayX
* param2: arrayY
* param2.5: CDF
* param3: ind
* param4: objxy
* param5: likelihood
* param6: I
* param6.5: u
* param6.75: weights
* param7: Nparticles
* param8: countOnes
* param9: max_size
* param10: k
* param11: IszY
* param12: Nfr
*****************************/
__attribute__((reqd_work_group_size(512,1,1)))
__kernel void likelihood_kernel(__global double * arrayX, __global double * arrayY,__global double * xj, __global double * yj, __global int * ind, __global int * objxy, __global double * likelihood, __global unsigned char * I, __global double * weights, const int Nparticles, const int countOnes, const int max_size, int k, const int IszY, const int Nfr, __global int *seed, __global double * partial_sums, __local double* buffer){
	int block_id = get_group_id(0);
	int thread_id = get_local_id(0);
	int i = get_global_id(0);
        size_t THREADS_PER_BLOCK = get_local_size(0);
	int y;
	int indX, indY;
	
	
	if(i < Nparticles){
		arrayX[i] = xj[i]; 
		arrayY[i] = yj[i]; 

		weights[i] = 1 / ((double) (Nparticles)); //Donnie - moved this line from end of find_index_kernel to prevent all weights from being reset before calculating position on final iteration.


		arrayX[i] = arrayX[i] + 1.0 + 5.0*d_randn(seed, i);
		arrayY[i] = arrayY[i] - 2.0 + 2.0*d_randn(seed, i);

	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	if(i < Nparticles)
	{
		for(y = 0; y < countOnes; y++){

			indX = dev_round_double(arrayX[i]) + objxy[y*2 + 1];
			indY = dev_round_double(arrayY[i]) + objxy[y*2];

			ind[i*countOnes + y] = abs(indX*IszY*Nfr + indY*Nfr + k);
			if(ind[i*countOnes + y] >= max_size)
				ind[i*countOnes + y] = 0;
		}
		likelihood[i] = calcLikelihoodSum(I, ind, countOnes, i);

		likelihood[i] = likelihood[i]/countOnes;

		// XXX: original line: weights[i] = weights[i] * exp(likelihood[i]);
		// XXX: I've changed it because the number is huge and implementation differences is notable
		// XXX: with this change, the number is smaller.
		weights[i] = weights[i] * likelihood[i];

	}
	
	buffer[thread_id] = 0.0; // DEBUG!!!!!!!!!!!!!!!!!!!!!!!!
	//buffer[thread_id] = i;
		
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


	if(i < Nparticles){
		buffer[thread_id] = weights[i];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	/* for some reason the get_local_size(0) call was not returning 512. */
	//for(unsigned int s=get_local_size(0)/2; s>0; s>>=1)
	for(unsigned int s=THREADS_PER_BLOCK/2; s>0; s>>=1)
	{
		if(thread_id < s)
		{
			buffer[thread_id] += buffer[thread_id + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if(thread_id == 0)
	{
		partial_sums[block_id] = buffer[0];
	}
	
}//*/
