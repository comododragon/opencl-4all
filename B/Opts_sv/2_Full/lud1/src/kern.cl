#define DIA_UNROLL 4
#define BSIZE 96

__attribute__((reqd_work_group_size(BSIZE,1,1)))
__kernel void lud_diagonal(__global float* restrict m, 
                                    int             matrix_dim,
                                    int             offset)
{ 
	int tx = get_local_id(0);
	__local float __attribute__((memory, numbanks(1), bankwidth(4*DIA_UNROLL), doublepump, numreadports(3), numwriteports(1))) shadow_row[BSIZE * BSIZE];
	__local float __attribute__((memory, numbanks(1), bankwidth(4*DIA_UNROLL), doublepump, numreadports(3), numwriteports(1))) shadow_col[BSIZE * BSIZE];

	int array_offset = offset;
	for(int i = 0; i < BSIZE; i++)
	{
		shadow_row[i * BSIZE + tx] = m[array_offset + tx];
		shadow_col[tx * BSIZE + i] = m[array_offset + tx];
		array_offset += matrix_dim;
	}
  
	barrier(CLK_LOCAL_MEM_FENCE);

	array_offset = offset + matrix_dim;
	for(int i = 0; i < BSIZE - 1; i++)
	{
		if (tx > i)
		{
			float sum = 0.0f;
			#pragma unroll DIA_UNROLL
			for(int j = 0; j < i; j++)
			{
				sum += shadow_row[tx * BSIZE + j] * shadow_col[i * BSIZE + j];
			}
			shadow_row[tx * BSIZE + i] = (shadow_row[tx * BSIZE + i] - sum) / shadow_col[i * BSIZE + i];
			//shadow_col[i * BSIZE + tx] = shadow_row[tx * BSIZE + i];		// commented out since it is not actually required and output is correct either way
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if (tx > i)
		{
			float sum = 0.0f;
			#pragma unroll DIA_UNROLL
			for(int j = 0; j < i + 1; j++)
			{
				sum += shadow_row[(i + 1) * BSIZE + j] * shadow_col[tx * BSIZE + j];
			}
			shadow_row[(i + 1) * BSIZE + tx] -= sum;
			shadow_col[tx * BSIZE + (i + 1)] = shadow_row[(i + 1) * BSIZE + tx];
		}
		m[array_offset + tx] = shadow_row[(i + 1) * BSIZE + tx];
		array_offset += matrix_dim;
	}
}
