/* ============================================================
//--cambine: kernel funtion of pgain
//--author:	created by Jianbin Fang
//--date:	02/03/2011
============================================================ */

//--9 parameters
/* kernel */
__attribute__((reqd_work_group_size(256,1,1)))
__kernel void pgain_kernel(
			 __global float *p_weight,
			 __global long *p_assign,
			 __global float *p_cost,			 
			 __global float *coord_d,
			 __global float * work_mem_d,			
			 __global int *center_table_d,
			 __global char *switch_membership_d,			
			 __local float *coord_s,
			 int dim,
			 long x,
			 int K){	
	/* block ID and global thread ID */
	const int thread_id = get_global_id(0);
	const int local_id = get_local_id(0);
	/* number of global workitems */
	size_t num = get_global_size(0);
	
	// coordinate mapping of point[x] to shared mem
	coord_s[local_id] = (local_id < dim)? coord_d[local_id * num + x] : 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// cost between this point and point[x]: euclidean distance multiplied by weight
	float x_cost = 0.0;
	for(int i=0; i<dim; i++)
		x_cost += (coord_d[(i*num)+thread_id]-coord_s[i]) * (coord_d[(i*num)+thread_id]-coord_s[i]);
	x_cost = x_cost * p_weight[thread_id];
	
	float current_cost = p_cost[thread_id];

	int base = thread_id*(K+1);	 
	// if computed cost is less then original (it saves), mark it as to reassign	  
	if ( x_cost < current_cost ){
		switch_membership_d[thread_id] = '1';
	    int addr_1 = base + K;
	    work_mem_d[addr_1] = x_cost - current_cost;
	}
	// if computed cost is larger, save the difference
	else {
	    int assign = p_assign[thread_id];
	    int addr_2 = base + center_table_d[assign];
	    work_mem_d[addr_2] += current_cost - x_cost;
	}
}
