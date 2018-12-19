#define DEFAULT_ORDER 256

__attribute__((reqd_work_group_size(256,1,1)))
__kernel void 
findK(	long height,

		__global int *knodesDLocation,
		__global int *knodesDIndices,
		__global int *knodesDKeys,
		__global bool *knodesDIsLeaf,
		__global int *knodesDNumKeys,

		long knodes_elem,
		__global int *recordsD,

		__global long *currKnodeD,
		__global long *offsetD,
		__global int *keysD, 
		__global int *ansD)
{

	// private thread IDs
	int thid = get_local_id(0);
	int bid = get_group_id(0);

	// processtree levels
	int i;
	for(i = 0; i < height; i++){

		// if value is between the two keys
		if((knodesDKeys[(currKnodeD[bid] * (DEFAULT_ORDER + 1)) + thid]) <= keysD[bid] && (knodesDKeys[(currKnodeD[bid] * (DEFAULT_ORDER + 1)) + thid+1] > keysD[bid])){
			// this conditional statement is inserted to avoid crush due to but in original code
			// "offset[bid]" calculated below that addresses knodes[] in the next iteration goes outside of its bounds cause segmentation fault
			// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
			if(knodesDIndices[(offsetD[bid] * (DEFAULT_ORDER + 1)) + thid] < knodes_elem){
				offsetD[bid] = knodesDIndices[(offsetD[bid] * (DEFAULT_ORDER + 1)) + thid];
			}
		}
		//__syncthreads();
		barrier(CLK_LOCAL_MEM_FENCE);
		// set for next tree level
		if(thid==0){
			currKnodeD[bid] = offsetD[bid];
		}
		//__syncthreads();
		barrier(CLK_LOCAL_MEM_FENCE);

	}

	//At this point, we have a candidate leaf node which may contain
	//the target record.  Check each key to hopefully find the record
	if(knodesDKeys[(currKnodeD[bid] * (DEFAULT_ORDER + 1)) + thid] == keysD[bid]){
		ansD[bid] = recordsD[knodesDIndices[(currKnodeD[bid] * (DEFAULT_ORDER + 1)) + thid]];
	}

}
