#include <float.h>
#include <stdlib.h>
#include <time.h>

#include "list.h"

#define DOMAIN_EDGE 20.0

float distance(cl_float3 *position, int i, int j) {
	cl_float3 iPos = position[i];
	cl_float3 jPos = position[j];
	float delX = iPos.x - jPos.x;
	float delY = iPos.y - jPos.y;
	float delZ = iPos.z - jPos.z;

	return delX * delX + delY * delY + delZ * delZ;
}

void insertInOrder(list_t *currDist, list_t *currList, int j, float distIJ, int maxNeighbors) {
	int i;
	float currMax = flist_back(currDist);

	if(distIJ > currMax)
		return;

	for(i = 0; i < flist_size(currDist); i++) {
		if(distIJ < flist_get(currDist, i)) {
			flist_insert(currDist, i, distIJ);
			dlist_insert(currList, i, j);

			flist_trim(&currDist, maxNeighbors);
			dlist_trim(&currList, maxNeighbors);

			return;
		}
	}
}

int populateNeighborList(list_t *currDist, list_t *currList, int i, int nAtom, int *neighborList, float cutsq) {
	int idx;
	int validPairs = 0;

	for(idx = 0; idx < dlist_size(currList); idx++) {
		neighborList[(idx * nAtom) + i] = dlist_get(currList, idx);

		if(flist_get(currDist, idx) < cutsq)
			validPairs++;
	}

	return validPairs;
}

int buildNeighborList(int nAtom, cl_float3 *position, int *neighborList, int maxNeighbors, float cutsq) {
	int i, j;
	int totalPairs = 0;

	for(i = 0; i < nAtom; i++) {
		list_t *currList = dlist_create();
		list_t *currDist = flist_create();
		for(j = 0; j < maxNeighbors; j++) {
			dlist_pushBack(currList, -1);
			flist_pushBack(currDist, DBL_MAX);
		}

		for(j = 0; j < nAtom; j++) {
			if(i == j)
				continue;

			float distIJ = distance(position, i, j);
			insertInOrder(currDist, currList, j, distIJ, maxNeighbors);
		}

		totalPairs += populateNeighborList(currDist, currList, i, nAtom, neighborList, cutsq);

		dlist_destroy(&currList);
		flist_destroy(&currDist);
	}

	return totalPairs;
}

#define PREAMBLE(force, forceSz, forceC, forceCSz,\
	position, positionSz, maxNeighbors, neighborList, neighborListSz, cutsq, lj1, lj2, nAtom) {\
	int _i = 0;\
\
	srand(0);\
\
	for(_i = 0; _i < nAtom; _i++) {\
		position[_i].x = (rand() / (float) RAND_MAX) * DOMAIN_EDGE;\
		position[_i].y = (rand() / (float) RAND_MAX) * DOMAIN_EDGE;\
		position[_i].z = (rand() / (float) RAND_MAX) * DOMAIN_EDGE;\
		/* We will test error margin, not the force value itself */\
		forceC[_i].x = 0;\
		forceC[_i].y = 0;\
		forceC[_i].z = 0;\
	}\
\
	buildNeighborList(nAtom, position, neighborList, maxNeighbors, cutsq);\
}

#define POSTAMBLE(force, forceSz, forceC, forceCSz,\
	position, positionSz, maxNeighbors, neighborList, neighborListSz, cutsq, lj1, lj2, nAtom) {\
	int _i, _j;\
\
	for(_i = 0; _i < nAtom; _i++) {\
		cl_float3 _iPos = position[_i];\
		cl_float3 _f = {0, 0, 0};\
\
		for(_j = 0; _j < maxNeighbors; _j++) {\
			int _jIdx = neighborList[_j * nAtom + _i];\
			cl_float3 _jPos = position[_jIdx];\
\
			float _delX = _iPos.x - _jPos.x;\
			float _delY = _iPos.y - _jPos.y;\
			float _delZ = _iPos.z - _jPos.z;\
\
			float _r2Inv = _delX * _delX + _delY * _delY + _delZ * _delZ;\
\
			if(_r2Inv < cutsq) {\
				_r2Inv = 1.0 / _r2Inv;\
				float _r6Inv = _r2Inv * _r2Inv * _r2Inv;\
				float _force = _r2Inv * _r6Inv * (lj1 * _r6Inv - lj2);\
				_f.x += _delX * _force;\
				_f.y += _delY * _force;\
				_f.z += _delZ * _force;\
			}\
		}\
\
		/* Substituting values for validation */\
		/* i.e. force[_i].x is now the error margin */\
		force[_i].x = (force[_i].x - _f.x) / force[_i].x;\
		force[_i].y = (force[_i].y - _f.y) / force[_i].y;\
		force[_i].z = (force[_i].z - _f.z) / force[_i].z;\
	}\
}
