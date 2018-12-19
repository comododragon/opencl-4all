#include <limits.h>

#include "graph.h"

graph_t *g = NULL;

#define PREAMBLE(levels, levelsSz, levelsC, levelsCSz, edgeArray, edgeArraySz, edgeArrayAux, edgeArrayAuxSz, W_SZ, CHUNK_SZ, numVertices, curr, flag) {\
	int _i;\
\
	levels[0] = 0;\
	for(_i = 1; _i < levelsSz; _i++)\
		levels[_i] = UINT_MAX;\
\
	g = graph_create();\
	graph_generateSimpleKWayGraph(g, levelsCSz, 2);\
\
	unsigned int *costs = graph_getVertexLengths(g, 0);\
	for(_i = 0; _i < levelsCSz; _i++)\
		levelsC[_i] = costs[_i];\
	free(costs);\
\
	for(_i = 0; _i < edgeArraySz; _i++)\
		edgeArray[_i] = g->edgeOffsets[_i];\
\
	for(_i = 0; _i < edgeArrayAuxSz; _i++)\
		edgeArrayAux[_i] = g->edgeList[_i];\
\
	numVertices = g->numVerts;\
}


#define LOOPPREAMBLE(levels, levelsSz, levelsC, levelsCSz, edgeArray, edgeArraySz, edgeArrayAux, edgeArrayAuxSz, W_SZ, CHUNK_SZ, numVertices, curr, flag, loopFlag) {\
	flag = false;\
}


#define LOOPPOSTAMBLE(levels, levelsSz, levelsC, levelsCSz, edgeArray, edgeArraySz, edgeArrayAux, edgeArrayAuxSz, W_SZ, CHUNK_SZ, numVertices, curr, flag, loopFlag) {\
	loopFlag = flag;\
	curr++;\
}


#define CLEANUP(levels, levelsSz, levelsC, levelsCSz, edgeArray, edgeArraySz, edgeArrayAux, edgeArrayAuxSz, W_SZ, CHUNK_SZ, numVertices, curr, flag) {\
	if(g)\
		graph_destroy(&g);\
}
