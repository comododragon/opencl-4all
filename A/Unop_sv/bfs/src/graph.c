#include "graph.h"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#include "list.h"

graph_t *graph_create(void) {
	graph_t *graph = malloc(sizeof(graph_t));
	graph->numVerts = 0;
	graph->numEdges = 0;
	graph->maxDegree = 0;
	graph->adjListLen = 0;
	graph->edgeOffsets = NULL;
	graph->edgeList = NULL;
	graph->edgeCosts = NULL;
	graph->graphType = -1;

	return graph;
}

int graph_getAdjListLen(graph_t *graph) {
	return graph->adjListLen;
}

void graph_destroy(graph_t **graph) {
	if((*graph)->edgeOffsets)
		free((*graph)->edgeOffsets);

	if((*graph)->edgeList)
		free((*graph)->edgeList);

	if((1 == (*graph)->graphType) && (*graph)->edgeCosts)
		free((*graph)->edgeCosts);

	free(*graph);
	*graph = NULL;
}

void graph_generateSimpleKWayGraph(graph_t *graph, unsigned int verts, unsigned int degree) {
	unsigned int i, j, offset = 0, temp;

	if(!(graph->edgeOffsets)) {
		graph->edgeOffsets = malloc((verts + 1) * sizeof(unsigned int));
		graph->edgeList = malloc((verts * (degree + 1)) * sizeof(unsigned int));
	}

	for(i = 0; i < verts; i++) {
		graph->edgeOffsets[i] = offset;

		for(j = 0; j < degree; j++) {
			temp = (i * degree) + (j + 1);
			if(temp < verts) {
				graph->edgeList[offset] = temp;
				offset++;
			}
		}
		if(i) {
			graph->edgeList[offset] = (unsigned int) floor((float) (i - 1) / (float) degree);
			offset++;
		}
	}

	graph->edgeOffsets[verts] = offset;

    graph->adjListLen = offset;
    graph->numEdges = offset / 2;
    graph->numVerts = verts;
    graph->graphType = 0;
    graph->maxDegree = degree + 1;
}

unsigned int *graph_getVertexLengths(graph_t *graph, unsigned int source) {
	int i;
	unsigned int *costs = malloc(graph->numVerts * sizeof(unsigned int));

	for(i = 0; i < graph->numVerts; i++)
		costs[i] = UINT_MAX;

	costs[source] = 0;
	unsigned int nid;
	unsigned int next, offset;
	int n;
	unsigned int vertsVisited = 0;
	list_t *q = dlist_create();
	dlist_pushBack(q, source);

	while(!dlist_isEmpty(q)) {
		n = dlist_front(q);
		vertsVisited++;
		dlist_popFront(q);
		offset = graph->edgeOffsets[n];
		next = graph->edgeOffsets[n + 1];
		while(offset < next) {
			nid = graph->edgeList[offset];
			offset++;
			if(costs[nid] > (costs[n] + 1)) {
				costs[nid] = costs[n] + 1;
				dlist_pushBack(q, nid);
			}
		}
	}

	dlist_destroy(&q);

	return costs;
}
