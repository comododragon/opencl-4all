// TODO: Add licence header, this h and its c was adapted from Aditya Sarwade

#ifndef GRAPH_H
#define GRAPH_H

#define MAX_LINE_LENGTH 500000

typedef struct {
	unsigned int numVerts;
	unsigned int numEdges;
	unsigned int adjListLen;
	unsigned int *edgeOffsets;
	unsigned int *edgeList;
	unsigned int *edgeCosts;
	unsigned int maxDegree;
	int graphType;
} graph_t;

graph_t *graph_create(void);
int graph_getAdjListLen(graph_t *graph);
void graph_destroy(graph_t **graph);
void graph_generateSimpleKWayGraph(graph_t *graph, unsigned int verts, unsigned int degree);
unsigned int *graph_getVertexLengths(graph_t *graph, unsigned int source);

#endif
