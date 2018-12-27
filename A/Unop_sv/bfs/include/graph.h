/**
 * Copyright (c) 2018 Andre Bannwart Perina
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
