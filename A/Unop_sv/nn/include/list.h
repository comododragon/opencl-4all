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

#ifndef LIST_H
#define LIST_H

#include <stdbool.h>

typedef union {
	int d;
	double lf;
	char *s;
} elem_u;

typedef struct elem_t {
	elem_u val;
	struct elem_t *next; 
} elem_t;

typedef struct {
	unsigned int size;
	unsigned int type;
	elem_t *head;
} list_t;

list_t *dlist_create(void);
void dlist_destroy(list_t **list);
void dlist_trim(list_t **list, unsigned int n);
void dlist_pushBack(list_t *list, int val);
void dlist_popFront(list_t *list);
int dlist_front(list_t *list);
int dlist_back(list_t *list);
int dlist_get(list_t *list, unsigned int pos);
void dlist_insert(list_t *list, unsigned int pos, int val);
void dlist_swap(list_t *list, unsigned int pos, int val);
bool dlist_isEmpty(list_t *list);
unsigned int dlist_size(list_t *list);

list_t *lflist_create(void);
void lflist_destroy(list_t **list);
void lflist_trim(list_t **list, unsigned int n);
void lflist_pushBack(list_t *list, double val);
void lflist_popFront(list_t *list);
double lflist_front(list_t *list);
double lflist_back(list_t *list);
double lflist_get(list_t *list, unsigned int pos);
void lflist_insert(list_t *list, unsigned int pos, double val);
void lflist_swap(list_t *list, unsigned int pos, double val);
bool lflist_isEmpty(list_t *list);
unsigned int lflist_size(list_t *list);

list_t *slist_create(void);
void slist_destroy(list_t **list);
void slist_trim(list_t **list, unsigned int n);
void slist_pushBack(list_t *list, char *val);
void slist_popFront(list_t *list);
char *slist_front(list_t *list);
char *slist_back(list_t *list);
char *slist_get(list_t *list, unsigned int pos);
void slist_insert(list_t *list, unsigned int pos, char *val);
void slist_swap(list_t *list, unsigned int pos, char *val);
bool slist_isEmpty(list_t *list);
unsigned int slist_size(list_t *list);

#endif
