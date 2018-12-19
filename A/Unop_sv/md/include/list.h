#ifndef LIST_H
#define LIST_H

#include <stdbool.h>

typedef union {
	int d;
	float lf;
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

list_t *flist_create(void);
void flist_destroy(list_t **list);
void flist_trim(list_t **list, unsigned int n);
void flist_pushBack(list_t *list, float val);
void flist_popFront(list_t *list);
float flist_front(list_t *list);
float flist_back(list_t *list);
float flist_get(list_t *list, unsigned int pos);
void flist_insert(list_t *list, unsigned int pos, float val);
void flist_swap(list_t *list, unsigned int pos, float val);
bool flist_isEmpty(list_t *list);
unsigned int flist_size(list_t *list);

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
