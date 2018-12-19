#include "list.h"

#define NUM_RECORDS 42764
#define RESULTS_COUNT 5
#define LAT 30
#define LNG 90
#define FILELIST_FILENAME "filelist"
#define OUTPUT_FILENAME "out"
#define REC_LEN 49

FILE *gFList = NULL;
FILE *gFp = NULL;
FILE *outFp = NULL;
list_t *gRecStringList = NULL;

#define PREAMBLE(d_locations, d_locationsSz, d_distances, d_distancesSz, numRecords, lat, lng) {\
	int _i, _j;\
	char _dbName[64];\
	int _recNum = 0;\
\
	gRecStringList = slist_create();\
\
	gFList = fopen(FILELIST_FILENAME, "r");\
	ASSERT_CALL(gFList, rv = EXIT_FAILURE);\
\
	_i = 0;\
	while(!feof(gFList)) {\
		ASSERT_CALL(1 == fscanf(gFList, "%s\n", _dbName), rv = EXIT_FAILURE);\
\
		gFp = fopen(_dbName, "r");\
		ASSERT_CALL(gFp, rv = EXIT_FAILURE);\
\
		for(; _i < NUM_RECORDS; _i++) {\
			char _recString[REC_LEN];\
			char _subStr[6];\
\
			fgets(_recString, 49, gFp);\
			fgetc(gFp);\
\
			if(feof(gFp))\
				break;\
\
			for(_j = 0; _j < 5; _j++)\
				_subStr[_j] = *(_recString + _j + 28);\
			_subStr[5] = '\0';\
			d_locations[_i].x = atof(_subStr);\
\
			for(_j = 0; _j < 5; _j++)\
				_subStr[_j] = *(_recString + _j + 33);\
			_subStr[5] = '\0';\
			d_locations[_i].y = atof(_subStr);\
\
			slist_pushBack(gRecStringList, _recString);\
		}\
\
		fclose(gFp);\
		gFp = NULL;\
	}\
\
	fclose(gFList);\
	gFList = NULL;\
\
	lat = LAT;\
	lng = LNG;\
}

#define POSTAMBLE(d_locations, d_locationsSz, d_distances, d_distancesSz, numRecords, lat, lng) {\
	int _i, _j;\
	float _val;\
	int _minLoc;\
	char *_tmpRecString;\
	char *_tmpRecStringAllocd;\
	float _tempDist;\
\
	for(_i = 0; _i < RESULTS_COUNT; _i++) {\
		_minLoc = _i;\
\
		for(_j = _i; _j < NUM_RECORDS; _j++) {\
			_val = d_distances[_j];\
\
			if(_val < d_distances[_minLoc])\
				_minLoc = _j;\
		}\
\
		_tmpRecString = slist_get(gRecStringList, _i);\
		_tmpRecStringAllocd = malloc(strlen(_tmpRecString) + 1);\
		strcpy(_tmpRecStringAllocd, _tmpRecString);\
		slist_swap(gRecStringList, _i, slist_get(gRecStringList, _minLoc));\
		slist_swap(gRecStringList, _minLoc, _tmpRecStringAllocd);\
		free(_tmpRecStringAllocd);\
\
		_tempDist = d_distances[_i];\
		d_distances[_i] = d_distances[_minLoc];\
		d_distances[_minLoc] = _tempDist;\
\
	}\
\
	outFp = fopen(OUTPUT_FILENAME, "w");\
\
	for(_i = 0; _i < RESULTS_COUNT; _i++)\
		fprintf(outFp, "%s %f\n", slist_get(gRecStringList, _i), d_distances[_i]);\
\
	fclose(outFp);\
	outFp = NULL;\
}

#define CLEANUP(d_locations, d_locationsSz, d_distances, d_distancesSz, numRecords, lat, lng) {\
	if(gFList)\
		fclose(gFList);\
\
	if(gFp)\
		fclose(gFp);\
\
	if(outFp)\
		fclose(outFp);\
\
	if(gRecStringList)\
		slist_destroy(&gRecStringList);\
}
