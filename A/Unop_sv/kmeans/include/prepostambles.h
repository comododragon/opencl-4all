#define NCLUSTERS 5
#define THRESHOLD 0.001

float *gTmpBuf = NULL;
float **gFeatures = NULL;
float **gClusterCentres = NULL;
int *gMembershipOrig = NULL;
float **gClusters = NULL;
int *gInitial = NULL;
int gInitialPoints;
int *gNewCentresLen = NULL;
float **gNewCentres = NULL;

#define PREAMBLE(feature, featureSz, clusters, clustersSz, membership, membershipSz, npoints, nclusters, nfeatures, offset, size) {\
	int _i, _j, _k;\
	int _temp;\
	FILE *_inFile;\
	char _line[1024];\
\
	_inFile = fopen("kmeans30000", "r");\
	ASSERT_CALL(_inFile, rv = EXIT_FAILURE);\
\
	gTmpBuf = malloc(npoints * nfeatures * sizeof(float));\
	gFeatures = malloc(npoints * sizeof(float *));\
	gFeatures[0] = malloc(npoints * nfeatures * sizeof(float));\
\
	for(_i = 1; _i < npoints; _i++)\
		gFeatures[_i] = gFeatures[_i - 1] + nfeatures;\
\
	_i = 0;\
	while(fgets(_line, 1024, _inFile)) {\
		if(!strtok(_line, " \t\n"))\
			continue;\
		for(_j = 0; _j < nfeatures; _j++) {\
			gTmpBuf[_i] = atof(strtok(NULL, " ,\t\n"));\
			_i++;\
		}\
	}\
\
	fclose(_inFile);\
\
	ASSERT_CALL(npoints >= NCLUSTERS, rv = EXIT_FAILURE);\
\
	srand(7);\
	memcpy(gFeatures[0], gTmpBuf, npoints * nfeatures * sizeof(float));\
	free(gTmpBuf);\
	gTmpBuf = NULL;\
\
	gClusterCentres = NULL;\
\
	/* This is the kmeans_swap kernel translated to plain C */\
	for(_i = 0; _i < npoints; _i++) {\
		for(_j = 0; _j < nfeatures; _j++)\
			feature[_j * npoints + _i] = gFeatures[0][_i * nfeatures + _j];\
	}\
\
	gMembershipOrig = malloc(npoints * sizeof(int));\
\
	gClusters = malloc(nclusters * sizeof(float *));\
	gClusters[0] = malloc(nclusters * nfeatures * sizeof(float));\
	for(_i = 1; _i < nclusters; _i++)\
		gClusters[_i] = gClusters[_i - 1] + nfeatures;\
\
	gInitial = malloc(npoints * sizeof(int));\
	for(_i = 1; _i < npoints; _i++)\
		gInitial[_i] = _i;\
	gInitialPoints = npoints;\
\
	_k = 0;\
	for(_i = 0; _i < nclusters && gInitialPoints >= 0; _i++) {\
		for(_j = 0; _j < nfeatures; _j++)\
			gClusters[_i][_j] = gFeatures[gInitial[_k]][_j];\
\
		_temp = gInitial[_k];\
		gInitial[_k] = gInitial[gInitialPoints - 1];\
		gInitial[gInitialPoints - 1] = _temp;\
		gInitialPoints--;\
		_k++;\
	}\
\
	for(_i = 0; _i < npoints; _i++)\
		gMembershipOrig[_i] = -1;\
\
	gNewCentresLen = calloc(nclusters, sizeof(int));\
	gNewCentres = malloc(nclusters * sizeof(float *));\
	gNewCentres[0] = calloc(nclusters * nfeatures, sizeof(float));\
	for(_i = 1; _i < nclusters; _i++)\
		gNewCentres[_i] = gNewCentres[_i - 1] + nfeatures;\
}

#define LOOPPREAMBLE(feature, featureSz, clusters, clustersSz, membership, membershipSz, npoints, nclusters, nfeatures, offset, size, loopFlag) {\
	memcpy(clusters, gClusters[0], nclusters * nfeatures * sizeof(float));\
}

#define LOOPPOSTAMBLE(feature, featureSz, clusters, clustersSz, membership, membershipSz, npoints, nclusters, nfeatures, offset, size, loopFlag) {\
	int _i, _j;\
	int _delta = 0;\
\
	for(_i = 0; _i < npoints; _i++) {\
		int _clusterId = membership[_i];\
		gNewCentresLen[_clusterId]++;\
\
		if(gMembershipOrig[_i] != membership[_i]) {\
			_delta++;\
			gMembershipOrig[_i] = membership[_i];\
		}\
\
		for(_j = 0; _j < nfeatures; _j++)\
			gNewCentres[_clusterId][_j] += gFeatures[_i][_j];\
	}\
\
	for(_i = 0; _i < nclusters; _i++) {\
		for(_j = 0; _j < nfeatures; _j++) {\
			if(gNewCentresLen[_i] > 0)\
				gClusters[_i][_j] = gNewCentres[_i][_j] / gNewCentresLen[_i];\
			gNewCentres[_i][_j] = 0;\
		}\
		gNewCentresLen[_i] = 0;\
	}\
\
	loopFlag = (_delta > THRESHOLD) && (i < 500);\
}

#define CLEANUP(feature, featureSz, clusters, clustersSz, membership, membershipSz, npoints, nclusters, nfeatures, offset, size) {\
	if(gNewCentres) {\
		free(gNewCentres[0]);\
		free(gNewCentres);\
	}\
\
	if(gNewCentresLen)\
		free(gNewCentresLen);\
\
	if(gInitial)\
		free(gInitial);\
\
	if(gClusters) {\
		free(gClusters[0]);\
		free(gClusters);\
	}\
\
	if(gClusterCentres) {\
		free(gClusterCentres[0]);\
		free(gClusterCentres);\
	}\
\
	if(gMembershipOrig)\
		free(gMembershipOrig);\
\
	if(gFeatures) {\
		free(gFeatures[0]);\
		free(gFeatures);\
	}\
\
	if(gTmpBuf)\
		free(gTmpBuf);\
}
