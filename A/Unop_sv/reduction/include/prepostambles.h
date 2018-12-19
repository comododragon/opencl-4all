#define PREAMBLE(g_idata, g_idataSz, g_odata, g_odataSz, g_odataC, g_odataCSz, n) {\
	int _i;\
\
	for(_i = 0; _i < g_idataSz; _i++)\
		g_idata[_i] = _i % 3;\
}

#define POSTAMBLE(g_idata, g_idataSz, g_odata, g_odataSz, g_odataC, g_odataCSz, n) {\
	int _i;\
	float iSum = 0, oSum = 0;\
\
	/* Disclaimer: this validation will be silly (but works) */\
\
	for(_i = 0; _i < g_idataSz; _i++)\
		iSum += g_idata[_i];\
\
	for(_i = 0; _i < g_odataSz; _i++)\
		oSum += g_odata[_i];\
\
	for(_i = 0; _i < g_odataSz; _i++)\
		g_odata[_i] = oSum;\
\
	for(_i = 0; _i < g_odataCSz; _i++)\
		g_odataC[_i] = iSum;\
}
