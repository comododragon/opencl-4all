/* ********************************************************************************************* */
/* * Gallois-field Arithmetic Functions Macros                                                 * */
/* * Author: André Bannwart Perina                                                             * */
/* ********************************************************************************************* */
/* * Copyright (c) 2016 André B. Perina                                                        * */
/* *                                                                                           * */
/* * Permission is hereby granted, free of charge, to any person obtaining a copy of this      * */
/* * software and associated documentation files (the "Software"), to deal in the Software     * */
/* * without restriction, including without limitation the rights to use, copy, modify,        * */
/* * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to        * */
/* * permit persons to whom the Software is furnished to do so, subject to the following       * */
/* * conditions:                                                                               * */
/* *                                                                                           * */
/* * The above copyright notice and this permission notice shall be included in all copies     * */
/* * or substantial portions of the Software.                                                  * */
/* *                                                                                           * */
/* * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,       * */
/* * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR  * */
/* * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE * */
/* * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      * */
/* * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER    * */
/* * DEALINGS IN THE SOFTWARE.                                                                 * */
/* ********************************************************************************************* */

#ifndef GFA_H
#define GFA_H

/**
 * @brief PPCHAR constant.
 */
#define GFA_PPCHAR 29

/**
 * @brief Gallois-field add.
 *
 * @param res Variable to receive result.
 * @param a First operand.
 * @param b Second operand.
 *
 * @note This is a macro, therefore @p res should not be a reference, but the variable itself.
 */
#define GFA_ADD(res, a, b) {\
	res = a ^ b;\
}

/**
 * @brief Gallois-field multiplication.
 *
 * @param ctr Variable to be used as a counter.
 * @param res Variable to receive result.
 * @param a First operand.
 * @param b Second operand.
 *
 * @note This is a macro, therefore @p ctr and @p res should not be references, but the variables themselves.
 */
#define GFA_MULT(ctr, res, a, b) {\
	res = 0;\
\
	for(ctr = 0; ctr < 8; ctr++)\
		if(b & (1 << ctr))\
			res ^= (unsigned short) (a << ctr);\
	for(ctr = 15; ctr > 7; ctr--)\
		if(res & (1 << ctr))\
			res ^= (unsigned short) (GFA_PPCHAR << (ctr - 8));\
\
	res &= 0xff;\
}

/**
 * @brief Gallois-field inversion.
 *
 * @param gfInvLUT Gallois-field inversion lookup table.
 * @param res Variable to receive result.
 * @param a First operand.
 *
 * @note This is a macro, therefore @p res should not be a reference, but the variable itself.
 */
#define GFA_INV(gfInvLUT, res, a) {\
	res = gfInvLUT[a];\
}

#endif

