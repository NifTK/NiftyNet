/**
 * @file _reg_maths.h
 * @brief Library that contains small math routines
 * @author Marc Modat
 * @date 25/03/2009
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */
#ifndef _REG_MATHS_H
#define _REG_MATHS_H

#include <limits>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include "nifti1_io.h"

#if defined (_OPENMP)
#include <omp.h>
#endif

#if _USE_SSE
#include <emmintrin.h>
#include <xmmintrin.h>
#ifdef __SSE3__
#include <pmmintrin.h>
#endif
#endif

#define NR_THROW_EXCEP 1

typedef enum
{
   DEF_FIELD,
   DISP_FIELD,
   CUB_SPLINE_GRID,
   DEF_VEL_FIELD,
   DISP_VEL_FIELD,
   SPLINE_VEL_GRID,
   LIN_SPLINE_GRID
} NREG_TRANS_TYPE;

/* *************************************************************** */
#define reg_pow2(a) ((a)*(a))
#define reg_ceil(a) (ceil(a))
#define reg_round(a) ((a)>0.0 ?(int)((a)+0.5):(int)((a)-0.5))
#ifdef _WIN32
#define reg_floor(a) ((a)>0?(int)(a):(int)((a)-1))
#define reg_floor_size_t(a) ((a)>0?(long)(a):(long)((a)-1))
#else
#define reg_floor(a) ((a)>=0?(int)(a):floor(a))
#endif
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define FMAX(a,b) (a > b ? a : b)
#define IMIN(a,b) (a < b ? a : b)
#define SQR(a) (a==0.0 ? 0.0 : a*a)
/* *************************************************************** */
#ifdef RNIFTYREG
#include <R.h>  // This may have to change to Rcpp.h or RcppEigen.h later
#define reg_exit(){error("[NiftyReg] Fatal error");}
#define reg_print_info(executable,text){Rprintf("[%s] %s\n", executable, text);}
#define reg_print_fct_debug(text){Rprintf("[NiftyReg DEBUG] Function: %s called\n", text);}
#define reg_print_msg_debug(text){Rprintf("[NiftyReg DEBUG] %s\n", text);}
#define reg_print_fct_warn(text){REprintf("[NiftyReg WARNING] Function: %s\n", text);}
#define reg_print_msg_warn(text){REprintf("[NiftyReg WARNING] %s\n", text);}
#define reg_print_fct_error(text){REprintf("[NiftyReg ERROR] Function: %s\n", text);}
#define reg_print_msg_error(text){REprintf("[NiftyReg ERROR] %s\n", text);}
#else
#ifdef NR_THROW_EXCEP
#define reg_exit(){ \
    throw std::runtime_error("[NiftyReg] Exception"); \
}
#else // NR_THROW_EXCEP
#define reg_exit(){ \
    fprintf(stderr,"[NiftyReg] Exit here. File: %s:%i\n",__FILE__, __LINE__); \
    exit(1); \
}
#endif // NR_THROW_EXCEP
#define reg_print_info(executable,text){printf("[%s] %s\n", executable, text);}
#define reg_print_fct_debug(text){printf("[NiftyReg DEBUG] Function: %s called\n", text);}
#define reg_print_msg_debug(text){printf("[NiftyReg DEBUG] %s\n", text);}
#define reg_print_fct_warn(text){printf("[NiftyReg WARNING] Function: %s\n", text);}
#define reg_print_msg_warn(text){printf("[NiftyReg WARNING] %s\n", text);}
#define reg_print_fct_error(text){fprintf(stderr,"[NiftyReg ERROR] Function: %s\n", text);}
#define reg_print_msg_error(text){fprintf(stderr,"[NiftyReg ERROR] %s\n", text);}
#endif
/* *************************************************************** */
#if defined(_WIN32) && !defined(__CYGWIN__)
#include <float.h>
#include <time.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef isnan
#define isnan(_X) _isnan(_X)
#endif
#if (_MSC_VER < 1900)
#ifndef strtof
#define strtof(_s, _t) (float) strtod(_s, _t)
#endif
#endif
template<class PrecisionType> inline int round(PrecisionType x)
{
   return int(x > 0.0 ? (x + 0.5) : (x - 0.5));
}
#if _MSC_VER < 1800 //test if visual studio version older than 2013
template<typename T>inline bool isinf(T value)
{
   return std::numeric_limits<T>::has_infinity && value == std::numeric_limits<T>::infinity();
}
#endif
inline int fabs(int _x)
{
   return (int)fabs((float)(_x));
}
#endif // If on windows...
/* *************************************************************** */
extern "C++" template <class T>
void reg_LUdecomposition(T *inputMatrix,
                         size_t dim,
                         size_t *index);
/* *************************************************************** */
extern "C++" template <class T>
void reg_matrixMultiply(T *mat1,
                        T *mat2,
                        size_t *dim1,
                        size_t *dim2,
                        T * &res);
/* *************************************************************** */
extern "C++" template <class T>
void reg_matrixInvertMultiply(T *mat,
                              size_t dim,
                              size_t *index,
                              T *vec);
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
extern "C++" template<class T>
T* reg_matrix1DAllocate(size_t arraySize);
/* *************************************************************** */
extern "C++" template<class T>
T* reg_matrix1DAllocateAndInitToZero(size_t arraySize);
/* *************************************************************** */
extern "C++" template<class T>
void reg_matrix1DDeallocate(T* mat);
/* *************************************************************** */
extern "C++" template<class T>
T** reg_matrix2DAllocate(size_t arraySizeX, size_t arraySizeY);
/* *************************************************************** */
extern "C++" template<class T>
T** reg_matrix2DAllocateAndInitToZero(size_t arraySizeX, size_t arraySizeY);
/* *************************************************************** */
extern "C++" template<class T>
void reg_matrix2DDeallocate(size_t arraySizeX, T** mat);
/* *************************************************************** */
extern "C++" template<class T>
T** reg_matrix2DTranspose(T** mat, size_t arraySizeX, size_t arraySizeY);
/* *************************************************************** */
extern "C++" template<class T>
T** reg_matrix2DMultiply(T** mat1, size_t mat1X, size_t mat1Y, T** mat2, size_t mat2X, size_t mat2Y, bool transposeMat2);
extern "C++" template<class T>
void reg_matrix2DMultiply(T** mat1, size_t mat1X, size_t mat1Y, T** mat2, size_t mat2X, size_t mat2Y, T** res, bool transposeMat2);
/* *************************************************************** */
extern "C++" template<class T>
T* reg_matrix2DVectorMultiply(T** mat, size_t m, size_t n, T* vect);
extern "C++" template<class T>
void reg_matrix2DVectorMultiply(T** mat, size_t m, size_t n, T* vect, T* res);
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
/** @brief Add two 3-by-3 matrices
*/
mat33 reg_mat33_add(mat33 const* A, mat33 const* B);
mat33 operator+(mat33 A, mat33 B);
/* *************************************************************** */
/** @brief Multipy two 3-by-3 matrices
*/
mat33 reg_mat33_mul(mat33 const* A,
    mat33 const* B);
mat33 operator*(mat33 A,
    mat33 B);
/* *************************************************************** */
//The mat33 represent a 3x3 matrix
void reg_mat33_mul(mat44 const* mat, float const* in, float *out);
void reg_mat33_mul(mat33 const* mat, float const* in, float *out);
/* *************************************************************** */
/** @brief Substract two 3-by-3 matrices
*/
mat33 reg_mat33_minus(mat33 const* A, mat33 const* B);
mat33 operator-(mat33 A, mat33 B);
/* *************************************************************** */
/** @brief Transpose a 3-by-3 matrix
*/
mat33 reg_mat33_trans(mat33 A);
/* *************************************************************** */
/** @brief Diagonalize a 3-by-3 matrix
*/
void reg_mat33_diagonalize(mat33 const* A, mat33 * Q, mat33 * D);
/* *************************************************************** */
/** @brief Set up a 3-by-3 matrix with an identity
*/
void reg_mat33_eye(mat33 *mat);
/* *************************************************************** */
/** @brief Compute the determinant of a 3-by-3 matrix
*/
template<class T> T reg_mat33_det(mat33 const* A);
/* *************************************************************** */
/** @brief Compute the determinant of a 3-by-3 matrix
*/
void reg_mat33_to_nan(mat33 *A);
/* *************************************************************** */
/** @brief Transform a mat44 to a mat33 matrix
*/
mat33 reg_mat44_to_mat33(mat44 const* A);
extern "C++"
void reg_heapSort(float *array_tmp, int *index_tmp, int blockNum);
/* *************************************************************** */
extern "C++" template <class T>
void reg_heapSort(T *array_tmp,int blockNum);
/* *************************************************************** */
/* *************************************************************** */
bool operator==(mat44 A,mat44 B);
/* *************************************************************** */
bool operator!=(mat44 A,mat44 B);
/* *************************************************************** */
/** @brief Multipy two 4-by-4 matrices
 */
mat44 reg_mat44_mul(mat44 const* A,
                    mat44 const* B);
mat44 operator*(mat44 A,
                mat44 B);
/* *************************************************************** */
/** @brief Multipy a vector with a 4-by-4 matrix
 */
void reg_mat44_mul(mat44 const* mat,
                   float const* in,
                   float *out);

void reg_mat44_mul(mat44 const* mat,
                   double const* in,
                   double *out);
/* *************************************************************** */
/** @brief Multipy a 4-by-4 matrix with a scalar
 */
mat44 reg_mat44_mul(mat44 const* mat,
                    double scalar);
/* *************************************************************** */
/** @brief Add two 4-by-4 matrices
 */
mat44 reg_mat44_add(mat44 const* A, mat44 const* B);
mat44 operator+(mat44 A,mat44 B);
/* *************************************************************** */
/** @brief Substract two 4-by-4 matrices
 */
mat44 reg_mat44_minus(mat44 const* A, mat44 const* B);
mat44 operator-(mat44 A,mat44 B);
/* *************************************************************** */
/** @brief Set up a 4-by-4 matrix with an identity
 */
void reg_mat44_eye(mat44 *mat);
/* *************************************************************** */
/** @brief Compute the determinant of a 4-by-4 matrix
 */
template<class T> T reg_mat44_det(mat44 const* A);
/* *************************************************************** */
float reg_mat44_norm_inf(mat44 const* mat);
/* *************************************************************** */
/** @brief Display a mat44 matrix
 */
void reg_mat44_disp(mat44 *mat,
                    char * title);
/* *************************************************************** */
/** @brief Display a mat33 matrix
 */
void reg_mat33_disp(mat33 *mat,
                    char * title);
/* *************************************************************** */
double get_square_distance3D(float * first_point3D, float * second_point3D);
/* *************************************************************** */
double get_square_distance2D(float * first_point2D, float * second_point2D);
/* *************************************************************** */
#endif // _REG_MATHS_H
