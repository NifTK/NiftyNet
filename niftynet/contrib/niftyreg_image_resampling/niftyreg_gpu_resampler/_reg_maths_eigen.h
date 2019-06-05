//_reg_maths_eigen.h
#ifndef _REG_MATHS_EIGEN_H
#define _REG_MATHS_EIGEN_H

#include "nifti1_io.h"

/* *************************************************************** */
/* Functions calling the Eigen library                             */
/* See http://eigen.tuxfamily.org/index.php?title=Main_Page        */
/* *************************************************************** */

/* *************************************************************** */
extern "C++" template <class T>
void svd(T **in, size_t m, size_t n, T * w, T **v);
/* *************************************************************** */
extern "C++" template <class T>
void svd(T **in, size_t m, size_t n, T ***U, T ***S, T ***V);
/* *************************************************************** */
extern "C++" template<class T>
T reg_matrix2DDet(T** mat, size_t m, size_t n);
/* *************************************************************** */
/** @brief Compute the inverse of a  4-by-4 matrix
*/
mat44 reg_mat44_inv(mat44 const* mat);
/* *************************************************************** */
/** @brief Compute the square root of a 4-by-4 matrix
*/
mat44 reg_mat44_sqrt(mat44 const* mat);
/* *************************************************************** */
/** @brief Compute the log of a 3-by-3 matrix
*/
void reg_mat33_expm(mat33 *in_tensor);
/* *************************************************************** */
/** @brief Compute the exp of a 4-by-4 matrix
*/
mat44 reg_mat44_expm(const mat44 *mat);
/* *************************************************************** */
/** @brief Compute the log of a 3-by-3 matrix
*/
void reg_mat33_logm(mat33 *in_tensor);
/* *************************************************************** */
/** @brief Compute the log of a 4-by-4 matrix
*/
mat44 reg_mat44_logm(const mat44 *mat);
/* *************************************************************** */
/** @brief Compute the average of two matrices using a log-euclidean
* framework
*/
mat44 reg_mat44_avg2(mat44 const* A, mat44 const* b);

#endif
