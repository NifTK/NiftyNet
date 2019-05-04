#define USE_EIGEN

#include "_reg_maths_eigen.h"
#include "_reg_maths.h"
#include "nifti1_io.h"

// Eigen headers are in there because of the nvcc preprocessing step
#include "Eigen/Core"
#include "Eigen/SVD"
#include "unsupported/Eigen/MatrixFunctions"

//_reg_maths_eigen.cpp
/* *************************************************************** */
/** @brief SVD
* @param in input matrix to decompose - in place
* @param size_m row
* @param size_n colomn
* @param w diagonal term
* @param v rotation part
*/
template<class T>
void svd(T **in, size_t size_m, size_t size_n, T * w, T **v) {
   if (size_m == 0 || size_n == 0) {
      reg_print_fct_error("svd");
      reg_print_msg_error("The specified matrix is empty");
      reg_exit();
   }

#ifdef _WIN32
   long sm, sn, sn2;
   long size__m = (long)size_m, size__n = (long)size_n;
#else
   size_t sm, sn, sn2;
   size_t size__m = size_m, size__n = size_n;
#endif
   Eigen::MatrixXd m(size_m, size_n);

   //Convert to Eigen matrix
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(in,m, size__m, size__n) \
   private(sm, sn)
#endif
   for (sm = 0; sm < size__m; sm++)
   {
      for (sn = 0; sn < size__n; sn++)
      {
         m(sm, sn) = static_cast<double>(in[sm][sn]);
      }
   }

   Eigen::JacobiSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);

#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(in,svd,v,w, size__n,size__m) \
   private(sn2, sn, sm)
#endif
   for (sn = 0; sn < size__n; sn++) {
      w[sn] = static_cast<T>(svd.singularValues()(sn));
      for (sn2 = 0; sn2 < size__n; sn2++) {
         v[sn2][sn] = static_cast<T>(svd.matrixV()(sn2, sn));
      }
      for (sm = 0; sm < size__m; sm++) {
         in[sm][sn] = static_cast<T>(svd.matrixU()(sm, sn));
      }
   }
}
template void svd<float>(float **in, size_t m, size_t n, float * w, float **v);
template void svd<double>(double **in, size_t m, size_t n, double * w, double **v);
/* *************************************************************** */
/**
* @brief SVD
* @param in input matrix to decompose
* @param size_m row
* @param size_n colomn
* @param U unitary matrices
* @param S diagonal matrix
* @param V unitary matrices
*  X = U*S*V'
*/
template<class T>
void svd(T **in, size_t size_m, size_t size_n, T ***U, T ***S, T ***V) {
   if (in == NULL) {
      reg_print_fct_error("svd");
      reg_print_msg_error("The specified matrix is empty");
      reg_exit();
   }

#ifdef _WIN32
   long sm, sn, sn2, min_dim, i, j;
   long size__m = (long)size_m, size__n = (long)size_n;
#else
   size_t sm, sn, min_dim, i, j;
   size_t size__m = size_m, size__n = size_n;
#endif
   Eigen::MatrixXd m(size__m, size__n);

   //Convert to Eigen matrix
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(in, m, size__m, size__n) \
   private(sm, sn)
#endif
   for (sm = 0; sm < size__m; sm++)
   {
      for (sn = 0; sn < size__n; sn++)
      {
         m(sm, sn) = static_cast<double>(in[sm][sn]);
      }
   }

   Eigen::JacobiSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);

   min_dim = std::min(size__m, size__n);
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(svd, min_dim, S) \
   private(i, j)
#endif
   //Convert to C matrix
   for (i = 0; i < min_dim; i++) {
      for (j = 0; j < min_dim; j++) {
         if (i == j) {
            (*S)[i][j] = static_cast<T>(svd.singularValues()(i));
         }
         else {
            (*S)[i][j] = 0;
         }
      }
   }

   if (size__m > size__n) {
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(svd, min_dim, V) \
   private(i, j)
#endif
      //Convert to C matrix
      for (i = 0; i < min_dim; i++) {
         for (j = 0; j < min_dim; j++) {
            (*V)[i][j] = static_cast<T>(svd.matrixV()(i, j));

         }
      }
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(svd, size__m, size__n, U) \
   private(i, j)
#endif
      for (i = 0; i < size__m; i++) {
         for (j = 0; j < size__n; j++) {
            (*U)[i][j] = static_cast<T>(svd.matrixU()(i, j));
         }
      }
   }
   else {
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(svd, min_dim, U) \
   private(i, j)
#endif
      //Convert to C matrix
      for (i = 0; i < min_dim; i++) {
         for (j = 0; j < min_dim; j++) {
            (*U)[i][j] = static_cast<T>(svd.matrixU()(i, j));

         }
      }
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(svd, size__m, size__n, V) \
   private(i, j)
#endif
      for (i = 0; i < size__n; i++) {
         for (j = 0; j < size__m; j++) {
            (*V)[i][j] = static_cast<T>(svd.matrixV()(i, j));
         }
      }
   }

}
template void svd<float>(float **in, size_t size_m, size_t size_n, float ***U, float ***S, float ***V);
template void svd<double>(double **in, size_t size_m, size_t size_n, double ***U, double ***S, double ***V);
/* *************************************************************** */
template<class T>
T reg_matrix2DDet(T** mat, size_t m, size_t n) {
   if (m != n) {
      char text[255]; sprintf(text, "The matrix have to be square: [%zu %zu]",
                              m, n);
      reg_print_fct_error("reg_matrix2DDeterminant");
      reg_print_msg_error(text);
      reg_exit();
   }
   double res;
   if (m == 2) {
      res = static_cast<double>(mat[0][0]) * static_cast<double>(mat[1][1]) - static_cast<double>(mat[1][0]) * static_cast<double>(mat[0][1]);
   }
   else if (m == 3) {
      res = (static_cast<double>(mat[0][0]) * (static_cast<double>(mat[1][1]) * static_cast<double>(mat[2][2]) - static_cast<double>(mat[1][2]) * static_cast<double>(mat[2][1]))) -
            (static_cast<double>(mat[0][1]) * (static_cast<double>(mat[1][0]) * static_cast<double>(mat[2][2]) - static_cast<double>(mat[1][2]) * static_cast<double>(mat[2][0]))) +
            (static_cast<double>(mat[0][2]) * (static_cast<double>(mat[1][0]) * static_cast<double>(mat[2][1]) - static_cast<double>(mat[1][1]) * static_cast<double>(mat[2][0])));
   }
   else {
      // Convert to Eigen format
      Eigen::MatrixXd eigenRes(m, n);
      for (size_t i = 0; i < m; i++) {
         for (size_t j = 0; j < n; j++) {
            eigenRes(i, j) = static_cast<double>(mat[i][j]);
         }
      }
      res = eigenRes.determinant();
   }
   return static_cast<T>(res);
}
template float reg_matrix2DDet<float>(float** mat, size_t m, size_t n);
template double reg_matrix2DDet<double>(double** mat, size_t m, size_t n);
/* *************************************************************** */
mat44 reg_mat44_sqrt(mat44 const* mat)
{
   mat44 X;
   Eigen::Matrix4d m;
   for (size_t i = 0; i < 4; ++i)
   {
      for (size_t j = 0; j < 4; ++j)
      {
         m(i, j) = static_cast<double>(mat->m[i][j]);
      }
   }
   m = m.sqrt();
   for (size_t i = 0; i < 4; ++i)
      for (size_t j = 0; j < 4; ++j)
         X.m[i][j] = static_cast<float>(m(i, j));
   return X;
}
/* *************************************************************** */
void reg_mat33_expm(mat33 *in_tensor)
{
   int sm, sn;
   Eigen::Matrix3d tensor;

   // Convert to Eigen format
   for (sm = 0; sm < 3; sm++){
      for (sn = 0; sn < 3; sn++){
         float val=in_tensor->m[sm][sn];
         if(val!=val) return;
         tensor(sm, sn) = static_cast<double>(val);
      }
   }

   // Compute exp(E)
   tensor = tensor.exp();

   // Convert the result to mat33 format
   for (sm = 0; sm < 3; sm++)
      for (sn = 0; sn < 3; sn++)
         in_tensor->m[sm][sn] = static_cast<float>(tensor(sm, sn));
}
/* *************************************************************** */
mat44 reg_mat44_expm(mat44 const* mat)
{
   mat44 X;
   Eigen::Matrix4d m;
   for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
         m(i, j) = static_cast<double>(mat->m[i][j]);
      }
   }
   m = m.exp();
   //
   for (size_t i = 0; i < 4; ++i)
      for (size_t j = 0; j < 4; ++j)
         X.m[i][j] = static_cast<float>(m(i, j));

   return X;
}
/* *************************************************************** */
void reg_mat33_logm(mat33 *in_tensor)
{
   int sm, sn;
   Eigen::Matrix3d tensor;

   // Convert to Eigen format
   bool all_zeros = true;
   double det = 0;
   for (sm = 0; sm < 3; sm++){
      for (sn = 0; sn < 3; sn++){
         float val=in_tensor->m[sm][sn];
         if(val!=0.f) all_zeros=false;
         if(val!=val) return;
         tensor(sm, sn) = static_cast<double>(val);
      }
   }
   // Actually R case requires invertible and no negative real ev,
   // but the only observed case so far was non-invertible.
   // determinant is not a perfect check for invertibility and
   // identity with zero not great either, but the alternative
   // is a general eigensolver and the logarithm function should
   // suceed unless convergence just isn't happening.
   det = tensor.determinant();
   if(all_zeros==true || det == 0){
      reg_mat33_to_nan(in_tensor);
      return;
   }

   // Compute the actual matrix log
   tensor = tensor.log();

   // Convert the result to mat33 format
   for (sm = 0; sm < 3; sm++)
      for (sn = 0; sn < 3; sn++)
         in_tensor->m[sm][sn] = static_cast<float>(tensor(sm, sn));
}
/* *************************************************************** */
mat44 reg_mat44_logm(mat44 const* mat)
{
   mat44 X;
   Eigen::Matrix4d m;
   for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
         m(i, j) = static_cast<double>(mat->m[i][j]);
      }
   }
   m = m.log();
   for (size_t i = 0; i < 4; ++i)
      for (size_t j = 0; j < 4; ++j)
         X.m[i][j] = static_cast<float>(m(i, j));
   return X;
}
/* *************************************************************** */
mat44 reg_mat44_inv(mat44 const* mat)
{
   mat44 out;
   Eigen::Matrix4d m, m_inv;
   for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
         m(i, j) = static_cast<double>(mat->m[i][j]);
      }
   }
   m_inv = m.inverse();
   for (size_t i = 0; i < 4; ++i)
      for (size_t j = 0; j < 4; ++j)
         out.m[i][j] = static_cast<float>(m_inv(i, j));
   //
   return out;

}
/* *************************************************************** */
mat44 reg_mat44_avg2(mat44 const* A, mat44 const* B)
{
   mat44 out;
   mat44 logA = reg_mat44_logm(A);
   mat44 logB = reg_mat44_logm(B);
   for (int i = 0; i < 4; ++i) {
      logA.m[3][i] = 0.f;
      logB.m[3][i] = 0.f;
   }
   logA = reg_mat44_add(&logA, &logB);
   out = reg_mat44_mul(&logA, 0.5);
   return reg_mat44_expm(&out);

}
