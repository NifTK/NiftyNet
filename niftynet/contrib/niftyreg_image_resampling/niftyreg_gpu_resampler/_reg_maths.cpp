#ifndef _REG_MATHS_CPP
#define _REG_MATHS_CPP

#include "_reg_maths.h"
//STD
#include <map>
#include <vector>

#define mat(i,j,dim) mat[i*dim+j]

/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_LUdecomposition(T *mat,
                         size_t dim,
                         size_t *index)
{
    T *vv = (T *)malloc(dim * sizeof(T));
    size_t i, j, k, imax = 0;

    for (i = 0; i < dim; ++i)
    {
        T big = 0.f;
        T temp;
        for (j = 0; j < dim; ++j)
            if ((temp = fabs(mat(i, j, dim)))>big)
                big = temp;
        if (big == 0.f)
        {
            reg_print_fct_error("reg_LUdecomposition");
            reg_print_msg_error("Singular matrix");
            reg_exit();
        }
        vv[i] = 1.0 / big;
    }
    for (j = 0; j < dim; ++j)
    {
        for (i = 0; i < j; ++i)
        {
            T sum = mat(i, j, dim);
            for (k = 0; k < i; k++) sum -= mat(i, k, dim)*mat(k, j, dim);
            mat(i, j, dim) = sum;
        }
        T big = 0.f;
        T dum;
        for (i = j; i < dim; ++i)
        {
            T sum = mat(i, j, dim);
            for (k = 0; k < j; ++k) sum -= mat(i, k, dim)*mat(k, j, dim);
            mat(i, j, dim) = sum;
            if ((dum = vv[i] * fabs(sum)) >= big)
            {
                big = dum;
                imax = i;
            }
        }
        if (j != imax)
        {
            for (k = 0; k < dim; ++k)
            {
                dum = mat(imax, k, dim);
                mat(imax, k, dim) = mat(j, k, dim);
                mat(j, k, dim) = dum;
            }
            vv[imax] = vv[j];
        }
        index[j] = imax;
        if (mat(j, j, dim) == 0) mat(j, j, dim) = 1.0e-20;
        if (j != dim - 1)
        {
            dum = 1.0 / mat(j, j, dim);
            for (i = j + 1; i < dim; ++i) mat(i, j, dim) *= dum;
        }
    }
    free(vv);
    return;
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_matrixInvertMultiply(T *mat,
                              size_t dim,
                              size_t *index,
                              T *vec)
{
    // Perform the LU decomposition if necessary
    if (index == NULL)
        reg_LUdecomposition(mat, dim, index);

    int ii = 0;
    for (size_t i = 0; i < dim; ++i)
    {
        int ip = index[i];
        T sum = vec[ip];
        vec[ip] = vec[i];
        if (ii != 0)
        {
            for (int j = ii - 1; j < (int)i; ++j)
                sum -= mat(i, j, dim)*vec[j];
        }
        else if (sum != 0)
            ii = i + 1;
        vec[i] = sum;
    }
    for (int i = (int)dim - 1; i > -1; --i)
    {
        T sum = vec[i];
        for (int j = i + 1; j < (int)dim; ++j)
            sum -= mat(i, j, dim)*vec[j];
        vec[i] = sum / mat(i, i, dim);
    }
}
template void reg_matrixInvertMultiply<float>(float *, size_t, size_t *, float *);
template void reg_matrixInvertMultiply<double>(double *, size_t, size_t *, double *);
/* *************************************************************** */
/* *************************************************************** */
template<class T>
void reg_matrixMultiply(T *mat1,
                        T *mat2,
                        size_t *dim1,
                        size_t *dim2,
                        T * &res)
{
    // First check that the dimension are appropriate
    if (dim1[1] != dim2[0])
    {
        char text[255]; sprintf(text, "Matrices can not be multiplied due to their size: [%zu %zu] [%zu %zu]",
            dim1[0], dim1[1], dim2[0], dim2[1]);
        reg_print_fct_error("reg_matrixMultiply");
        reg_print_msg_error(text);
        reg_exit();
    }
    size_t resDim[2] = {dim1[0], dim2[1]};
    // Allocate the result matrix
    if (res != NULL)
        free(res);
    res = (T *)calloc(resDim[0] * resDim[1], sizeof(T));
    // Multiply both matrices
    for (size_t j = 0; j < resDim[1]; ++j)
    {
        for (size_t i = 0; i < resDim[0]; ++i)
        {
            double sum = 0.0;
            for (size_t k = 0; k < dim1[1]; ++k)
            {
                sum += mat1[k * dim1[0] + i] * mat2[j * dim2[0] + k];
            }
            res[j * resDim[0] + i] = sum;
        } // i
    } // j
}
template void reg_matrixMultiply<float>(float *, float *, size_t *, size_t *, float * &);
template void reg_matrixMultiply<double>(double *, double *, size_t *, size_t *, double * &);
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
template<class T>
T* reg_matrix1DAllocate(size_t arraySize) {
    T* res = (T*)malloc(arraySize*sizeof(T));
    return res;
}
template bool* reg_matrix1DAllocate<bool>(size_t arraySize);
template float* reg_matrix1DAllocate<float>(size_t arraySize);
template double* reg_matrix1DAllocate<double>(size_t arraySize);
/* *************************************************************** */
template<class T>
T* reg_matrix1DAllocateAndInitToZero(size_t arraySize) {
    T* res = (T*)calloc(arraySize, sizeof(T));
    return res;
}
template bool* reg_matrix1DAllocateAndInitToZero<bool>(size_t arraySize);
template float* reg_matrix1DAllocateAndInitToZero<float>(size_t arraySize);
template double* reg_matrix1DAllocateAndInitToZero<double>(size_t arraySize);
/* *************************************************************** */
template<class T>
void reg_matrix1DDeallocate(T* mat) {
    free(mat);
}
template void reg_matrix1DDeallocate<bool>(bool* mat);
template void reg_matrix1DDeallocate<float>(float* mat);
template void reg_matrix1DDeallocate<double>(double* mat);
/* *************************************************************** */
template<class T>
T** reg_matrix2DAllocate(size_t arraySizeX, size_t arraySizeY) {
    T** res;
    res = (T**)malloc(arraySizeX*sizeof(T*));
    for (size_t i = 0; i < arraySizeX; i++) {
        res[i] = (T*)malloc(arraySizeY*sizeof(T));
    }
    return res;
}
template float** reg_matrix2DAllocate<float>(size_t arraySizeX, size_t arraySizeY);
template double** reg_matrix2DAllocate<double>(size_t arraySizeX, size_t arraySizeY);
/* *************************************************************** */
template<class T>
T** reg_matrix2DAllocateAndInitToZero(size_t arraySizeX, size_t arraySizeY) {
    T** res;
    res = (T**)calloc(arraySizeX, sizeof(T*));
    for (size_t i = 0; i < arraySizeX; i++) {
        res[i] = (T*)calloc(arraySizeY, sizeof(T));
    }
    return res;
}
template float** reg_matrix2DAllocateAndInitToZero<float>(size_t arraySizeX, size_t arraySizeY);
template double** reg_matrix2DAllocateAndInitToZero<double>(size_t arraySizeX, size_t arraySizeY);
/* *************************************************************** */
template<class T>
void reg_matrix2DDeallocate(size_t arraySizeX, T** mat) {
    for (size_t i = 0; i < arraySizeX; i++) {
        free(mat[i]);
    }
    free(mat);
}
template void reg_matrix2DDeallocate<float>(size_t arraySizeX, float** mat);
template void reg_matrix2DDeallocate<double>(size_t arraySizeX, double** mat);
/* *************************************************************** */
template<class T>
T** reg_matrix2DTranspose(T** mat, size_t arraySizeX, size_t arraySizeY) {
    T** res;
    res = (T**)malloc(arraySizeY*sizeof(T*));
    for (size_t i = 0; i < arraySizeY; i++) {
        res[i] = (T*)malloc(arraySizeX*sizeof(T));
    }
    for (size_t i = 0; i < arraySizeX; i++) {
        for (size_t j = 0; j < arraySizeY; j++) {
            res[j][i] = mat[i][j];
        }
    }
    return res;
}
template float** reg_matrix2DTranspose<float>(float** mat, size_t arraySizeX, size_t arraySizeY);
template double** reg_matrix2DTranspose<double>(double** mat, size_t arraySizeX, size_t arraySizeY);
/* *************************************************************** */
template<class T>
T** reg_matrix2DMultiply(T** mat1, size_t mat1X, size_t mat1Y, T** mat2, size_t mat2X, size_t mat2Y, bool transposeMat2) {
    if (transposeMat2 == false) {
        // First check that the dimension are appropriate
        if (mat1Y != mat2X) {
            char text[255]; sprintf(text, "Matrices can not be multiplied due to their size: [%zu %zu] [%zu %zu]",
                mat1X, mat1Y, mat2X, mat2Y);
            reg_print_fct_error("reg_matrix2DMultiply");
            reg_print_msg_error(text);
            reg_exit();
        }

        size_t nbElement = mat1Y;
        double resTemp = 0;
        T** res = reg_matrix2DAllocate<T>(mat1X,mat2Y);

        for (size_t i = 0; i < mat1X; i++) {
            for (size_t j = 0; j < mat2Y; j++) {
                resTemp = 0;
                for (size_t k = 0; k < nbElement; k++) {
                    resTemp += static_cast<double>(mat1[i][k]) * static_cast<double>(mat2[k][j]);
                }
                res[i][j] = static_cast<T>(resTemp);
            }
        }
        //Output
       return res;
    }
    else {
        // First check that the dimension are appropriate
        if (mat1Y != mat2Y) {
            char text[255]; sprintf(text, "Matrices can not be multiplied due to their size: [%zu %zu] [%zu %zu]",
                mat1X, mat1Y, mat2Y, mat2X);
            reg_print_fct_error("reg_matrix2DMultiply");
            reg_print_msg_error(text);
            reg_exit();
        }
        size_t nbElement = mat1Y;
        double resTemp = 0;
        T** res = reg_matrix2DAllocate<T>(mat1X,mat2X);

        for (size_t i = 0; i < mat1X; i++) {
            for (size_t j = 0; j < mat2X; j++) {
                resTemp = 0;
                for (size_t k = 0; k < nbElement; k++) {
                    resTemp += static_cast<double>(mat1[i][k]) * static_cast<double>(mat2[j][k]);
                }
                res[i][j] = static_cast<T>(resTemp);
            }
        }
        //Output
        return res;
    }
}
template float** reg_matrix2DMultiply<float>(float** mat1, size_t mat1X, size_t mat1Y, float** mat2, size_t mat2X, size_t mat2Y, bool transposeMat2);
template double** reg_matrix2DMultiply<double>(double** mat1, size_t mat1X, size_t mat1Y, double** mat2, size_t mat2X, size_t mat2Y, bool transposeMat2);
/* *************************************************************** */
template<class T>
void reg_matrix2DMultiply(T** mat1, size_t mat1X, size_t mat1Y, T** mat2, size_t mat2X, size_t mat2Y, T** resT, bool transposeMat2) {
    if (transposeMat2 == false) {
        // First check that the dimension are appropriate
        if (mat1Y != mat2X) {
            char text[255]; sprintf(text, "Matrices can not be multiplied due to their size: [%zu %zu] [%zu %zu]",
                mat1X, mat1Y, mat2X, mat2Y);
            reg_print_fct_error("reg_matrix2DMultiply");
            reg_print_msg_error(text);
            reg_exit();
        }
        size_t nbElement = mat1Y;
        double resTemp;

        for (size_t i = 0; i < mat1X; i++) {
            for (size_t j = 0; j < mat2Y; j++) {
                resTemp = 0;
                for (size_t k = 0; k < nbElement; k++) {
                    resTemp += static_cast<double>(mat1[i][k]) * static_cast<double>(mat2[k][j]);
                }
                resT[i][j] = static_cast<T>(resTemp);
            }
        }
    }
    else {
        // First check that the dimension are appropriate
        if (mat1Y != mat2Y) {
            char text[255]; sprintf(text, "Matrices can not be multiplied due to their size: [%zu %zu] [%zu %zu]",
                mat1X, mat1Y, mat2Y, mat2X);
            reg_print_fct_error("reg_matrix2DMultiply");
            reg_print_msg_error(text);
            reg_exit();
        }
        size_t nbElement = mat1Y;
        double resTemp;

        for (size_t i = 0; i < mat1X; i++) {
            for (size_t j = 0; j < mat2X; j++) {
                resTemp = 0;
                for (size_t k = 0; k < nbElement; k++) {
                    resTemp += static_cast<double>(mat1[i][k]) * static_cast<double>(mat2[j][k]);
                }
                resT[i][j] = static_cast<T>(resTemp);
            }
        }
    }
}
template void reg_matrix2DMultiply<float>(float** mat1, size_t mat1X, size_t mat1Y, float** mat2, size_t mat2X, size_t mat2Y, float** resT, bool transposeMat2);
template void reg_matrix2DMultiply<double>(double** mat1, size_t mat1X, size_t mat1Y, double** mat2, size_t mat2X, size_t mat2Y, double** resT, bool transposeMat2);
/* *************************************************************** */
// Multiply a matrix with a vector - we assume correct dimension
template<class T>
T* reg_matrix2DVectorMultiply(T** mat, size_t m, size_t n, T* vect) {

    T* res = reg_matrix1DAllocate<T>(m);
    double resTemp;

    for (size_t i = 0; i < m; i++) {
        resTemp = 0;
        for (size_t k = 0; k < n; k++) {
            resTemp += static_cast<double>(mat[i][k]) * static_cast<double>(vect[k]);
        }
        res[i] = static_cast<T>(resTemp);
    }
    return res;
}
template float* reg_matrix2DVectorMultiply<float>(float** mat, size_t m, size_t n, float* vect);
template double* reg_matrix2DVectorMultiply<double>(double** mat, size_t m, size_t n, double* vect);
/* *************************************************************** */
template<class T>
void reg_matrix2DVectorMultiply(T** mat, size_t m, size_t n, T* vect, T* res) {

    double resTemp = 0;

    for (size_t i = 0; i < m; i++) {
        resTemp = 0;
        for (size_t k = 0; k < n; k++) {
            resTemp += static_cast<double>(mat[i][k]) * static_cast<double>(vect[k]);
        }
        res[i] = static_cast<T>(resTemp);
    }
}
template void reg_matrix2DVectorMultiply<float>(float** mat, size_t m, size_t n, float* vect, float* res);
template void reg_matrix2DVectorMultiply<double>(double** mat, size_t m, size_t n, double* vect, double* res);
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
// Heap sort
void reg_heapSort(float *array_tmp, int *index_tmp, int blockNum)
{
    float *array = &array_tmp[-1];
    int *index = &index_tmp[-1];
    int l = (blockNum >> 1) + 1;
    int ir = blockNum;
    float val;
    int iVal;
    for (;;)
    {
        if (l > 1)
        {
            val = array[--l];
            iVal = index[l];
        }
        else
        {
            val = array[ir];
            iVal = index[ir];
            array[ir] = array[1];
            index[ir] = index[1];
            if (--ir == 1)
            {
                array[1] = val;
                index[1] = iVal;
                break;
            }
        }
        int i = l;
        int j = l + l;
        while (j <= ir)
        {
            if (j < ir && array[j] < array[j + 1])
                j++;
            if (val < array[j])
            {
                array[i] = array[j];
                index[i] = index[j];
                i = j;
                j <<= 1;
            }
            else
                break;
        }
        array[i] = val;
        index[i] = iVal;
    }
}
/* *************************************************************** */
// Heap sort
template<class DTYPE>
void reg_heapSort(DTYPE *array_tmp, int blockNum)
{
    DTYPE *array = &array_tmp[-1];
    int l = (blockNum >> 1) + 1;
    int ir = blockNum;
    DTYPE val;
    for (;;)
    {
        if (l > 1)
        {
            val = array[--l];
        }
        else
        {
            val = array[ir];
            array[ir] = array[1];
            if (--ir == 1)
            {
                array[1] = val;
                break;
            }
        }
        int i = l;
        int j = l + l;
        while (j <= ir)
        {
            if (j < ir && array[j] < array[j + 1])
                j++;
            if (val < array[j])
            {
                array[i] = array[j];
                i = j;
                j <<= 1;
            }
            else
                break;
        }
        array[i] = val;
    }
}
template void reg_heapSort<float>(float *array_tmp, int blockNum);
template void reg_heapSort<double>(double *array_tmp, int blockNum);
/* *************************************************************** */
/* *************************************************************** */
bool operator==(mat44 A, mat44 B)
{
    for (unsigned i = 0; i < 4; ++i)
    {
        for (unsigned j = 0; j < 4; ++j)
        {
            if (A.m[i][j] != B.m[i][j])
                return false;
        }
    }
    return true;
}
/* *************************************************************** */
bool operator!=(mat44 A, mat44 B)
{
    for (unsigned i = 0; i < 4; ++i)
    {
        for (unsigned j = 0; j < 4; ++j)
        {
            if (A.m[i][j] != B.m[i][j])
                return true;
        }
    }
    return false;
}
/* *************************************************************** */
/* *************************************************************** */
template<class T>
T reg_mat44_det(mat44 const* A)
{
    double D =
        static_cast<double>(A->m[0][0]) * static_cast<double>(A->m[1][1]) * static_cast<double>(A->m[2][2]) * static_cast<double>(A->m[3][3])
        - static_cast<double>(A->m[0][0]) * static_cast<double>(A->m[1][1]) * static_cast<double>(A->m[3][2]) * static_cast<double>(A->m[2][3])
        - static_cast<double>(A->m[0][0]) * static_cast<double>(A->m[2][1]) * static_cast<double>(A->m[1][2]) * static_cast<double>(A->m[3][3])
        + static_cast<double>(A->m[0][0]) * static_cast<double>(A->m[2][1]) * static_cast<double>(A->m[3][2]) * static_cast<double>(A->m[1][3])
        + static_cast<double>(A->m[0][0]) * static_cast<double>(A->m[3][1]) * static_cast<double>(A->m[1][2]) * static_cast<double>(A->m[2][3])
        - static_cast<double>(A->m[0][0]) * static_cast<double>(A->m[3][1]) * static_cast<double>(A->m[2][2]) * static_cast<double>(A->m[1][3])
        - static_cast<double>(A->m[1][0]) * static_cast<double>(A->m[0][1]) * static_cast<double>(A->m[2][2]) * static_cast<double>(A->m[3][3])
        + static_cast<double>(A->m[1][0]) * static_cast<double>(A->m[0][1]) * static_cast<double>(A->m[3][2]) * static_cast<double>(A->m[2][3])
        + static_cast<double>(A->m[1][0]) * static_cast<double>(A->m[2][1]) * static_cast<double>(A->m[0][2]) * static_cast<double>(A->m[3][3])
        - static_cast<double>(A->m[1][0]) * static_cast<double>(A->m[2][1]) * static_cast<double>(A->m[3][2]) * static_cast<double>(A->m[0][3])
        - static_cast<double>(A->m[1][0]) * static_cast<double>(A->m[3][1]) * static_cast<double>(A->m[0][2]) * static_cast<double>(A->m[2][3])
        + static_cast<double>(A->m[1][0]) * static_cast<double>(A->m[3][1]) * static_cast<double>(A->m[2][2]) * static_cast<double>(A->m[0][3])
        + static_cast<double>(A->m[2][0]) * static_cast<double>(A->m[0][1]) * static_cast<double>(A->m[1][2]) * static_cast<double>(A->m[3][3])
        - static_cast<double>(A->m[2][0]) * static_cast<double>(A->m[0][1]) * static_cast<double>(A->m[3][2]) * static_cast<double>(A->m[1][3])
        - static_cast<double>(A->m[2][0]) * static_cast<double>(A->m[1][1]) * static_cast<double>(A->m[0][2]) * static_cast<double>(A->m[3][3])
        + static_cast<double>(A->m[2][0]) * static_cast<double>(A->m[1][1]) * static_cast<double>(A->m[3][2]) * static_cast<double>(A->m[0][3])
        + static_cast<double>(A->m[2][0]) * static_cast<double>(A->m[3][1]) * static_cast<double>(A->m[0][2]) * static_cast<double>(A->m[1][3])
        - static_cast<double>(A->m[2][0]) * static_cast<double>(A->m[3][1]) * static_cast<double>(A->m[1][2]) * static_cast<double>(A->m[0][3])
        - static_cast<double>(A->m[3][0]) * static_cast<double>(A->m[0][1]) * static_cast<double>(A->m[1][2]) * static_cast<double>(A->m[2][3])
        + static_cast<double>(A->m[3][0]) * static_cast<double>(A->m[0][1]) * static_cast<double>(A->m[2][2]) * static_cast<double>(A->m[1][3])
        + static_cast<double>(A->m[3][0]) * static_cast<double>(A->m[1][1]) * static_cast<double>(A->m[0][2]) * static_cast<double>(A->m[2][3])
        - static_cast<double>(A->m[3][0]) * static_cast<double>(A->m[1][1]) * static_cast<double>(A->m[2][2]) * static_cast<double>(A->m[0][3])
        - static_cast<double>(A->m[3][0]) * static_cast<double>(A->m[2][1]) * static_cast<double>(A->m[0][2]) * static_cast<double>(A->m[1][3])
        + static_cast<double>(A->m[3][0]) * static_cast<double>(A->m[2][1]) * static_cast<double>(A->m[1][2]) * static_cast<double>(A->m[0][3]);
    return static_cast<T>(D);
}
template float reg_mat44_det<float>(mat44 const* A);
template double reg_mat44_det<double>(mat44 const* A);
/* *************************************************************** */
/* *************************************************************** */
template<class T>
T reg_mat33_det(mat33 const* A)
{
    double D = static_cast<T>((static_cast<double>(A->m[0][0]) * (static_cast<double>(A->m[1][1]) * static_cast<double>(A->m[2][2]) - static_cast<double>(A->m[1][2]) * static_cast<double>(A->m[2][1]))) -
        (static_cast<double>(A->m[0][1]) * (static_cast<double>(A->m[1][0]) * static_cast<double>(A->m[2][2]) - static_cast<double>(A->m[1][2]) * static_cast<double>(A->m[2][0]))) +
        (static_cast<double>(A->m[0][2]) * (static_cast<double>(A->m[1][0]) * static_cast<double>(A->m[2][1]) - static_cast<double>(A->m[1][1]) * static_cast<double>(A->m[2][0]))));
    return static_cast<T>(D);
}
template float reg_mat33_det<float>(mat33 const* A);
template double reg_mat33_det<double>(mat33 const* A);
/* *************************************************************** */
/* *************************************************************** */
void reg_mat33_to_nan(mat33 *A)
{
   for(int i=0;i<3;++i)
      for(int j=0;j<3;++j)
         A->m[i][j] = std::numeric_limits<float>::quiet_NaN();
}
/* *************************************************************** */
/* *************************************************************** */
mat33 reg_mat44_to_mat33(mat44 const* A)
{
    mat33 out;
    out.m[0][0] = A->m[0][0];
    out.m[0][1] = A->m[0][1];
    out.m[0][2] = A->m[0][2];
    out.m[1][0] = A->m[1][0];
    out.m[1][1] = A->m[1][1];
    out.m[1][2] = A->m[1][2];
    out.m[2][0] = A->m[2][0];
    out.m[2][1] = A->m[2][1];
    out.m[2][2] = A->m[2][2];
    return out;
}
/* *************************************************************** */
/* *************************************************************** */
mat44 reg_mat44_mul(mat44 const* A, mat44 const* B)
{
    mat44 R;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            R.m[i][j] = static_cast<float>(static_cast<double>(A->m[i][0]) * static_cast<double>(B->m[0][j]) +
                                           static_cast<double>(A->m[i][1]) * static_cast<double>(B->m[1][j]) +
                                           static_cast<double>(A->m[i][2]) * static_cast<double>(B->m[2][j]) +
                                           static_cast<double>(A->m[i][3]) * static_cast<double>(B->m[3][j]));
        }
    }
    return R;
}
/* *************************************************************** */
mat44 operator*(mat44 A, mat44 B)
{
    return reg_mat44_mul(&A, &B);
}
/* *************************************************************** */
void reg_mat33_mul(mat44 const* mat,
    float const* in,
    float *out)
{
    out[0] = static_cast<float>(
        static_cast<double>(in[0])*static_cast<double>(mat->m[0][0]) +
        static_cast<double>(in[1])*static_cast<double>(mat->m[0][1]) +
        static_cast<double>(mat->m[0][3]));
    out[1] = static_cast<float>(
        static_cast<double>(in[0])*static_cast<double>(mat->m[1][0]) +
        static_cast<double>(in[1])*static_cast<double>(mat->m[1][1]) +
        static_cast<double>(mat->m[1][3]));
    return;
}
/* *************************************************************** */
void reg_mat33_mul(mat33 const* mat,
    float const* in,
    float *out)
{
    out[0] = static_cast<float>(
        static_cast<double>(in[0])*static_cast<double>(mat->m[0][0]) +
        static_cast<double>(in[1])*static_cast<double>(mat->m[0][1]) +
        static_cast<double>(mat->m[0][2]));
    out[1] = static_cast<float>(
        static_cast<double>(in[0])*static_cast<double>(mat->m[1][0]) +
        static_cast<double>(in[1])*static_cast<double>(mat->m[1][1]) +
        static_cast<double>(mat->m[1][2]));
    return;
}
/* *************************************************************** */
mat33 reg_mat33_mul(mat33 const* A, mat33 const* B)
{
    mat33 R;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            R.m[i][j] = static_cast<float>(static_cast<double>(A->m[i][0]) * static_cast<double>(B->m[0][j]) +
                                           static_cast<double>(A->m[i][1]) * static_cast<double>(B->m[1][j]) +
                                           static_cast<double>(A->m[i][2]) * static_cast<double>(B->m[2][j]));
        }
    }
    return R;
}
/* *************************************************************** */
mat33 operator*(mat33 A, mat33 B)
{
    return reg_mat33_mul(&A, &B);
}
/* *************************************************************** */
/* *************************************************************** */
mat33 reg_mat33_add(mat33 const* A, mat33 const* B)
{
    mat33 R;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            R.m[i][j] = static_cast<float>(static_cast<double>(A->m[i][j]) + static_cast<double>(B->m[i][j]));
        }
    }
    return R;
}
/* *************************************************************** */
/* *************************************************************** */
mat33 reg_mat33_trans(mat33 A)
{
    mat33 R;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            R.m[j][i] = A.m[i][j];
        }
    }
    return R;
}
/* *************************************************************** */
/* *************************************************************** */
mat33 operator+(mat33 A, mat33 B)
{
    return reg_mat33_add(&A, &B);
}
/* *************************************************************** */
/* *************************************************************** */
mat44 reg_mat44_add(mat44 const* A, mat44 const* B)
{
    mat44 R;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            R.m[i][j] = static_cast<float>(static_cast<double>(A->m[i][j]) + static_cast<double>(B->m[i][j]));
        }
    }
    return R;
}
/* *************************************************************** */
/* *************************************************************** */
mat44 operator+(mat44 A, mat44 B)
{
    return reg_mat44_add(&A, &B);
}
/* *************************************************************** */
/* *************************************************************** */
mat33 reg_mat33_minus(mat33 const* A, mat33 const* B)
{
    mat33 R;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            R.m[i][j] = static_cast<float>(static_cast<double>(A->m[i][j]) - static_cast<double>(B->m[i][j]));
        }
    }
    return R;
}
/* *************************************************************** */
/* *************************************************************** */
void reg_mat33_diagonalize(mat33 const* A, mat33 * Q, mat33 * D)
{
    // A must be a symmetric matrix.
    // returns Q and D such that
    // Diagonal matrix D = QT * A * Q;  and  A = Q*D*QT
    const int maxsteps = 24;  // certainly wont need that many.
    int k0, k1, k2;
    float o[3], m[3];
    float q[4] = { 0.0, 0.0, 0.0, 1.0 };
    float jr[4];
    float sqw, sqx, sqy, sqz;
    float tmp1, tmp2, mq;
    mat33 AQ;
    float thet, sgn, t, c;
    for (int i = 0; i < maxsteps; ++i)
    {
        // quat to matrix
        sqx = q[0] * q[0];
        sqy = q[1] * q[1];
        sqz = q[2] * q[2];
        sqw = q[3] * q[3];
        Q->m[0][0] = (sqx - sqy - sqz + sqw);
        Q->m[1][1] = (-sqx + sqy - sqz + sqw);
        Q->m[2][2] = (-sqx - sqy + sqz + sqw);
        tmp1 = q[0] * q[1];
        tmp2 = q[2] * q[3];
        Q->m[1][0] = 2.0 * (tmp1 + tmp2);
        Q->m[0][1] = 2.0 * (tmp1 - tmp2);
        tmp1 = q[0] * q[2];
        tmp2 = q[1] * q[3];
        Q->m[2][0] = 2.0 * (tmp1 - tmp2);
        Q->m[0][2] = 2.0 * (tmp1 + tmp2);
        tmp1 = q[1] * q[2];
        tmp2 = q[0] * q[3];
        Q->m[2][1] = 2.0 * (tmp1 + tmp2);
        Q->m[1][2] = 2.0 * (tmp1 - tmp2);

        // AQ = A * Q
        AQ.m[0][0] = Q->m[0][0] * A->m[0][0] + Q->m[1][0] * A->m[0][1] + Q->m[2][0] * A->m[0][2];
        AQ.m[0][1] = Q->m[0][1] * A->m[0][0] + Q->m[1][1] * A->m[0][1] + Q->m[2][1] * A->m[0][2];
        AQ.m[0][2] = Q->m[0][2] * A->m[0][0] + Q->m[1][2] * A->m[0][1] + Q->m[2][2] * A->m[0][2];
        AQ.m[1][0] = Q->m[0][0] * A->m[0][1] + Q->m[1][0] * A->m[1][1] + Q->m[2][0] * A->m[1][2];
        AQ.m[1][1] = Q->m[0][1] * A->m[0][1] + Q->m[1][1] * A->m[1][1] + Q->m[2][1] * A->m[1][2];
        AQ.m[1][2] = Q->m[0][2] * A->m[0][1] + Q->m[1][2] * A->m[1][1] + Q->m[2][2] * A->m[1][2];
        AQ.m[2][0] = Q->m[0][0] * A->m[0][2] + Q->m[1][0] * A->m[1][2] + Q->m[2][0] * A->m[2][2];
        AQ.m[2][1] = Q->m[0][1] * A->m[0][2] + Q->m[1][1] * A->m[1][2] + Q->m[2][1] * A->m[2][2];
        AQ.m[2][2] = Q->m[0][2] * A->m[0][2] + Q->m[1][2] * A->m[1][2] + Q->m[2][2] * A->m[2][2];
        // D = Qt * AQ
        D->m[0][0] = AQ.m[0][0] * Q->m[0][0] + AQ.m[1][0] * Q->m[1][0] + AQ.m[2][0] * Q->m[2][0];
        D->m[0][1] = AQ.m[0][0] * Q->m[0][1] + AQ.m[1][0] * Q->m[1][1] + AQ.m[2][0] * Q->m[2][1];
        D->m[0][2] = AQ.m[0][0] * Q->m[0][2] + AQ.m[1][0] * Q->m[1][2] + AQ.m[2][0] * Q->m[2][2];
        D->m[1][0] = AQ.m[0][1] * Q->m[0][0] + AQ.m[1][1] * Q->m[1][0] + AQ.m[2][1] * Q->m[2][0];
        D->m[1][1] = AQ.m[0][1] * Q->m[0][1] + AQ.m[1][1] * Q->m[1][1] + AQ.m[2][1] * Q->m[2][1];
        D->m[1][2] = AQ.m[0][1] * Q->m[0][2] + AQ.m[1][1] * Q->m[1][2] + AQ.m[2][1] * Q->m[2][2];
        D->m[2][0] = AQ.m[0][2] * Q->m[0][0] + AQ.m[1][2] * Q->m[1][0] + AQ.m[2][2] * Q->m[2][0];
        D->m[2][1] = AQ.m[0][2] * Q->m[0][1] + AQ.m[1][2] * Q->m[1][1] + AQ.m[2][2] * Q->m[2][1];
        D->m[2][2] = AQ.m[0][2] * Q->m[0][2] + AQ.m[1][2] * Q->m[1][2] + AQ.m[2][2] * Q->m[2][2];
        o[0] = D->m[1][2];
        o[1] = D->m[0][2];
        o[2] = D->m[0][1];
        m[0] = fabs(o[0]);
        m[1] = fabs(o[1]);
        m[2] = fabs(o[2]);

        k0 = (m[0] > m[1] && m[0] > m[2]) ? 0 : (m[1] > m[2]) ? 1 : 2; // index of largest element of offdiag
        k1 = (k0 + 1) % 3;
        k2 = (k0 + 2) % 3;
        if (o[k0] == 0.0)
        {
            break;                          // diagonal already
        }
        thet = (D->m[k2][k2] - D->m[k1][k1]) / (2.0*o[k0]);
        sgn = (thet > 0.0) ? 1.0 : -1.0;
        thet *= sgn;                      // make it positive
        t = sgn / (thet + ((thet < 1.E6) ? sqrt(thet*thet + 1.0) : thet)); // sign(T)/(|T|+sqrt(T^2+1))
        c = 1.0 / sqrt(t*t + 1.0);        //  c= 1/(t^2+1) , t=s/c
        if (c == 1.0)
        {
            break;                          // no room for improvement - reached machine precision.
        }
        jr[0] = jr[1] = jr[2] = jr[3] = 0.0;
        jr[k0] = sgn*sqrt((1.0 - c) / 2.0);    // using 1/2 angle identity sin(a/2) = sqrt((1-cos(a))/2)
        jr[k0] *= -1.0;                     // since our quat-to-matrix convention was for v*M instead of M*v
        jr[3] = sqrt(1.0f - jr[k0] * jr[k0]);
        if (jr[3] == 1.0)
        {
            break;                          // reached limits of floating point precision
        }
        q[0] = (q[3] * jr[0] + q[0] * jr[3] + q[1] * jr[2] - q[2] * jr[1]);
        q[1] = (q[3] * jr[1] - q[0] * jr[2] + q[1] * jr[3] + q[2] * jr[0]);
        q[2] = (q[3] * jr[2] + q[0] * jr[1] - q[1] * jr[0] + q[2] * jr[3]);
        q[3] = (q[3] * jr[3] - q[0] * jr[0] - q[1] * jr[1] - q[2] * jr[2]);
        mq = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        q[0] /= mq;
        q[1] /= mq;
        q[2] /= mq;
        q[3] /= mq;
    }
}

/* *************************************************************** */
/* *************************************************************** */
mat33 operator-(mat33 A, mat33 B)
{
    return reg_mat33_minus(&A, &B);
}
/* *************************************************************** */
/* *************************************************************** */
void reg_mat33_eye(mat33 *mat)
{
    mat->m[0][0] = 1.f;
    mat->m[0][1] = mat->m[0][2] = 0.f;
    mat->m[1][1] = 1.f;
    mat->m[1][0] = mat->m[1][2] = 0.f;
    mat->m[2][2] = 1.f;
    mat->m[2][0] = mat->m[2][1] = 0.f;
}
/* *************************************************************** */
/* *************************************************************** */
mat44 reg_mat44_minus(mat44 const* A, mat44 const* B)
{
    mat44 R;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            R.m[i][j] = static_cast<float>(static_cast<double>(A->m[i][j]) - static_cast<double>(B->m[i][j]));
        }
    }
    return R;
}

/* *************************************************************** */
/* *************************************************************** */
mat44 operator-(mat44 A, mat44 B)
{
    return reg_mat44_minus(&A, &B);
}

/* *************************************************************** */
/* *************************************************************** */
void reg_mat44_eye(mat44 *mat)
{
    mat->m[0][0] = 1.f;
    mat->m[0][1] = mat->m[0][2] = mat->m[0][3] = 0.f;
    mat->m[1][1] = 1.f;
    mat->m[1][0] = mat->m[1][2] = mat->m[1][3] = 0.f;
    mat->m[2][2] = 1.f;
    mat->m[2][0] = mat->m[2][1] = mat->m[2][3] = 0.f;
    mat->m[3][3] = 1.f;
    mat->m[3][0] = mat->m[3][1] = mat->m[3][2] = 0.f;
}
/* *************************************************************** */
/* *************************************************************** */
float reg_mat44_norm_inf(mat44 const* mat)
{
    float maxval = 0.0;
    float newval = 0.0;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            newval = fabsf(mat->m[i][j]);
            maxval = (newval > maxval) ? newval : maxval;
        }
    }
    return maxval;
}
/* *************************************************************** */
/* *************************************************************** */
void reg_mat44_mul(mat44 const* mat,
    float const* in,
    float *out)
{
    out[0] = static_cast<float>(static_cast<double>(mat->m[0][0]) * static_cast<double>(in[0]) +
        static_cast<double>(mat->m[0][1]) * static_cast<double>(in[1]) +
        static_cast<double>(mat->m[0][2]) * static_cast<double>(in[2]) +
        static_cast<double>(mat->m[0][3]));
    out[1] = static_cast<float>(static_cast<double>(mat->m[1][0]) * static_cast<double>(in[0]) +
        static_cast<double>(mat->m[1][1]) * static_cast<double>(in[1]) +
        static_cast<double>(mat->m[1][2]) * static_cast<double>(in[2]) +
        static_cast<double>(mat->m[1][3]));
    out[2] = static_cast<float>(static_cast<double>(mat->m[2][0]) * static_cast<double>(in[0]) +
        static_cast<double>(mat->m[2][1]) * static_cast<double>(in[1]) +
        static_cast<double>(mat->m[2][2]) * static_cast<double>(in[2]) +
        static_cast<double>(mat->m[2][3]));
}
/* *************************************************************** */
/* *************************************************************** */
void reg_mat44_mul(mat44 const* mat,
    double const* in,
    double *out)
{
    double matD[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            matD[i][j] = static_cast<double>(mat->m[i][j]);

    out[0] = matD[0][0] * in[0] +
        matD[0][1] * in[1] +
        matD[0][2] * in[2] +
        matD[0][3];
    out[1] = matD[1][0] * in[0] +
        matD[1][1] * in[1] +
        matD[1][2] * in[2] +
        matD[1][3];
    out[2] = matD[2][0] * in[0] +
        matD[2][1] * in[1] +
        matD[2][2] * in[2] +
        matD[2][3];
    return;
}
/* *************************************************************** */
/* *************************************************************** */
mat44 reg_mat44_mul(mat44 const* A, double scalar)
{
    mat44 out;
    out.m[0][0] = A->m[0][0] * scalar;
    out.m[0][1] = A->m[0][1] * scalar;
    out.m[0][2] = A->m[0][2] * scalar;
    out.m[0][3] = A->m[0][3] * scalar;
    out.m[1][0] = A->m[1][0] * scalar;
    out.m[1][1] = A->m[1][1] * scalar;
    out.m[1][2] = A->m[1][2] * scalar;
    out.m[1][3] = A->m[1][3] * scalar;
    out.m[2][0] = A->m[2][0] * scalar;
    out.m[2][1] = A->m[2][1] * scalar;
    out.m[2][2] = A->m[2][2] * scalar;
    out.m[2][3] = A->m[2][3] * scalar;
    out.m[3][0] = A->m[3][0] * scalar;
    out.m[3][1] = A->m[3][1] * scalar;
    out.m[3][2] = A->m[3][2] * scalar;
    out.m[3][3] = A->m[3][3] * scalar;
    return out;
}
/* *************************************************************** */
void reg_mat44_disp(mat44 *mat, char * title){
    printf("%s:\n%.7g\t%.7g\t%.7g\t%.7g\n%.7g\t%.7g\t%.7g\t%.7g\n%.7g\t%.7g\t%.7g\t%.7g\n%.7g\t%.7g\t%.7g\t%.7g\n", title,
        mat->m[0][0], mat->m[0][1], mat->m[0][2], mat->m[0][3],
        mat->m[1][0], mat->m[1][1], mat->m[1][2], mat->m[1][3],
        mat->m[2][0], mat->m[2][1], mat->m[2][2], mat->m[2][3],
        mat->m[3][0], mat->m[3][1], mat->m[3][2], mat->m[3][3]);
}

/* *************************************************************** */
/* *************************************************************** */
void reg_mat33_disp(mat33 *mat, char * title){
    printf("%s:\n%g\t%g\t%g\n%g\t%g\t%g\n%g\t%g\t%g\n", title,
        mat->m[0][0], mat->m[0][1], mat->m[0][2],
        mat->m[1][0], mat->m[1][1], mat->m[1][2],
        mat->m[2][0], mat->m[2][1], mat->m[2][2]);
}
/* *************************************************************** */
//is it square distance or just distance?
// Helper function: Get the square of the Euclidean distance
double get_square_distance3D(float * first_point3D, float * second_point3D) {
    return sqrt(reg_pow2(first_point3D[0] - second_point3D[0]) +
          reg_pow2(first_point3D[1] - second_point3D[1]) +
          reg_pow2(first_point3D[2] - second_point3D[2]));
}
/* *************************************************************** */
//is it square distance or just distance?
double get_square_distance2D(float * first_point2D, float * second_point2D) {
    return sqrt(reg_pow2(first_point2D[0] - second_point2D[0]) +
          reg_pow2(first_point2D[1] - second_point2D[1]));
}
/* *************************************************************** */
// Calculate pythagorean distance
template<class T>
T pythag(T a, T b)
{
    T absa, absb;
    absa = fabs(a);
    absb = fabs(b);

    if (absa > absb)
        return (T)(absa * sqrt(1.0f + SQR(absb / absa)));
    else
        return (absb == 0.0f ? 0.0f : (T)(absb * sqrt(1.0f + SQR(absa / absb))));
}
#endif // _REG_MATHS_CPP
