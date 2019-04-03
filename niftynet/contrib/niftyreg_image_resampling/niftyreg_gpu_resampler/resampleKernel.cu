#include <stdio.h>
#include <assert.h>
#include "cuda_runtime.h"
#include "cuda.h"
#include"_reg_resampling.h"
#include"_reg_maths.h"
#include "resampleKernel.h"
#include "_reg_common_cuda.h"
#include"_reg_tools.h"
#include "interpolations.h"

#define SINC_KERNEL_RADIUS 3
#define SINC_KERNEL_SIZE SINC_KERNEL_RADIUS*2

/* *************************************************************** */
unsigned int min1(unsigned int a, unsigned int b)
{
  return (a < b) ? a : b;
}
/* *************************************************************** */
template<class DTYPE>
__device__ __inline__ void reg_mat44_mul_cuda(DTYPE const* mat, DTYPE const* in, DTYPE *out)
{
  out[0] = (DTYPE)((double)mat[0 * 4 + 0] * (double)in[0] + (double)mat[0 * 4 + 1] * (double)in[1] + (double)mat[0 * 4 + 2] * (double)in[2] + (double)mat[0 * 4 + 3]);
  out[1] = (DTYPE)((double)mat[1 * 4 + 0] * (double)in[0] + (double)mat[1 * 4 + 1] * (double)in[1] + (double)mat[1 * 4 + 2] * (double)in[2] + (double)mat[1 * 4 + 3]);
  out[2] = (DTYPE)((double)mat[2 * 4 + 0] * (double)in[0] + (double)mat[2 * 4 + 1] * (double)in[1] + (double)mat[2 * 4 + 2] * (double)in[2] + (double)mat[2 * 4 + 3]);
  return;
}
/* *************************************************************** */
template<class DTYPE>
__device__ __inline__ void reg_mat44_mul_cuda(float* mat, DTYPE const* in, DTYPE *out)
{
  out[0] = (DTYPE)((double)mat[0 * 4 + 0] * (double)in[0] + (double)mat[0 * 4 + 1] * (double)in[1] + (double)mat[0 * 4 + 2] * (double)in[2] + (double)mat[0 * 4 + 3]);
  out[1] = (DTYPE)((double)mat[1 * 4 + 0] * (double)in[0] + (double)mat[1 * 4 + 1] * (double)in[1] + (double)mat[1 * 4 + 2] * (double)in[2] + (double)mat[1 * 4 + 3]);
  out[2] = (DTYPE)((double)mat[2 * 4 + 0] * (double)in[0] + (double)mat[2 * 4 + 1] * (double)in[1] + (double)mat[2 * 4 + 2] * (double)in[2] + (double)mat[2 * 4 + 3]);
  return;
}
/* *************************************************************** */
__device__ __inline__ int cuda_reg_floor(double a)
{
  return (int) (floor(a));
}
/* *************************************************************** */
template<class FieldTYPE>
__device__ __inline__ void interpolantCubicSpline(FieldTYPE ratio, FieldTYPE *basis)
{
  if (ratio < 0.0)
    ratio = 0.0; //reg_rounding error
  double FF = (double) ratio * ratio;
  basis[0] = (FieldTYPE) ((ratio * (((double)2.0 - ratio) * ratio - (double)1.0)) / (double)2.0);
  basis[1] = (FieldTYPE) ((FF * ((double)3.0 * ratio - 5.0) + 2.0) / (double)2.0);
  basis[2] = (FieldTYPE) ((ratio * (((double)4.0 - (double)3.0 * ratio) * ratio + (double)1.0)) / (double)2.0);
  basis[3] = (FieldTYPE) ((ratio - (double)1.0) * FF / (double)2.0);
}
/* *************************************************************** */
__inline__ __device__ void interpWindowedSincKernel(double relative, double *basis)
{
  if (relative < 0.0)
    relative = 0.0; //reg_rounding error
  int j = 0;
  double sum = 0.;
  for (int i = -SINC_KERNEL_RADIUS; i < SINC_KERNEL_RADIUS; ++i) {
    double x = relative - (double) (i);
    if (x == 0.0)
      basis[j] = 1.0;
    else if (abs(x) >= (double) (SINC_KERNEL_RADIUS))
      basis[j] = 0;
    else {
      double pi_x = M_PI * x;
      basis[j] = (SINC_KERNEL_RADIUS) * sin(pi_x) * sin(pi_x / SINC_KERNEL_RADIUS) / (pi_x * pi_x);
    }
    sum += basis[j];
    j++;
  }
  for (int i = 0; i < SINC_KERNEL_SIZE; ++i)
    basis[i] /= sum;
}
/* *************************************************************** */
__inline__ __device__ void interpCubicSplineKernel(double relative, double *basis)
{
  // if (relative < 0.0)
  //   relative = 0.0; //reg_rounding error
  double FF = relative * relative;
  basis[0] = (relative * ((2.0 - relative) * relative - 1.0)) / 2.0;
  basis[1] = (FF * (3.0 * relative - 5.0) + 2.0) / 2.0;
  basis[2] = (relative * ((4.0 - 3.0 * relative) * relative + 1.0)) / 2.0;
  basis[3] = (relative - 1.0) * FF / 2.0;
}
/* *************************************************************** */
__inline__ __device__ void interpLinearKernel(double relative, double *basis)
{
  if (relative < 0.0)
    relative = 0.0; //reg_rounding error
  basis[1] = relative;
  basis[0] = 1.0 - relative;
}
/* *************************************************************** */
__inline__ __device__ void interpNearestNeighKernel(double relative, double *basis)
{
  if (relative < 0.0)
    relative = 0.0; //reg_rounding error
  basis[0] = basis[1] = 0.0;
  if (relative >= 0.5)
    basis[1] = 1;
  else
    basis[0] = 1;
}
/* *************************************************************** */
__inline__ __device__ double interpLoop2D(const float* floatingIntensity,
                                          double* xBasis,
                                          double* yBasis,
                                          double* zBasis,
                                          int *previous,
                                          uint3 fi_xyz,
                                          const float paddingValue,
                                          const unsigned int kernel_size)
{
  double intensity = (double)(0.0);

  for (int b = 0; b < kernel_size; b++) {
    int Y = previous[1] + b;
    bool yInBounds = -1 < Y && Y < fi_xyz.y;
    double xTempNewValue = 0.0;

    for (int a = 0; a < kernel_size; a++) {
      int X = previous[0] + a;
      bool xInBounds = -1 < X && X < fi_xyz.x;

      const unsigned int idx = Y * fi_xyz.x + X;

      xTempNewValue += (xInBounds && yInBounds) ? floatingIntensity[idx] * xBasis[a] : paddingValue * xBasis[a];
    }
    intensity += xTempNewValue * yBasis[b];
  }
  return intensity;
}
/* *************************************************************** */
template <const bool tDoClamp>
__inline__ __device__ double interpLoop2DBoundary(const float* floatingIntensity,
                                                  double* xBasis,
                                                  double* yBasis,
                                                  double* zBasis,
                                                  int *previous,
                                                  uint3 fi_xyz,
                                                  const unsigned int kernel_size)
{
  double intensity = (double)(0.0);

  for (int b = 0; b < kernel_size; b++) {
    const int offset_x = reg_applyBoundary<tDoClamp, !tDoClamp>(previous[1] + b, fi_xyz.y)*fi_xyz.x;

    double xTempNewValue = 0.0;

    for (int a = 0; a < kernel_size; a++) {
      const unsigned int idx = offset_x + reg_applyBoundary<tDoClamp, !tDoClamp>(previous[0] + a, fi_xyz.x);

      xTempNewValue += floatingIntensity[idx]*xBasis[a];
    }
    intensity += xTempNewValue*yBasis[b];
  }

  return intensity;
}
/* *************************************************************** */
__inline__ __device__ double interpLoop3D(const float* floatingIntensity,
                                          double* xBasis,
                                          double* yBasis,
                                          double* zBasis,
                                          int *previous,
                                          uint3 fi_xyz,
                                          float paddingValue,
                                          unsigned int kernel_size)
{
  double intensity = (double)(0.0);
  for (int c = 0; c < kernel_size; c++) {
    int Z = previous[2] + c;
    bool zInBounds = -1 < Z && Z < fi_xyz.z;
    double yTempNewValue = 0.0;
    for (int b = 0; b < kernel_size; b++) {
      int Y = previous[1] + b;
      bool yInBounds = -1 < Y && Y < fi_xyz.y;
      double xTempNewValue = 0.0;
      for (int a = 0; a < kernel_size; a++) {
        int X = previous[0] + a;
        bool xInBounds = -1 < X && X < fi_xyz.x;
        const unsigned int idx = Z * fi_xyz.x * fi_xyz.y + Y * fi_xyz.x + X;

        xTempNewValue += (xInBounds && yInBounds && zInBounds) ? floatingIntensity[idx] * xBasis[a] : paddingValue * xBasis[a];
      }
      yTempNewValue += xTempNewValue * yBasis[b];
    }
    intensity += yTempNewValue * zBasis[c];
  }
  return intensity;
}
/* *************************************************************** */
template <const bool tDoClamp>
__inline__ __device__ double interpLoop3DBoundary(const float* floatingIntensity,
                                                  double* xBasis,
                                                  double* yBasis,
                                                  double* zBasis,
                                                  int *previous,
                                                  uint3 fi_xyz,
                                                  unsigned int kernel_size)
{
  double intensity = (double)(0.0);

  for (int c = 0; c < kernel_size; c++) {
    const int offset_y = reg_applyBoundary<tDoClamp, !tDoClamp>(previous[2] + c, fi_xyz.z)*fi_xyz.y;

    double yTempNewValue = 0.0;

    for (int b = 0; b < kernel_size; b++) {
      const int offset_x = (offset_y + reg_applyBoundary<tDoClamp, !tDoClamp>(previous[1] + b, fi_xyz.y))*fi_xyz.x;

      double xTempNewValue = 0.0;

      for (int a = 0; a < kernel_size; a++) {
        const unsigned int idx = offset_x + reg_applyBoundary<tDoClamp, !tDoClamp>(previous[0] + a, fi_xyz.x);

        xTempNewValue += floatingIntensity[idx]*xBasis[a];
      }
      yTempNewValue += xTempNewValue*yBasis[b];
    }
    intensity += yTempNewValue*zBasis[c];
  }

  return intensity;
}
/* *************************************************************** */
template <const bool tDoClamp, const bool tDoReflect>
__global__ void ResampleImage2D(const float* floatingImage,
                                const float* deformationField,
                                float* warpedImage,
                                ulong2 voxelNumber,
                                uint3 fi_xyz,
                                uint2 wi_tu,
                                const float paddingValue,
                                const int kernelType)
{
  const float *sourceIntensityPtr = (floatingImage);
  float *resultIntensityPtr = (warpedImage);
  const float *deformationFieldPtrX = (deformationField);
  const float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber.x];

  long index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < voxelNumber.x) {

    for (unsigned int t = 0; t < wi_tu.x * wi_tu.y; t++) {

      float *resultIntensity = &resultIntensityPtr[t * voxelNumber.x];
      const float *floatingIntensity = &sourceIntensityPtr[t * voxelNumber.y];
      double intensity = paddingValue;

      int previous[3];
      float position[3];
      double relative[3];

      position[0] = (float)(deformationFieldPtrX[index]);
      position[1] = (float)(deformationFieldPtrY[index]);

      previous[0] = cuda_reg_floor(position[0]);
      previous[1] = cuda_reg_floor(position[1]);

      relative[0] = (double)(position[0]) - (double)(previous[0]);
      relative[1] = (double)(position[1]) - (double)(previous[1]);

      if (kernelType == 0) {

        double xBasisIn[2], yBasisIn[2], zBasisIn[2];
        interpNearestNeighKernel(relative[0], xBasisIn);
        interpNearestNeighKernel(relative[1], yBasisIn);
        if (tDoClamp || tDoReflect) {
          intensity = interpLoop2DBoundary<tDoClamp>(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, 2);
        } else {
          intensity = interpLoop2D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 2);
        }
      }
      else if (kernelType == 1) {

        double xBasisIn[2], yBasisIn[2], zBasisIn[2];
        interpLinearKernel(relative[0], xBasisIn);
        interpLinearKernel(relative[1], yBasisIn);
        if (tDoClamp || tDoReflect) {
          intensity = interpLoop2DBoundary<tDoClamp>(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, 2);
        } else {
          intensity = interpLoop2D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 2);
        }
      }
      else if (kernelType == 4) {

        double xBasisIn[6], yBasisIn[6], zBasisIn[6];

        previous[0] -= SINC_KERNEL_RADIUS;
        previous[1] -= SINC_KERNEL_RADIUS;
        previous[2] -= SINC_KERNEL_RADIUS;

        interpWindowedSincKernel(relative[0], xBasisIn);
        interpWindowedSincKernel(relative[1], yBasisIn);
        if (tDoClamp || tDoReflect) {
          intensity = interpLoop2DBoundary<tDoClamp>(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, 6);
        } else {
          intensity = interpLoop2D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 6);
        }
      }
      else {

        double xBasisIn[4], yBasisIn[4], zBasisIn[4];

        previous[0]--;
        previous[1]--;
        previous[2]--;

        reg_getNiftynetCubicSpline(relative[0], xBasisIn);
        reg_getNiftynetCubicSpline(relative[1], yBasisIn);
        if (tDoClamp || tDoReflect) {
          intensity = interpLoop2DBoundary<tDoClamp>(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, 4);
        } else {
          intensity = interpLoop2D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 4);
        }
      }

      resultIntensity[index] = (float)intensity;
    }
    index += blockDim.x * gridDim.x;
  }
}
/* *************************************************************** */
template <const bool tDoClamp, const bool tDoReflect>
__global__ void ResampleImage3D(const float* floatingImage,
                                const float* deformationField,
                                float* warpedImage,
                                const ulong2 voxelNumber,
                                uint3 fi_xyz,
                                uint2 wi_tu,
                                const float paddingValue,
                                int kernelType)
{
  const float *sourceIntensityPtr = (floatingImage);
  float *resultIntensityPtr = (warpedImage);
  const float *deformationFieldPtrX = (deformationField);
  const float *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber.x];
  const float *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber.x];

  long index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < voxelNumber.x) {

    for (unsigned int t = 0; t < wi_tu.x * wi_tu.y; t++) {

      float *resultIntensity = &resultIntensityPtr[t * voxelNumber.x];
      const float *floatingIntensity = &sourceIntensityPtr[t * voxelNumber.y];
      double intensity = paddingValue;

      int previous[3];
      float position[3];
      double relative[3];

      position[0] = (float) (deformationFieldPtrX[index]);
      position[1] = (float) (deformationFieldPtrY[index]);
      position[2] = (float) (deformationFieldPtrZ[index]);

      previous[0] = cuda_reg_floor(position[0]);
      previous[1] = cuda_reg_floor(position[1]);
      previous[2] = cuda_reg_floor(position[2]);

      relative[0] = (double)(position[0]) - (double)(previous[0]);
      relative[1] = (double)(position[1]) - (double)(previous[1]);
      relative[2] = (double)(position[2]) - (double)(previous[2]);

      if (kernelType == 0) {

        double xBasisIn[2], yBasisIn[2], zBasisIn[2];
        interpNearestNeighKernel(relative[0], xBasisIn);
        interpNearestNeighKernel(relative[1], yBasisIn);
        interpNearestNeighKernel(relative[2], zBasisIn);
        if (tDoClamp || tDoReflect) {
          intensity = interpLoop3DBoundary<tDoClamp>(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, 2);
        } else {
          intensity = interpLoop3D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 2);
        }
      } else if (kernelType == 1) {

        double xBasisIn[2], yBasisIn[2], zBasisIn[2];
        interpLinearKernel(relative[0], xBasisIn);
        interpLinearKernel(relative[1], yBasisIn);
        interpLinearKernel(relative[2], zBasisIn);
        if (tDoClamp || tDoReflect) {
          intensity = interpLoop3DBoundary<tDoClamp>(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, 2);
        } else {
          intensity = interpLoop3D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 2);
        }
      } else if (kernelType == 4) {

        double xBasisIn[6], yBasisIn[6], zBasisIn[6];

        previous[0] -= SINC_KERNEL_RADIUS;
        previous[1] -= SINC_KERNEL_RADIUS;
        previous[2] -= SINC_KERNEL_RADIUS;

        interpWindowedSincKernel(relative[0], xBasisIn);
        interpWindowedSincKernel(relative[1], yBasisIn);
        interpWindowedSincKernel(relative[2], zBasisIn);
        if (tDoClamp || tDoReflect) {
          intensity = interpLoop3DBoundary<tDoClamp>(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, 6);
        } else {
          intensity = interpLoop3D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 6);
        }
      } else {

        double xBasisIn[4], yBasisIn[4], zBasisIn[4];

        previous[0]--;
        previous[1]--;
        previous[2]--;

        reg_getNiftynetCubicSpline(relative[0], xBasisIn);
        reg_getNiftynetCubicSpline(relative[1], yBasisIn);
        reg_getNiftynetCubicSpline(relative[2], zBasisIn);
        if (tDoClamp || tDoReflect) {
          intensity = interpLoop3DBoundary<tDoClamp>(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, 4);
        } else {
          intensity = interpLoop3D(floatingIntensity, xBasisIn, yBasisIn, zBasisIn, previous, fi_xyz, paddingValue, 4);
        }
      }
      resultIntensity[index] = (float)intensity;
    }
    index += blockDim.x * gridDim.x;
  }
}
/* *************************************************************** */
void launchResample(const nifti_image *floatingImage,
                    const nifti_image *warpedImage,
                    const int interp,
                    const resampler_boundary_e boundary,
                    const float *floatingImage_d,
                    float *warpedImage_d,
                    const float *deformationFieldImage_d) {
  const float paddingValue = get_padding_value<float>(boundary);


  long targetVoxelNumber = (long) warpedImage->nx * warpedImage->ny * warpedImage->nz;
  ulong2 voxelNumber = make_ulong2(warpedImage->nx * warpedImage->ny * warpedImage->nz, floatingImage->nx * floatingImage->ny * floatingImage->nz);
  dim3 mygrid;
  dim3 myblocks;
  uint3 fi_xyz = make_uint3(floatingImage->nx, floatingImage->ny, floatingImage->nz);
  uint2 wi_tu = make_uint2(warpedImage->nt, warpedImage->nu);

  cudaCommon_computeGridConfiguration(myblocks, mygrid, targetVoxelNumber);
  if (floatingImage->nz > 1) {
    switch (boundary) {
    case resampler_boundary_e::CLAMPING:
      ResampleImage3D<true, false> <<<mygrid, myblocks >>>(floatingImage_d,
                                                           deformationFieldImage_d,
                                                           warpedImage_d,
                                                           voxelNumber,
                                                           fi_xyz,
                                                           wi_tu,
                                                           paddingValue,
                                                           interp);
      break;

    case resampler_boundary_e::REFLECTING:
      ResampleImage3D<false, true> <<<mygrid, myblocks >>>(floatingImage_d,
                                                           deformationFieldImage_d,
                                                           warpedImage_d,
                                                           voxelNumber,
                                                           fi_xyz,
                                                           wi_tu,
                                                           paddingValue,
                                                           interp);
      break;

    default:
      ResampleImage3D<false, false> <<<mygrid, myblocks >>>(floatingImage_d,
                                                     deformationFieldImage_d,
                                                     warpedImage_d,
                                                     voxelNumber,
                                                     fi_xyz,
                                                     wi_tu,
                                                     paddingValue,
                                                     interp);
    }
  } else{
    switch (boundary) {
    case resampler_boundary_e::CLAMPING:
      ResampleImage2D<true, false> <<<mygrid, myblocks >>>(floatingImage_d,
                                                           deformationFieldImage_d,
                                                           warpedImage_d,
                                                           voxelNumber,
                                                           fi_xyz,
                                                           wi_tu,
                                                           paddingValue,
                                                           interp);
      break;

    case resampler_boundary_e::REFLECTING:
      ResampleImage2D<false, true> <<<mygrid, myblocks >>>(floatingImage_d,
                                                           deformationFieldImage_d,
                                                           warpedImage_d,
                                                           voxelNumber,
                                                           fi_xyz,
                                                           wi_tu,
                                                           paddingValue,
                                                           interp);
      break;


    default:
      ResampleImage2D<false, false> <<<mygrid, myblocks >>>(floatingImage_d,
                                                            deformationFieldImage_d,
                                                            warpedImage_d,
                                                            voxelNumber,
                                                            fi_xyz,
                                                            wi_tu,
                                                            paddingValue,
                                                            interp);
    }
  }
#ifndef NDEBUG
  NR_CUDA_CHECK_KERNEL(mygrid, myblocks)
#else
    NR_CUDA_SAFE_CALL(cudaThreadSynchronize());
#endif
}
/* *************************************************************** */
__host__ nifti_image* resample(nifti_image &r_displacements, const nifti_image &floating, const int interp_code, const resampler_boundary_e boundary, const bool is_displacement_argument) {
  nifti_image *p_warped = nifti_copy_nim_info(&floating);

  assert(floating.dim[1] == r_displacements.dim[1] && floating.dim[2] == r_displacements.dim[2]
         && (floating.dim[3] == r_displacements.dim[3] || floating.ndim == 2));

  if (is_displacement_argument) {
    reg_getDeformationFromDisplacement(&r_displacements);
  }

  {
    float *dp_floating;
    float *dp_warped;
    float *dp_deformation;

    assert(floating.datatype == NIFTI_TYPE_FLOAT32);
    cudaCommon_allocateArrayToDevice(&dp_floating, floating.nvox);
    cudaCommon_allocateArrayToDevice(&dp_warped, floating.nvox);
    cudaCommon_allocateArrayToDevice(&dp_deformation, r_displacements.nvox);

    cudaCommon_transferNiftiToArrayOnDevice(&dp_floating, &floating);
    cudaCommon_transferNiftiToArrayOnDevice(&dp_deformation, &r_displacements);
    launchResample(&floating, p_warped, interp_code, boundary, dp_floating, dp_warped, dp_deformation);
    p_warped->data = std::malloc(p_warped->nvox*sizeof(float));
    cudaCommon_transferFromDeviceToNifti(p_warped, &dp_warped);

    cudaFree(dp_floating);
    cudaFree(dp_warped);
    cudaFree(dp_deformation);
  }

  return p_warped;
}
/* *************************************************************** */
