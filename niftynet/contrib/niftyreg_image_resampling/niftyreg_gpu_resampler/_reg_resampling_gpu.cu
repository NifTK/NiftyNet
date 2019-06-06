/*
 *  _reg_resampling_gpu.cu
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_RESAMPLING_GPU_CU
#define _REG_RESAMPLING_GPU_CU

#include "_reg_resampling_gpu.h"
#include "_reg_tools.h"
#include "interpolations.h"

/* *************************************************************** */
/* *************************************************************** */
template <const bool tIs3D, const resampler_boundary_e tBoundary>
__global__ void reg_getImageGradient_spline_kernel(float *p_gradientArray,
                                                   const float *pc_floating,
                                                   const float *pc_deformation,
                                                   const int3 floating_dims,
                                                   const int3 deformation_dims,
                                                   const float paddingValue,
                                                   const int ref_size) {
  const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
  const int nof_dims = 2 + int(tIs3D);
  const int kernel_size = 4;

  if(tid<ref_size){
    //Get the voxel-based deformation in the floating space
    float voxeldeformation[nof_dims];
    int voxel[nof_dims];
    float basis[nof_dims][kernel_size];
    float derivative[nof_dims][kernel_size];

    for (int d = 0; d < nof_dims; ++d) {
      float relative;

      voxeldeformation[d] = pc_deformation[tid+d*ref_size];
      voxel[d] = int(voxeldeformation[d]);

      relative = fabsf(voxeldeformation[d] - voxel[d]);

      reg_getNiftynetCubicSpline(relative, basis[d]);
      reg_getNiftynetCubicSplineDerivative(relative, derivative[d]);

      voxel[d] -= 1;
    }

    float4 gradientValue=make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (tIs3D) {
      for(short c = 0; c < kernel_size; ++c) {
        int z = reg_applyBoundary<tBoundary>(voxel[2] + c, floating_dims.z);
        float3 tempValueY = make_float3(0.0f, 0.0f, 0.0f);

        for(short b = 0; b < kernel_size; ++b){
          float2 tempValueX = make_float2(0.0f, 0.0f);
          int y = reg_applyBoundary<tBoundary>(voxel[1] + b, floating_dims.y);

          for(short a = 0; a < kernel_size; ++a){
            int x = reg_applyBoundary<tBoundary>(voxel[0] + a, floating_dims.x);
            float intensity = paddingValue;

            if (reg_checkImageDimensionIndex<tBoundary>(x, floating_dims.x)
                && reg_checkImageDimensionIndex<tBoundary>(y, floating_dims.y)
                && reg_checkImageDimensionIndex<tBoundary>(z, floating_dims.z)) {
              intensity = pc_floating[((z*floating_dims.y)+y)*floating_dims.x+x];
            }

            tempValueX.x += intensity*derivative[0][a];
            tempValueX.y += intensity*basis[0][a];
          }
          tempValueY.x += tempValueX.x*basis[1][b];
          tempValueY.y += tempValueX.y*derivative[1][b];
          tempValueY.z += tempValueX.y*basis[1][b];
        }
        gradientValue.x += tempValueY.x*basis[2][c];
        gradientValue.y += tempValueY.y*basis[2][c];
        gradientValue.z += tempValueY.z*derivative[2][c];
      }
    } else {
      for(short b = 0; b < kernel_size; ++b){
        float2 tempValueX = make_float2(0.0f, 0.0f);
        int y = reg_applyBoundary<tBoundary>(voxel[1] + b, floating_dims.y);

        for(short a = 0; a < kernel_size; ++a){
          int x = reg_applyBoundary<tBoundary>(voxel[0] + a, floating_dims.x);
          float intensity=paddingValue;

          if (reg_checkImageDimensionIndex<tBoundary>(x, floating_dims.x)
              && reg_checkImageDimensionIndex<tBoundary>(y, floating_dims.y)) {
            intensity = pc_floating[y*floating_dims.x+x];
          }

          tempValueX.x +=  intensity*derivative[0][a];
          tempValueX.y +=  intensity*basis[0][a];
        }
        gradientValue.x += tempValueX.x*basis[1][b];
        gradientValue.y += tempValueX.y*derivative[1][b];
      }
    }

    p_gradientArray[tid] = gradientValue.x;
    p_gradientArray[ref_size+tid] = gradientValue.y;
    if (tIs3D) {
      p_gradientArray[2*ref_size+tid] = gradientValue.z;
    }
  }
}
/* *************************************************************** */
template <const bool tIs3D, const resampler_boundary_e tBoundary>
__global__ void reg_getImageGradient_kernel(float *p_gradientArray,
                                            const float *pc_floating,
                                            const float *pc_deformation,
                                            const int3 floating_dims,
                                            const int3 deformation_dims,
                                            const float paddingValue,
                                            const int ref_size)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;

    if(tid<ref_size){
        //Get the voxel-based deformation in the floating space
        float3 voxeldeformation;

        voxeldeformation.x = pc_deformation[tid];
        voxeldeformation.y = pc_deformation[ref_size+tid];
        if (tIs3D) {
          voxeldeformation.z = pc_deformation[2*ref_size+tid];
        }

        int3 voxel;
        voxel.x = (int)(voxeldeformation.x);
        voxel.y = (int)(voxeldeformation.y);
        if (tIs3D) {
          voxel.z = (int)(voxeldeformation.z);
        }

        float xBasis[2];
        float relative = fabsf(voxeldeformation.x - (float)voxel.x);
        xBasis[0]=1.0f-relative;
        xBasis[1]=relative;

        float yBasis[2];
        relative = fabsf(voxeldeformation.y - (float)voxel.y);
        yBasis[0]=1.0f-relative;
        yBasis[1]=relative;

        float zBasis[2];
        if (tIs3D) {
          relative = fabsf(voxeldeformation.z - (float)voxel.z);
          zBasis[0]=1.0f-relative;
          zBasis[1]=relative;
        }

        float4 gradientValue=make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (tIs3D) {
          for(short c=0; c<2; c++){
            int z = reg_applyBoundary<tBoundary>(voxel.z + c, floating_dims.z);

            float3 tempValueY=make_float3(0.0f, 0.0f, 0.0f);
            for(short b=0; b<2; b++){
              float2 tempValueX=make_float2(0.0f, 0.0f);
              int y = reg_applyBoundary<tBoundary>(voxel.y + b, floating_dims.y);

              for(short a=0; a<2; a++){
                int x= reg_applyBoundary<tBoundary>(voxel.x + a, floating_dims.x);
                float intensity=paddingValue;

                if (reg_checkImageDimensionIndex<tBoundary>(x, floating_dims.x)
                    && reg_checkImageDimensionIndex<tBoundary>(y, floating_dims.y)
                    && reg_checkImageDimensionIndex<tBoundary>(z, floating_dims.z)) {
                  intensity = pc_floating[((z*floating_dims.y)+y)*floating_dims.x+x];
                }

                tempValueX.x += (1 - 2*int(a == 0))*intensity;
                tempValueX.y +=  intensity * xBasis[a];
              }
              tempValueY.x += tempValueX.x * yBasis[b];
              tempValueY.y += (1 - 2*int(b == 0))*tempValueX.y;
              tempValueY.z += tempValueX.y * yBasis[b];
            }
            gradientValue.x += tempValueY.x * zBasis[c];
            gradientValue.y += tempValueY.y * zBasis[c];
            gradientValue.z += (1 - 2*int(c == 0))*tempValueY.z;
          }
        } else {
          for(short b=0; b<2; b++){
            float2 tempValueX=make_float2(0.0f, 0.0f);
            int y = reg_applyBoundary<tBoundary>(voxel.y + b, floating_dims.y);

            for(short a=0; a<2; a++){
              int x = reg_applyBoundary<tBoundary>(voxel.x + a, floating_dims.x);
              float intensity=paddingValue;

              if (reg_checkImageDimensionIndex<tBoundary>(x, floating_dims.x)
                  && reg_checkImageDimensionIndex<tBoundary>(y, floating_dims.y)) {
                intensity = pc_floating[y*floating_dims.x+x];
              }

              tempValueX.x +=  intensity*(1 - 2*(a == 0));
              tempValueX.y +=  intensity * xBasis[a];
            }
            gradientValue.x += tempValueX.x * yBasis[b];
            gradientValue.y += tempValueX.y*(1 - 2*(b == 0));
          }
        }

        p_gradientArray[tid] = gradientValue.x;
        p_gradientArray[ref_size+tid] = gradientValue.y;
        if (tIs3D) {
          p_gradientArray[2*ref_size+tid] = gradientValue.z;
        }
    }
}
/* *************************************************************** */
template <const bool tIs3D, const resampler_boundary_e tBoundary>
static void _launchGradientKernelBoundary(const nifti_image &sourceImage,
                                          const nifti_image &deformationImage,
                                          const float *sourceImageArray_d,
                                          const float *positionFieldImageArray_d,
                                          float *resultGradientArray_d,
                                          const float pad,
                                          const int interpolation) {
  int3 floatingDim = make_int3(sourceImage.nx, sourceImage.ny, sourceImage.nz);
  int3 deformationDim = make_int3(deformationImage.nx, deformationImage.ny, deformationImage.nz);
  dim3 B1;
  dim3 G1;
  int ref_size = deformationImage.nx*deformationImage.ny*deformationImage.nz;

  cudaCommon_computeGridConfiguration(B1, G1, ref_size);

  if (interpolation == 3) {
    reg_getImageGradient_spline_kernel<tIs3D, tBoundary> <<<G1, B1>>> (resultGradientArray_d,
                                                                       sourceImageArray_d,
                                                                       positionFieldImageArray_d,
                                                                       floatingDim,
                                                                       deformationDim,
                                                                       pad,
                                                                       ref_size);
  } else {
    reg_getImageGradient_kernel<tIs3D, tBoundary> <<<G1, B1>>> (resultGradientArray_d,
                                                                sourceImageArray_d,
                                                                positionFieldImageArray_d,
                                                                floatingDim,
                                                                deformationDim,
                                                                pad,
                                                                ref_size);
  }
}
/* *************************************************************** */
template <const bool tIs3D>
static void _launchGradientKernelND(const nifti_image &sourceImage,
                                    const nifti_image &deformationImage,
                                    const float *sourceImageArray_d,
                                    const float *positionFieldImageArray_d,
                                    float *resultGradientArray_d,
                                    const resampler_boundary_e boundary,
                                    const int interpolation) {
  const float pad = reg_getPaddingValue<float>(boundary);

  switch (boundary) {
  case resampler_boundary_e::CLAMPING:
    _launchGradientKernelBoundary<tIs3D, resampler_boundary_e::CLAMPING>(sourceImage,
                                                                         deformationImage,
                                                                         sourceImageArray_d,
                                                                         positionFieldImageArray_d,
                                                                         resultGradientArray_d,
                                                                         pad,
                                                                         interpolation);
    break;

  case resampler_boundary_e::REFLECTING:
    _launchGradientKernelBoundary<tIs3D, resampler_boundary_e::REFLECTING>(sourceImage,
                                                                           deformationImage,
                                                                           sourceImageArray_d,
                                                                           positionFieldImageArray_d,
                                                                           resultGradientArray_d,
                                                                           pad,
                                                                           interpolation);
    break;

  default:
    _launchGradientKernelBoundary<tIs3D, resampler_boundary_e::ZEROPAD>(sourceImage,
                                                                        deformationImage,
                                                                        sourceImageArray_d,
                                                                        positionFieldImageArray_d,
                                                                        resultGradientArray_d,
                                                                        pad,
                                                                        interpolation);
  }
}
/* *************************************************************** */
void reg_getImageGradient_gpu(const nifti_image &sourceImage,
                              const nifti_image &deformationImage,
                              const float *sourceImageArray_d,
                              const float *positionFieldImageArray_d,
                              float *resultGradientArray_d,
                              const resampler_boundary_e boundary,
                              const int interpolation) {
  if (sourceImage.nz > 1 || deformationImage.nz > 1) {
    _launchGradientKernelND<true>(sourceImage,
                                  deformationImage,
                                  sourceImageArray_d,
                                  positionFieldImageArray_d,
                                  resultGradientArray_d,
                                  boundary,
                                  interpolation);
  } else {
    _launchGradientKernelND<false>(sourceImage,
                                   deformationImage,
                                   sourceImageArray_d,
                                   positionFieldImageArray_d,
                                   resultGradientArray_d,
                                   boundary,
                                   interpolation);
  }
}
/* *************************************************************** */
/* *************************************************************** */

#endif
