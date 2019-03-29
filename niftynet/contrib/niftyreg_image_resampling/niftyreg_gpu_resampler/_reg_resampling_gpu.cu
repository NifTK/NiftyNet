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

/* *************************************************************** */
/* *************************************************************** */
template <const bool tIs3D, const bool tDoClamp>
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

        float deriv[2];
        deriv[0]=-1.0f;
        deriv[1]=1.0f;

        float4 gradientValue=make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (tIs3D) {
          for(short c=0; c<2; c++){
            int z = voxel.z + c;

            if (tDoClamp) z = clampIndex(z, floating_dims.z);

            float3 tempValueY=make_float3(0.0f, 0.0f, 0.0f);
            for(short b=0; b<2; b++){
              float2 tempValueX=make_float2(0.0f, 0.0f);
              int y = voxel.y + b;

              if (tDoClamp) y = clampIndex(y, floating_dims.y);
              for(short a=0; a<2; a++){
                int x= voxel.x + a;
                float intensity=paddingValue;

                if (tDoClamp) x = clampIndex(x, floating_dims.x);
                if (tDoClamp || (0 <= x && x < floating_dims.x
                                 && 0 <= y && y < floating_dims.y
                                 && 0 <= z && z < floating_dims.z)) {
                  intensity = pc_floating[((z*floating_dims.y)+y)*floating_dims.x+x];
                }

                tempValueX.x +=  intensity * deriv[a];
                tempValueX.y +=  intensity * xBasis[a];
              }
              tempValueY.x += tempValueX.x * yBasis[b];
              tempValueY.y += tempValueX.y * deriv[b];
              tempValueY.z += tempValueX.y * yBasis[b];
            }
            gradientValue.x += tempValueY.x * zBasis[c];
            gradientValue.y += tempValueY.y * zBasis[c];
            gradientValue.z += tempValueY.z * deriv[c];
          }
        } else {
          for(short b=0; b<2; b++){
            float2 tempValueX=make_float2(0.0f, 0.0f);
            int y = voxel.y + b;

            if (tDoClamp) y = clampIndex(y, floating_dims.y);
            for(short a=0; a<2; a++){
              int x = voxel.x + a;
              float intensity=paddingValue;

              if (tDoClamp) x = clampIndex(x, floating_dims.x);
              if (tDoClamp || (0 <= x && x < floating_dims.x
                               && 0 <= y && y < floating_dims.y)) {
                intensity = pc_floating[y*floating_dims.x+x];
              }

              tempValueX.x +=  intensity * deriv[a];
              tempValueX.y +=  intensity * xBasis[a];
            }
            gradientValue.x += tempValueX.x * yBasis[b];
            gradientValue.y += tempValueX.y * deriv[b];
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
template <const bool tIs3D, const bool tDoClamp>
static void _launchGradientKernelBoundary(const nifti_image &sourceImage,
                                          const nifti_image &deformationImage,
                                          const float *sourceImageArray_d,
                                          const float *positionFieldImageArray_d,
                                          float *resultGradientArray_d,
                                          const float pad) {
  int3 floatingDim = make_int3(sourceImage.nx, sourceImage.ny, sourceImage.nz);
  int3 deformationDim = make_int3(deformationImage.nx, deformationImage.ny, deformationImage.nz);
  dim3 B1;
  dim3 G1;
  int ref_size = deformationImage.nx*deformationImage.ny*deformationImage.nz;

  cudaCommon_computeGridConfiguration(B1, G1, ref_size);
  reg_getImageGradient_kernel<tIs3D, tDoClamp> <<<G1, B1>>> (resultGradientArray_d,
                                                             sourceImageArray_d,
                                                             positionFieldImageArray_d,
                                                             floatingDim,
                                                             deformationDim,
                                                             pad,
                                                             ref_size);
}
/* *************************************************************** */
template <const bool tIs3D>
static void _launchGradientKernelND(const nifti_image &sourceImage,
                                    const nifti_image &deformationImage,
                                    const float *sourceImageArray_d,
                                    const float *positionFieldImageArray_d,
                                    float *resultGradientArray_d,
                                    const resampler_boundary_e boundary) {
  const float pad = get_padding_value<float>(boundary);

  if (boundary == resampler_boundary_e::CLAMPING) {
    _launchGradientKernelBoundary<tIs3D, true>(sourceImage,
                                               deformationImage,
                                               sourceImageArray_d,
                                               positionFieldImageArray_d,
                                               resultGradientArray_d,
                                               pad);
  } else {
    _launchGradientKernelBoundary<tIs3D, false>(sourceImage,
                                                deformationImage,
                                                sourceImageArray_d,
                                                positionFieldImageArray_d,
                                                resultGradientArray_d,
                                                pad);
  }
}
/* *************************************************************** */
void reg_getImageGradient_gpu(const nifti_image &sourceImage,
                              const nifti_image &deformationImage,
                              const float *sourceImageArray_d,
                              const float *positionFieldImageArray_d,
                              float *resultGradientArray_d,
                              const resampler_boundary_e boundary) {
  if (sourceImage.nz > 1 || deformationImage.nz > 1) {
    _launchGradientKernelND<true>(sourceImage,
                                  deformationImage,
                                  sourceImageArray_d,
                                  positionFieldImageArray_d,
                                  resultGradientArray_d,
                                  boundary);
  } else {
    _launchGradientKernelND<false>(sourceImage,
                                   deformationImage,
                                   sourceImageArray_d,
                                   positionFieldImageArray_d,
                                   resultGradientArray_d,
                                   boundary);
  }
}
/* *************************************************************** */
/* *************************************************************** */

#endif
