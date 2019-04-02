/*
 *  _reg_resampling_gpu.h
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_RESAMPLING_GPU_H
#define _REG_RESAMPLING_GPU_H

#include "_reg_common_cuda.h"
#include "resampler_boundary.h"

extern "C++"
void reg_getImageGradient_gpu(const nifti_image &sourceImage,
                              const nifti_image &deformationImage,
                              const float *sourceImageArray_d,
                              const float *positionFieldImageArray_d,
                              float *resultGradientArray_d,
                              const resampler_boundary_e boundary,
                              const int interpolation);
#endif
