/**
 * @file _reg_resampling.h
 * @author Marc Modat
 * @date 24/03/2009
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_RESAMPLING_H
#define _REG_RESAMPLING_H

#include "nifti1_io.h"
#include "resampler_boundary.h"

/** @brief This function resample a floating image into the space of a reference/warped image.
 * The deformation is provided by a 4D nifti image which is in the space of the reference image.
 * In the 4D image, for each voxel i,j,k, the position in the real word for the floating image is store.
 * Interpolation can be nearest Neighbor (0), linear (1) or cubic spline (3).
 * The cubic spline interpolation assume a padding value of 0
 * The padding value for the NN and the LIN interpolation are user defined.
 * @param floatingImage Floating image that is interpolated
 * @param warpedImage Warped image that is being generated
 * @param deformationField Vector field image that contains the dense correspondences
 * @param interp Interpolation type. 0, 1 or 3 correspond to nearest neighbor, linear or cubic
 * interpolation
 * @param boundaryTreatment specifies how to treat image boundaries
 * reference image space.
 * \sa resampler_boundary_e
 */
extern "C++"
void reg_resampleImage(nifti_image *floatingImage,
                       nifti_image *warpedImage,
                       nifti_image *deformationField,
                       int interp,
                       resampler_boundary_e boundaryTreatment);

extern "C++"
void reg_getImageGradient(nifti_image *floatingImage,
                          nifti_image *warImgGradient,
                          nifti_image *deformationField,
                          int interp,
                          resampler_boundary_e boundaryTreatment,
                          int active_timepoint,
                          nifti_image *warpedImage = NULL);

/**
 * \brief Computes the tensor product of an outgoing gradient and the derivative of the warped image wrt. the floating image
 * \param r_destination output image (same dimensions as floating image)
 * \param image floating image
 * \param deformation sampling indices/deformation field
 * \param gradient_out downstream derivative wrt. the warped image
 * \param boundary boundary treatment flag
 * \parm interpolation standard (0, 1, 3) interpolation code
 */
void compute_gradient_product(nifti_image &r_destination, const nifti_image &image, const nifti_image &deformation, const nifti_image &gradient_out,
                              const resampler_boundary_e boundary, const int interpolation);

#endif
