/**
 * @file _reg_tools.h
 * @author Marc Modat
 * @date 25/03/2009
 * @brief Set of useful functions
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_TOOLS_H
#define _REG_TOOLS_H

#include <fstream>
#include <map>

#include "_reg_maths.h"
#include "resampler_boundary.h"

typedef enum
{
   MEAN_KERNEL,
   LINEAR_KERNEL,
   GAUSSIAN_KERNEL,
   CUBIC_SPLINE_KERNEL
} NREG_CONV_KERNEL_TYPE;
/* *************************************************************** */
/** @brief This function check some header parameters and correct them in
 * case of error. For example no dimension is lower than one. The scl_sclope
 * can not be equal to zero. The qto_xyz and qto_ijk are populated if
 * both qform_code and sform_code are set to zero.
 * @param image Input image to check and correct if necessary
 */
extern "C++"
void reg_checkAndCorrectDimension(nifti_image *image);

/* *************************************************************** */
/** @brief reg_getRealImageSpacing
 * @param image image
 * @param spacingValues spacingValues
 */
extern "C++"
void reg_getRealImageSpacing(nifti_image *image,
                             float *spacingValues);

/* *************************************************************** */
/** @brief Check if the specified filename corresponds to an image.
 * @param name Input filename
 * @return True is the specified filename corresponds to an image,
 * false otherwise.
 */
extern "C++"
bool reg_isAnImageFileName(char *name);

/* *************************************************************** */
/** @brief Rescale an input image between two user-defined values.
 * Some threshold can also be applied concurrenlty
 * @param image Image to be rescaled
 * @param newMin Intensity lower bound after rescaling
 * @param newMax Intensity higher bound after rescaling
 * @param lowThr Intensity to use as lower threshold
 * @param upThr Intensity to use as higher threshold
 */
extern "C++"
void reg_intensityRescale(nifti_image *image,
                          int timepoint,
                          float newMin,
                          float newMax
                         );

/* *************************************************************** */
/** @brief This function converts an image containing deformation
 * field into a displacement field
 * The conversion is done using the appropriate qform/sform
 * @param image Image that contains a deformation field and will be
 * converted into a displacement field
 */
extern "C++"
int reg_getDisplacementFromDeformation(nifti_image *image);
/* *************************************************************** */
/** @brief This function converts an image containing a displacement field
 * into a displacement field.
 * The conversion is done using the appropriate qform/sform
 * @param image Image that contains a deformation field and will be
 * converted into a displacement field
 */
extern "C++"
int reg_getDeformationFromDisplacement(nifti_image *image);
/* *************************************************************** */
/** @brief The functions returns the largest ratio between input image intensities
 * The returned value is the largest value computed as ((A/B)-1)
 * If A or B are zeros then the (A-B) value is returned.
 */
extern "C++"
double reg_test_compare_images(nifti_image *imgA,
                              nifti_image *imgB);
/* *************************************************************** */
/** @brief The absolute operator is applied to the input image
 */
extern "C++"
void reg_tools_abs_image(nifti_image *img);
/* *************************************************************** */
extern "C++"
void mat44ToCptr(mat44 mat, float* cMat);
/* *************************************************************** */
extern "C++"
void cPtrToMat44(mat44 *mat, float* cMat);
/* *************************************************************** */
extern "C++"
void mat33ToCptr(mat33* mat, float* cMat, const unsigned int numMats);
/* *************************************************************** */
extern "C++"
void cPtrToMat33(mat33 *mat, float* cMat);
/* *************************************************************** */
extern "C++" template<typename T>
void matmnToCptr(T** mat, T* cMat, unsigned int m, unsigned int n);
/* *************************************************************** */
extern "C++" template<typename T>
void cPtrToMatmn(T** mat, T* cMat, unsigned int m, unsigned int n);
/* *************************************************************** */
void coordinateFromLinearIndex(int index, int maxValue_x, int maxValue_y, int &x, int &y, int &z);
/* *************************************************************** */
#endif
