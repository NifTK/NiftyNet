#pragma once

#include "resampler_boundary.h"
#include "nifti1_io.h"

/**
 * \brief Runs a resampling operation on the image referenced by pc_floating.
 * \param[in] pc_floating floating image description (header only)
 * \param[in] pc_warped destination image description (header only)
 * \param interp interpolation code (0, 1, 3)
 * \param[out] dp_warped device base pointer of destination buffer.
 */
void launchResample(const nifti_image *pc_floating, const nifti_image *pc_warped, const int interp,
                    const resampler_boundary_e boundary, const float *floatingImage_d, float *dp_warped,
                    const float *dpc_deformation);

/**
 * \brief Resamples the floating argument image based on the displacement/deformation field passed in r_displacements
 * \param r_displacements displacement/deformation field, if given as displacements it is converted to a deformation field in-place, thus destroying the original data.
 * \param floating the image to resample
 * \param interp_code a NiftyReg interpolation code (0, 1, 3, 4)
 * \param is_displacement_argument indicates if the first argument is a displacement field or a deformation field (default: true)
 * \returns the resampled image, to be freed by client
 */
nifti_image* resample(nifti_image &r_displacements, const nifti_image &floating, const int interp_code, const resampler_boundary_e boundary, const bool is_displacement_argument = true);

