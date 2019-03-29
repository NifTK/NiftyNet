#include <cstdlib>
#include <iostream>
#include <string>
#include <boost/filesystem.hpp>

#include "_reg_maths.h"
#include "resampleKernel.h"

namespace bfs = boost::filesystem;

static void _test_assert(const bool value, const char condition[]) {
  if (!value) {
    std::cerr << "FAILED: " << condition << std::endl;
    std::abort();
  }
}

#define RESAMPLING_TEST_ASSERT(condition) _test_assert((condition), #condition)

static void _check_image_displacement(const nifti_image &floating, const nifti_image &warped, const float voxel_displacement, const int stride, const int interpolation) {
  const bool use_nearest = interpolation == 0;
  const int max_idx_displacement = use_nearest? int(std::round(voxel_displacement)) : int(std::ceil(voxel_displacement));
  const float *floating_begin = (const float*)(floating.data);
  const float *floating_end = floating_begin + floating.nvox;
  const float *warped_begin = (const float*)(warped.data);

  int nof_nans = 0;
  int boundary_size = 0;

  RESAMPLING_TEST_ASSERT(floating.nvox == warped.nvox);

  for (const float *pc_floating = floating_begin; pc_floating < floating_end; ++pc_floating) {
    if (pc_floating + max_idx_displacement*stride >= floating_begin && pc_floating + max_idx_displacement*stride < floating_end) {
      int offset = pc_floating - floating_begin;
      float ref_value = *(pc_floating + max_idx_displacement*stride);
      float test_value = *(warped_begin + offset);

      if (!use_nearest && voxel_displacement - std::floor(voxel_displacement) > 0.1) {
        ref_value = (*(pc_floating + (max_idx_displacement - 1)*stride) + ref_value)/2;
      }

      if (std::isfinite(test_value)) {
        RESAMPLING_TEST_ASSERT(std::fabs(ref_value - test_value) <= 1e-2*std::fabs(ref_value));
      } else {
        nof_nans += 1;
      }
    }
  }

  if (floating.ndim == 2) {
    boundary_size = (1 + (std::max)(max_idx_displacement, 1)*(interpolation + 1))*(floating.ny + floating.nx);
  } else {
    boundary_size = (7 + (std::max)(max_idx_displacement, 1)*(interpolation + 1))*(std::max)((std::max)(floating.ny*floating.nx, floating.ny*floating.nz), floating.nx*floating.nz);
  }

  RESAMPLING_TEST_ASSERT(nof_nans <= boundary_size);
}

static void _test_displacement(const std::string &image_path) {
  nifti_image const *pc_image = nifti_image_read(image_path.c_str(), 1);
  nifti_image *p_displacements = nifti_copy_nim_info(pc_image);
  std::vector<float> displacement_field(pc_image->nvox*pc_image->dim[0]);

  for (const int interp: {0, 1, 3}) {
    for (int d = 0; d < pc_image->dim[0] - int(pc_image->dim[pc_image->dim[0]] <= 1); ++d) {
      int stride = 1;

      for (int i = 0; i < d; ++i) {
        stride *= pc_image->dim[1+i];
      }

      for (const float voxel_displacement: {0.0f, 1.0f, 0.500001f, 1.50001f}) {
        const float displacement = voxel_displacement*pc_image->pixdim[1+d];

        nifti_image *p_warped;

        std::fill(displacement_field.begin(), displacement_field.end(), 0.f);
        std::fill(displacement_field.begin() + d*pc_image->nvox, displacement_field.begin() + (d + 1)*pc_image->nvox, displacement);

        p_displacements->ndim = p_displacements->dim[0] = 5;
        p_displacements->nt = p_displacements->dim[4] = 1;
        p_displacements->nu = p_displacements->dim[5] = pc_image->dim[0];
        p_displacements->nvox = pc_image->nvox*p_displacements->nu;
        p_displacements->intent_p1 = DISP_FIELD;

        p_displacements->data = (void*)(displacement_field.data());
        p_warped = resample(*p_displacements, *pc_image, interp, std::numeric_limits<float>::quiet_NaN());

        _check_image_displacement(*pc_image, *p_warped, voxel_displacement, stride, interp);

        nifti_image_free(p_warped);
      }
    }
  }
  p_displacements->data = nullptr;
  nifti_image_free(p_displacements);
  nifti_image_free(const_cast<nifti_image*>(pc_image));
}

int main(int argc, char* argv[]) {
  const bfs::path data_path = bfs::path(argv[0]).parent_path();

  _test_displacement((data_path/"data"/"test_image_2.nii").native());
  _test_displacement((data_path/"data"/"test_image_3.nii").native());

  return 0;
}
