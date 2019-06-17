#include "niftyreg_cpu_resample_gradient_op.h"
#include "niftyreg_cpu_resample_op.h"

#include "_reg_resampling.h"

void NiftyRegCPUResampleGradientOp::Compute(tf::OpKernelContext *p_context) {
  float const *pc_floating = this->extract_floating_image(p_context).flat<float>().data();
  float const *pc_deformation = this->extract_deformation_image(p_context).flat<float>().data();
  float *p_gradient = nullptr;
  nifti_image *p_img_nim = nifti_simple_init_nim();
  nifti_image *p_deformation_nim = nifti_simple_init_nim();
  nifti_image gradient;
  int image_size;
  int displacement_size;
  tf::Tensor *p_output;

  this->populate_nifti_headers_from_context(*p_img_nim, *p_deformation_nim, p_context);

  image_size = p_img_nim->nvox;
  displacement_size = p_deformation_nim->nvox;

  p_context->allocate_output(0, this->compute_gradient_shape(p_context), &p_output);
  this->load_nifti_dimensions_from_tensor(gradient, *p_output);
  p_gradient = p_output->flat<float>().data();
  if (gradient.nu != p_deformation_nim->nu) {
    gradient.nu = gradient.dim[5] = p_deformation_nim->nu;
    gradient.nvox = p_deformation_nim->nvox;
  }

  for (int b = 0; b < this->batch_size(p_context); ++b) {
    for (int m = 0; m < p_img_nim->nu*p_img_nim->nt; ++m) {
      p_deformation_nim->data = const_cast<float*>(pc_deformation);
      p_img_nim->data = const_cast<float*>(pc_floating);
      gradient.data = p_gradient;

      reg_getImageGradient(p_img_nim,
                           &gradient,
                           p_deformation_nim,
                           this->interpolation(),
                           this->boundary(),
                           m,
                           nullptr);

      p_gradient += displacement_size;
    }
    pc_deformation += displacement_size;
    pc_floating += image_size;
  }

  p_img_nim->data = p_deformation_nim->data = nullptr;
  nifti_image_free(p_img_nim);
  nifti_image_free(p_deformation_nim);
}
