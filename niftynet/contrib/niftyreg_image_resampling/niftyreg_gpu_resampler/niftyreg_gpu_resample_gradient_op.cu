#include "niftyreg_gpu_resample_gradient_op.h"
#include "niftyreg_gpu_resample_op.h"
#include "_reg_resampling_gpu.h"

#include <iostream>

void NiftyRegGPUResampleGradientOp::Compute(tf::OpKernelContext *p_context) {
  const auto &input_image = this->extract_floating_image(p_context);
  const auto &input_deformation = this->extract_deformation_image(p_context);

  nifti_image img_nim;
  nifti_image deformation_nim;
  float *dp_gradient;
  tf::Tensor *p_output;

  this->load_nifti_dimensions_from_tensors(img_nim, deformation_nim, input_image, input_deformation);

  p_context->allocate_output(0, this->compute_gradient_shape(p_context), &p_output);
  dp_gradient = p_output->flat<float>().data();

  if (this->interpolation() != 1 && this->interpolation() != 3) {
    std::cerr << "WARNING: gradient is only available for cubic/linear interpolation.\n";
  }

  for (int b = 0; b < this->batch_size(p_context); ++b) {
    for (int m = 0; m < img_nim.nu; ++m) {
      const float *dpc_source = input_image.flat<float>().data() + (b + m)*img_nim.nx*img_nim.ny*img_nim.nz;
      const float *dpc_deformation = input_deformation.flat<float>().data() + b*deformation_nim.nvox;

      reg_getImageGradient_gpu(img_nim,
                               deformation_nim,
                               dpc_source,
                               dpc_deformation,
                               dp_gradient,
                               this->boundary(),
                               this->interpolation());

      dp_gradient += deformation_nim.nvox;
    }
  }
}
