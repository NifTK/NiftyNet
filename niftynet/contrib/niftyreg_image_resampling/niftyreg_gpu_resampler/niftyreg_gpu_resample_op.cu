#include "niftyreg_gpu_resample_op.h"
#include "resampleKernel.h"
#include "_reg_common_cuda.h"
#ifdef DT_INT32
#undef DT_INT32
#endif
#ifdef DT_FLOAT
#undef DT_FLOAT
#endif

#include <tensorflow/core/framework/types.pb.h>
#include <cmath>
#include <limits>
#include <numeric>

void NiftyRegGPUResampleOp::Compute(tf::OpKernelContext *p_context) {
  const auto &input_image = this->extract_floating_image(p_context);
  const auto &input_deformation = this->extract_deformation_image(p_context);

  tf::Tensor *p_destination;
  nifti_image floating_image;
  nifti_image warped_image;
  nifti_image deformation_image;

  this->load_nifti_dimensions_from_tensors(floating_image, deformation_image, input_image, input_deformation);

  p_context->allocate_output(0, this->compute_output_shape(p_context), &p_destination);
  this->load_nifti_dimensions_from_tensor(warped_image, *p_destination);

  for (int b = 0; b < this->batch_size(p_context); ++b) {
    const float *dpc_floating = input_image.flat<float>().data() + b*floating_image.nvox;
    const float *dpc_deformation = input_deformation.flat<float>().data() + b*deformation_image.nvox;

    float *dp_warped = p_destination->flat<float>().data() + b*warped_image.nvox;

    launchResample(&floating_image, &warped_image, this->interpolation(), this->boundary(), dpc_floating, dp_warped, dpc_deformation);
  }
}
