#include "niftyreg_cpu_resample_image_gradient_op.h"
#include "_reg_resampling.h"

tf::TensorShape NiftyRegCPUResampleImageGradientOp::compute_image_gradient_product_shape(tf::OpKernelContext *p_context) {
  return NiftyRegCPUResampleImageGradientOp::extract_image_shape(p_context);
}

tf::shape_inference::ShapeHandle NiftyRegCPUResampleImageGradientOp::compute_image_gradient_product_shape(tf::shape_inference::InferenceContext *p_context) {
  return p_context->input(0);
}

void NiftyRegCPUResampleImageGradientOp::Compute(tf::OpKernelContext *p_context) {
  const float *pc_image = p_context->input(0).flat<float>().data();
  const float *pc_deformation = p_context->input(1).flat<float>().data();
  const tf::Tensor &gradient_out = p_context->input(2);

  nifti_image image_nim;
  nifti_image deformation_nim;
  nifti_image gradient_out_nim;
  nifti_image gradient_image_nim;
  tf::Tensor *p_output;

  this->populate_nifti_headers_from_context(image_nim, deformation_nim, p_context);
  p_context->allocate_output(0, this->compute_image_gradient_product_shape(p_context), &p_output);
  this->load_nifti_dimensions_from_tensor(gradient_out_nim, gradient_out);
  this->load_nifti_dimensions_from_tensor(gradient_image_nim, *p_output);

  for (int b = 0; b < this->batch_size(p_context); ++b) {
    image_nim.data = const_cast<float*>(pc_image + b*image_nim.nvox);
    deformation_nim.data = const_cast<float*>(pc_deformation + b*deformation_nim.nvox);
    gradient_out_nim.data = const_cast<float*>(gradient_out.flat<float>().data() + b*gradient_out_nim.nvox);
    gradient_image_nim.data = p_output->flat<float>().data() + b*gradient_image_nim.nvox;

    compute_gradient_product(gradient_image_nim, image_nim, deformation_nim, gradient_out_nim, this->boundary(), this->interpolation());
  }
}
