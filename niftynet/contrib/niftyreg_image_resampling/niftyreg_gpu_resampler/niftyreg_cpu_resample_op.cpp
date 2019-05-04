#include "niftyreg_cpu_resample_op.h"
#include "niftyreg_cpu_resample_gradient_op.h"
#include "niftyreg_cpu_resample_image_gradient_op.h"
#ifdef GOOGLE_CUDA
#include "niftyreg_gpu_resample_op.h"
#include "niftyreg_gpu_resample_gradient_op.h"
#endif

#include "_reg_resampling.h"
#undef DT_INT32
#undef DT_FLOAT

#include <tensorflow/core/framework/shape_inference.h>
#include <limits>

void NiftyRegCPUResampleOp::populate_nifti_headers_from_context(nifti_image &r_image, nifti_image &r_deformation, tf::OpKernelContext *p_context) const {
  const tf::Tensor& input_image = this->extract_floating_image(p_context);
  const tf::Tensor& input_deformation = this->extract_deformation_image(p_context);

  this->load_nifti_dimensions_from_tensors(r_image, r_deformation, input_image, input_deformation);

  r_deformation.datatype = r_image.datatype = NIFTI_TYPE_FLOAT32;
  r_deformation.nbyper = r_image.nbyper = sizeof(float);
}

void NiftyRegCPUResampleOp::Compute(tf::OpKernelContext *p_context) {
  float const *pc_floating = this->extract_floating_image(p_context).flat<float>().data();
  float const *pc_deformation = this->extract_deformation_image(p_context).flat<float>().data();
  float *p_warped = nullptr;
  nifti_image *p_img_nim = nifti_simple_init_nim();
  nifti_image *p_deformation_nim = nifti_simple_init_nim();
  nifti_image warped;
  tf::Tensor *p_output;

  this->populate_nifti_headers_from_context(*p_img_nim, *p_deformation_nim, p_context);
  OP_REQUIRES(p_context, p_context->input(0).dims() == p_context->input(1).dims(), tf::errors::InvalidArgument("Image and sample index tensors must have the same dimensionality."));

  p_context->allocate_output(0, this->compute_output_shape(p_context), &p_output);
  this->load_nifti_dimensions_from_tensor(warped, *p_output);
  p_warped = p_output->flat<float>().data();

  for (int b = 0; b < this->batch_size(p_context); ++b) {
    p_deformation_nim->data = const_cast<float*>(pc_deformation);
    p_img_nim->data = const_cast<float*>(pc_floating);
    warped.data = p_warped;

   reg_resampleImage(p_img_nim,
                     &warped,
                     p_deformation_nim,
                     this->interpolation(),
                     this->boundary());

   pc_deformation += p_deformation_nim->nvox;
   pc_floating += p_img_nim->nvox;
   p_warped += warped.nvox;
  }

  p_img_nim->data = p_deformation_nim->data = nullptr;
  nifti_image_free(p_img_nim);
  nifti_image_free(p_deformation_nim);
}

static const char gc_opname[] = "NiftyregImageResampling";

REGISTER_OP(gc_opname)
.Attr("interpolation: int = 1")
.Attr("boundary: int = 1")
.Input("image: float")
.Input("deformation: float")
.Output("warped: float")
.SetShapeFn([](tf::shape_inference::InferenceContext *p_context) {
    p_context->set_output(0, NiftyRegCPUResampleOp::compute_output_shape(p_context));

    return tf::Status::OK();
  });

REGISTER_KERNEL_BUILDER(Name(gc_opname)
                        .Device(tf::DEVICE_CPU),
                        NiftyRegCPUResampleOp);

#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name(gc_opname)
                        .Device(tf::DEVICE_GPU),
                        NiftyRegGPUResampleOp);
#endif

static const char gc_gradient_opname[] = "NiftyregImageResamplingGradient";

REGISTER_OP(gc_gradient_opname)
.Attr("interpolation: int = 1")
.Attr("boundary: int = 1")
.Input("image: float")
.Input("deformation: float")
.Output("gradient: float")
.SetShapeFn([](tf::shape_inference::InferenceContext *p_context) {
    p_context->set_output(0, NiftyRegCPUResampleGradientOp::compute_gradient_shape(p_context));

    return tf::Status::OK();
  });

REGISTER_KERNEL_BUILDER(Name(gc_gradient_opname)
                        .Device(tf::DEVICE_CPU),
                        NiftyRegCPUResampleGradientOp);

#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name(gc_gradient_opname)
                        .Device(tf::DEVICE_GPU),
                        NiftyRegGPUResampleGradientOp);
#endif

static const char gc_image_gradient_opname[] = "NiftyregImageResamplingImageGradient";

REGISTER_OP(gc_image_gradient_opname)
.Attr("interpolation: int = 1")
.Attr("boundary: int = 1")
.Input("image: float")
.Input("deformation: float")
.Input("loss_gradient: float")
.Output("image_gradient: float")
.SetShapeFn([](tf::shape_inference::InferenceContext *p_context) {
    p_context->set_output(0, NiftyRegCPUResampleImageGradientOp::compute_image_gradient_product_shape(p_context));

    return tf::Status::OK();
  });

REGISTER_KERNEL_BUILDER(Name(gc_image_gradient_opname)
                        .Device(tf::DEVICE_CPU),
                        NiftyRegCPUResampleImageGradientOp);

