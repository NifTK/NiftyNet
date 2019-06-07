#pragma once

#include "niftyreg_cpu_resample_op.h"

/**
 * \brief CPU implementation of NiftyRegResampleOp
 */
class NiftyRegCPUResampleImageGradientOp : public NiftyRegCPUResampleOp {
  /**
   * \name Typedefs
   * @{
   */
public:
  typedef NiftyRegCPUResampleOp Superclass;
  /** @} */

  /**
   * \name Op API
   * @{
   */
public:
  /** \returns the size of the gradient outputted by this operation, at execution time */
  static tf::TensorShape compute_image_gradient_product_shape(tf::OpKernelContext *p_context);

  /** \returns the size of the gradient outputted by this operation, at graph-compilation time */
  static tf::shape_inference::ShapeHandle compute_image_gradient_product_shape(tf::shape_inference::InferenceContext *p_context);

  virtual void Compute(tf::OpKernelContext *p_context) override;
  /** @} */

  /**
   * \name Instantiation
   * @{
   */
public:
  explicit NiftyRegCPUResampleImageGradientOp(tf::OpKernelConstruction* context) : Superclass(context) {}
  /** @} */
};
