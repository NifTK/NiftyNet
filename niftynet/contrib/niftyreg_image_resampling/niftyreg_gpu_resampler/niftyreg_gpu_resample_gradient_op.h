#pragma once

#include "niftyreg_gpu_resample_op.h"

/**
 * \brief GPU implementation of gradient wrt. deformation
 */
class NiftyRegGPUResampleGradientOp : public NiftyRegGPUResampleOp {
  /**
   * \name Typedefs
   * @{
   */
public:
  typedef NiftyRegGPUResampleOp Superclass;
  /** @} */

  /**
   * \name Op API
   * @{
   */
public:
  virtual void Compute(tf::OpKernelContext *p_context) override;
  /** @} */

  /**
   * \name Instantiation
   * @{
   */
public:
  explicit NiftyRegGPUResampleGradientOp(tf::OpKernelConstruction* context) : Superclass(context) {}
  /** @} */
};
