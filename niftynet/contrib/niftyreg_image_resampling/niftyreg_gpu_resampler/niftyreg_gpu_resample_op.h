#pragma once

#include "niftyreg_resample_op.h"

/**
 * \brief GPU Implementation of NiftyReg-powered image resampling operation.
 */
class NiftyRegGPUResampleOp : public NiftyRegResampleOp<Eigen::GpuDevice> {
  /**
   * \name Typedefs
   * @{
   */
public:
  typedef NiftyRegResampleOp<Eigen::GpuDevice> Superclass;
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
  explicit NiftyRegGPUResampleOp(tf::OpKernelConstruction* context) : Superclass(context) {}
  /** @} */
};
