#pragma once

#include "niftyreg_cpu_resample_op.h"

/**
 * \brief CPU implementation of gradient wrt. deformation
 */
class NiftyRegCPUResampleGradientOp : public NiftyRegCPUResampleOp {
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
  virtual void Compute(tf::OpKernelContext *p_context) override;
  /** @} */

  /**
   * \name Instantiation
   * @{
   */
public:
  explicit NiftyRegCPUResampleGradientOp(tf::OpKernelConstruction* context) : Superclass(context) {}
  /** @} */
};
