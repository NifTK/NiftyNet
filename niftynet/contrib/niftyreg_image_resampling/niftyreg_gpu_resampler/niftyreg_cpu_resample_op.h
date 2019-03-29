#pragma once

#include "niftyreg_resample_op.h"

/**
 * \brief CPU implementation of NiftyRegResampleOp
 */
class NiftyRegCPUResampleOp : public NiftyRegResampleOp<Eigen::ThreadPoolDevice> {
  /**
   * \name Typedefs
   * @{
   */
public:
  typedef NiftyRegResampleOp<Eigen::ThreadPoolDevice> Superclass;
  /** @} */

  /**
   * \name Op API
   * @{
   */
protected:
  /** \brief Fully populates the image and deformation-field Nifti-headers given the kernel context */
  void populate_nifti_headers_from_context(nifti_image &r_image, nifti_image &r_deformation, tf::OpKernelContext *p_context) const;

public:
  virtual void Compute(tf::OpKernelContext *p_context) override;
  /** @} */

  /**
   * \name Instantiation
   * @{
   */
public:
  explicit NiftyRegCPUResampleOp(tf::OpKernelConstruction* context) : Superclass(context) {}
  /** @} */
};
