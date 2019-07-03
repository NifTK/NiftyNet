#pragma once

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif


#include "resampler_boundary.h"

#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>

struct _nifti_image;

namespace tf = tensorflow;

/**
 * \brief Base class for NiftyReg image resampling operations.
 *
 * Assumes that the input data (and subsequently the output data) have been transposed. I.e., the displacement-component index is assumed
 * to be the slowest moving index.
 */
template <class TDevice>
class NiftyRegResampleOp : public tf::OpKernel {
  /**
   * \name Typedefs
   * @{
   */
public:
  typedef tf::OpKernel Superclass;
  /** @} */

  /**
   * \name Settings
   * @{
   */
private:
  int m_interpolation;
  resampler_boundary_e m_boundary;

protected:
  /** \returns the NiftyReg interpolation code */
  int interpolation(void) const {
    return m_interpolation;
  }

  /** \returns the padding value appropriate for the requested boundary treatment */
  float padding(void) const {
    return m_boundary == resampler_boundary_e::ZEROPAD? 0.f : std::numeric_limits<float>::quiet_NaN();
  }

  /** \returns the requested boundary treatment */
  resampler_boundary_e boundary(void) const {
    return m_boundary;
  }
  /** @} */

  /**
   * \name Utility functions
   * @{
   */
private:
  static tf::TensorShape _make_fake_dim(tf::shape_inference::InferenceContext *p_context, const int input_idx);

protected:
  /** \returns the deformation field input */
  static const tf::Tensor& extract_floating_image(tf::OpKernelContext *p_context) {
    return p_context->input(0);
  }

  /** \returns the deformation field input */
  static const tf::Tensor& extract_deformation_image(tf::OpKernelContext *p_context) {
    return p_context->input(1);
  }

  /** \returns the shape of the i-th input */
  static const tf::TensorShape& extract_input_shape(tf::OpKernelContext *p_context, const int i) {
    return p_context->input(i).shape();
  }

  /** \returns the shape of the input image */
  static const tf::TensorShape& extract_image_shape(tf::OpKernelContext *p_context) {
    return NiftyRegResampleOp::extract_input_shape(p_context, 0);
  }

  /** \returns the partial shape of the input image (all but first dimension must be known in advance) */
  static tf::TensorShape extract_image_shape(tf::shape_inference::InferenceContext *p_context) {
    return _make_fake_dim(p_context, 0);
  }

  /** \returns the shape of the deformation image */
  static const tf::TensorShape& extract_deformation_shape(tf::OpKernelContext *p_context) {
    return NiftyRegResampleOp::extract_input_shape(p_context, 1);
  }

  /** \returns the partial shape of the input deformation (all but first dimension must be known in advance) */
  static tf::TensorShape extract_deformation_shape(tf::shape_inference::InferenceContext *p_context) {
    return _make_fake_dim(p_context, 1);
  }

  /** \returns the spatial rank of the image */
  static int infer_spatial_rank(tf::OpKernelContext *p_context) {
    return NiftyRegResampleOp::extract_deformation_shape(p_context).dim_size(1);
  }

  /** \returns the batch size */
  static int batch_size(tf::OpKernelContext *p_context) {
    return NiftyRegResampleOp::extract_floating_image(p_context).dim_size(0);
  }

  /** \brief Converts tensor dimensions into Nifti-image dimension arrays */
  static void load_nifti_dimensions_from_tensors(_nifti_image &r_image, _nifti_image &r_deformation, const tf::Tensor &image_tensor, const tf::Tensor &deformation_tensor);

  /** \brief Converts tensor dimensions into Nifti-image dimension arrays */
  static void load_nifti_dimensions_from_tensor(_nifti_image &r_image, const tf::Tensor &tensor);

public:
  /**
   * \returns the shape of the output image
   */
  static tf::TensorShape compute_output_shape(tf::OpKernelContext *p_context);

  /**
   * \returns the shape of the output image at Op-registration time
   */
  static tf::shape_inference::ShapeHandle compute_output_shape(tf::shape_inference::InferenceContext *p_context);

  /**
   * \returns the shape of the output image
   */
  static tf::TensorShape compute_output_shape(const tf::TensorShape &image, const tf::TensorShape &deformation);

  /**
   * \returns the shape of the gradient image
   */
  static tf::TensorShape compute_gradient_shape(tf::OpKernelContext *p_context);

  /**
   * \returns the shape of the gradient image at Op-registration time
   */
  static tf::shape_inference::ShapeHandle compute_gradient_shape(tf::shape_inference::InferenceContext *p_context);

  /**
   * \returns the shape of the gradient image
   */
  static tf::TensorShape compute_gradient_shape(const tf::TensorShape &image, const tf::TensorShape &deformation);
  /** @} */

  /**
   * \name Instantiation
   * @{
   */
public:
  NiftyRegResampleOp(tf::OpKernelConstruction* p_context);
  /** @} */
};

#include "niftyreg_resample_op.tpp"
