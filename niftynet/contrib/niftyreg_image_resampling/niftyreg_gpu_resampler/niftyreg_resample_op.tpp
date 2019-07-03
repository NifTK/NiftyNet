#pragma once

#include "niftyreg_resample_op.h"
#include "_reg_resampling.h"

#include <algorithm>
#include <cassert>
#include <vector>

template <class TDevice>
tf::TensorShape NiftyRegResampleOp<TDevice>::compute_output_shape(const tf::TensorShape &image, const tf::TensorShape &deformation) {
  tf::TensorShape output_shape;

  output_shape.AddDim(image.dim_size(0));
  output_shape.AddDim(image.dim_size(1));
  for (int d = 2; d < deformation.dims(); ++d) {
    output_shape.AddDim(deformation.dim_size(d));
  }

  return output_shape;
}

template <class TDevice>
tf::TensorShape NiftyRegResampleOp<TDevice>::compute_output_shape(tf::OpKernelContext *p_context) {
  return NiftyRegResampleOp::compute_output_shape(NiftyRegResampleOp::extract_image_shape(p_context), NiftyRegResampleOp::extract_deformation_shape(p_context));
}

template <class TDevice>
tf::TensorShape NiftyRegResampleOp<TDevice>::_make_fake_dim(tf::shape_inference::InferenceContext *p_context, const int iidx) {
  const auto &known_shape = p_context->input(iidx);

  tf::TensorShape shape;

  shape.AddDim(1);
  for (int i = 1; i < p_context->Rank(known_shape); ++i) {
    if (!p_context->ValueKnown(p_context->Dim(known_shape, i)) || p_context->Value(p_context->Dim(known_shape, i)) < 0) {
      shape.AddDim(0);
      std::cerr << "Warning: unknown dimension size " << i << " in input " << iidx << std::endl;
    } else {
      shape.AddDim(p_context->Value(p_context->Dim(known_shape, i)));
    }
  }

  return shape;
}

template <class TDevice>
tf::shape_inference::ShapeHandle NiftyRegResampleOp<TDevice>::compute_output_shape(tf::shape_inference::InferenceContext *p_context) {
  tf::TensorShape inferred_shape;
  tf::TensorShape image_shape;
  tf::TensorShape deformation_shape;
  tf::shape_inference::ShapeHandle inferred_shape_handle;

  if (p_context->input_tensor(0) == nullptr || p_context->input_tensor(1) == nullptr) {
    image_shape = NiftyRegResampleOp::extract_image_shape(p_context);
    deformation_shape = NiftyRegResampleOp::extract_deformation_shape(p_context);
  } else {
    image_shape = p_context->input_tensor(0)->shape();
    deformation_shape = p_context->input_tensor(1)->shape();
  }

  inferred_shape = NiftyRegResampleOp::compute_output_shape(image_shape, deformation_shape);
  p_context->MakeShapeFromTensorShape(inferred_shape, &inferred_shape_handle);
  p_context->ReplaceDim(inferred_shape_handle, 0, p_context->UnknownDim(), &inferred_shape_handle);
  for (int d = 1; d < inferred_shape.dims(); ++d) {
    if (inferred_shape.dim_size(d) <= 0) {
      p_context->ReplaceDim(inferred_shape_handle, d, p_context->UnknownDim(), &inferred_shape_handle);
    }
  }

  return inferred_shape_handle;
}

template <class TDevice>
tf::TensorShape NiftyRegResampleOp<TDevice>::compute_gradient_shape(const tf::TensorShape &image_shape, const tf::TensorShape &deformation_shape) {
  tf::TensorShape shape = deformation_shape;

  shape.set_dim(1, image_shape.dim_size(1)*deformation_shape.dim_size(1));

  return shape;
}

template <class TDevice>
tf::TensorShape NiftyRegResampleOp<TDevice>::compute_gradient_shape(tf::OpKernelContext *p_context) {
  return NiftyRegResampleOp::compute_gradient_shape(NiftyRegResampleOp::extract_image_shape(p_context), NiftyRegResampleOp::extract_deformation_shape(p_context));
}

template <class TDevice>
tf::shape_inference::ShapeHandle NiftyRegResampleOp<TDevice>::compute_gradient_shape(tf::shape_inference::InferenceContext *p_context) {
  tf::shape_inference::ShapeHandle inferred_shape = p_context->input(1);

  if (p_context->ValueKnown(p_context->Dim(inferred_shape, 1)) && p_context->Value(p_context->Dim(inferred_shape, 1)) > 0
      && p_context->ValueKnown(p_context->Dim(p_context->input(0), 1)) && p_context->Value(p_context->Dim(p_context->input(0), 1)) > 0) {
    const int dim_size = p_context->Value(p_context->Dim(inferred_shape, 1))*p_context->Value(p_context->Dim(p_context->input(0), 1));

    p_context->ReplaceDim(inferred_shape, 1, p_context->MakeDim(dim_size), &inferred_shape);
  } else {
    std::cerr << "Warning: computing gradient with unknown number of per-voxel components." << std::endl;
    p_context->ReplaceDim(inferred_shape, 1, p_context->UnknownDim(), &inferred_shape);
  }

  return inferred_shape;
}

template <class TDevice>
void NiftyRegResampleOp<TDevice>::load_nifti_dimensions_from_tensors(nifti_image &r_image, nifti_image &r_deformation, const tf::Tensor &image_tensor, const tf::Tensor &deformation_tensor) {
  NiftyRegResampleOp::load_nifti_dimensions_from_tensor(r_image, image_tensor);
  NiftyRegResampleOp::load_nifti_dimensions_from_tensor(r_deformation, deformation_tensor);
}

template <class TDevice>
void NiftyRegResampleOp<TDevice>::load_nifti_dimensions_from_tensor(nifti_image &r_image, const tf::Tensor &tensor) {
  std::fill(r_image.dim + 1, r_image.dim + 8, 1.f);
  std::fill(r_image.pixdim, r_image.pixdim + 8, 1.f);

  if (tensor.dim_size(1) != 1) {
    r_image.dim[0] = 5;
    r_image.dim[5] = tensor.dim_size(1);
  } else {
    r_image.dim[0] = tensor.dims() - 2;
  }

  for (int i = 1; i <= tensor.dims() - 2; ++i) {
    r_image.dim[i] = tensor.dim_size(tensor.dims() - i);
  }
  r_image.datatype = NIFTI_TYPE_FLOAT32;
  r_image.nbyper = 4;
  nifti_update_dims_from_array(&r_image);
}

template <class TDevice>
NiftyRegResampleOp<TDevice>::NiftyRegResampleOp(tf::OpKernelConstruction* p_context) : Superclass(p_context) {
  int bdy_code = int(resampler_boundary_e::SENTINEL);

  p_context->GetAttr("interpolation", &m_interpolation);
  p_context->GetAttr("boundary", &bdy_code);

  OP_REQUIRES(p_context, bdy_code >= int(resampler_boundary_e::ZEROPAD) && bdy_code < int(resampler_boundary_e::SENTINEL),
              tf::errors::InvalidArgument("Invalid boundary code."));
  m_boundary = resampler_boundary_e(bdy_code);
}
