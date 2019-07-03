#pragma once

#include "resampler_boundary.h"

/**
 * \brief Cubic spline formula matching the one from niftynet.layer.resampler
 * \param relative floating point index relative to kernel base index.
 */
template <typename TCoord, typename TBasis>
NR_HOST_DEV void reg_getNiftynetCubicSpline(const TCoord relative, TBasis *p_basis);

/**
 * \brief Analytic derivative of cubic spline formula matching the one from niftynet.layer.resampler
 * \param relative floating point index relative to kernel base index.
 */
template <typename TCoord, typename TBasis>
NR_HOST_DEV void reg_getNiftynetCubicSplineDerivative(const TCoord relative, TBasis *p_basis);

#include "interpolations.tpp"
