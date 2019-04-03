#pragma once

#include "interpolations.h"

template <typename TCoord, typename TBasis>
NR_HOST_DEV void reg_getNiftynetCubicSpline(const TCoord relative, TBasis *p_basis) {
  const TCoord sqr_relative = relative*relative;

  p_basis[0] = (((-relative + 3)*relative - 3)*relative + 1)/6;
  p_basis[1] = ((3*relative - 6)*sqr_relative + 4)/6;
  p_basis[2] = (((-3*relative + 3)*relative + 3)*relative + 1)/6;
  p_basis[3] = sqr_relative*relative/6;
}

template <typename TCoord, typename TBasis>
NR_HOST_DEV void reg_getNiftynetCubicSplineDerivative(const TCoord relative, TBasis *p_basis) {
  p_basis[0] = ((-3*relative + 6)*relative - 3)/6;
  p_basis[1] = (9*relative - 12)*relative/6;
  p_basis[2] = ((-9*relative + 6)*relative + 3)/6;
  p_basis[3] = relative*relative/2;
}
