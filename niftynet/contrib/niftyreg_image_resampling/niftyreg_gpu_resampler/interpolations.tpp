#pragma once

#include "interpolations.h"

template <typename TCoord, typename TBasis>
NR_HOST_DEV void reg_getNiftynetCubicSpline(const TCoord relative, TBasis *p_basis) {
  const TCoord FF = relative*relative;

  p_basis[0] = (-FF*relative + 3*FF - 3*relative + 1)/6;
  p_basis[1] = (3*FF*relative - 6*FF + 4)/6;
  p_basis[2] = (-3*FF*relative + 3*FF + 3*relative + 1)/6;
  p_basis[3] = FF*relative/6;
}

template <typename TCoord, typename TBasis>
NR_HOST_DEV void reg_getNiftynetCubicSplineDerivative(const TCoord relative, TBasis *p_basis) {
  p_basis[0] = ((-3*relative + 6)*relative - 3)/6;
  p_basis[1] = (9*relative - 12)*relative/6;
  p_basis[2] = ((-9*relative + 6)*relative + 3)/6;
  p_basis[3] = relative*relative/2;
}
