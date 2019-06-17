#pragma once

#include "resampler_boundary.h"

#include <cstdlib>
#include <limits>
#include <type_traits>

template <const resampler_boundary_e tBoundary>
NR_HOST_DEV int reg_applyBoundary(const int idx, const int bound) {
  int bdyIdx = idx;

  switch (tBoundary) {
  case resampler_boundary_e::CLAMPING:
    bdyIdx = bdyIdx >= 0? bdyIdx : 0;
    bdyIdx = bdyIdx < bound? bdyIdx : bound - 1;
    break;

  case resampler_boundary_e::REFLECTING: {
    const int wrap_size = 2*bound - 2;

    bdyIdx = bound - 1 - std::abs(bound - 1 - (bdyIdx%wrap_size + wrap_size)%wrap_size);
    break;
  }
  }

  return bdyIdx;
}
/* *************************************************************** */
template <const resampler_boundary_e tBoundary, typename TIndex, typename TBound>
NR_HOST_DEV bool reg_checkImageDimensionIndex(const TIndex index, const TBound bound) {
  return resampler_boundary_e(tBoundary) == resampler_boundary_e::CLAMPING || resampler_boundary_e(tBoundary) == resampler_boundary_e::REFLECTING || (index >= 0 && index < bound);
}
/* *************************************************************** */
template <typename TVoxel>
NR_HOST_DEV constexpr TVoxel reg_getPaddingValue(const resampler_boundary_e boundary) {
  return boundary == resampler_boundary_e::ZEROPAD? TVoxel(0)
    : (std::is_integral<TVoxel>::value? std::numeric_limits<TVoxel>::lowest() : std::numeric_limits<TVoxel>::quiet_NaN());
}
