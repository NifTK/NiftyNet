#pragma once

#include <cstdlib>
#include <limits>
#include <type_traits>

/** \brief Resampler boundary treatment options */
enum class resampler_boundary_e {
  ZEROPAD = 0, /**< Zero-padding */
  NANPAD, /**< NaN-padding */
  CLAMPING, /**< Clamp to nearest boundary voxel intensity */
  REFLECTING, /**< Reflect indices at boundaries */
  SENTINEL, /**< Sentinel code */
};

/* *************************************************************** */
#ifdef __CUDACC__
#define NR_HOST_DEV __host__ __device__
#else
#define NR_HOST_DEV
#endif
/* *************************************************************** */
/**
 * \brief Boundary index modification function
 * \tparam tDoClamp Clamping boundary
 * \tparam tDoReflect Clamping boundary
 * \returns an appropriately modified index
 */
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
  return tBoundary == resampler_boundary_e::CLAMPING || tBoundary == resampler_boundary_e::REFLECTING || (index >= 0 && index < bound);
}

/* *************************************************************** */
/** \returns the appropriate padding value for a given boundary treatment */
template <typename TVoxel>
NR_HOST_DEV constexpr TVoxel reg_getPaddingValue(const resampler_boundary_e boundary) {
  return boundary == resampler_boundary_e::ZEROPAD? TVoxel(0)
    : (std::is_integral<TVoxel>::value? std::numeric_limits<TVoxel>::lowest() : std::numeric_limits<TVoxel>::quiet_NaN());
}
