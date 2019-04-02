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
template <const bool tDoClamp, const bool tDoReflect, typename TIndex, typename TBound>
NR_HOST_DEV TIndex reg_applyBoundary(const TIndex idx, const TBound bound) {
  TIndex bdyIdx = idx;

#ifndef __CUDACC__
  static_assert(!(tDoReflect && tDoClamp), "clamping and reflecting cannot be requested at the same time.");
#endif

  if (tDoClamp) {
    bdyIdx = bdyIdx >= 0? bdyIdx : 0;
    bdyIdx = bdyIdx < bound? bdyIdx : bound - 1;
  } else if (tDoReflect) {
    bdyIdx = bdyIdx%(2*bound - 2);
    bdyIdx = bdyIdx <= -bound? std::abs(bound + bdyIdx)
      : (bdyIdx < 0? std::abs(bdyIdx)
         : (bdyIdx >= bound? bound - (bdyIdx - bound + 2)
            : bdyIdx));
  }

  return bdyIdx;
}
/* *************************************************************** */
/** \returns the appropriate padding value for a given boundary treatment */
template <typename TVoxel>
NR_HOST_DEV constexpr TVoxel get_padding_value(const resampler_boundary_e boundary) {
  return boundary == resampler_boundary_e::ZEROPAD? TVoxel(0)
    : (std::is_integral<TVoxel>::value? std::numeric_limits<TVoxel>::lowest() : std::numeric_limits<TVoxel>::quiet_NaN());
}
