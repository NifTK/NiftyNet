#pragma once

#ifdef __CUDACC__
#define NR_HOST_DEV __host__ __device__
#else
#define NR_HOST_DEV
#endif

/** \brief Resampler boundary treatment options */
enum class resampler_boundary_e {
  ZEROPAD = 0, /**< Zero-padding */
  NANPAD, /**< NaN-padding */
  CLAMPING, /**< Clamp to nearest boundary voxel intensity */
  REFLECTING, /**< Reflect indices at boundaries */
  SENTINEL, /**< Sentinel code */
};

/**
 * \brief Boundary index modification function
 * \tparam tBoundary boundary treatment enum value
 * \returns an appropriately modified index
 */
template <const resampler_boundary_e tBoundary>
NR_HOST_DEV int reg_applyBoundary(const int idx, const int bound);

/**
 * \param bound upper bound on index
 * \tparam tBoundary boundary treatment enum value
 * \returns true if the argument index lies between 0 (incl) and bound (excl), or the index is guearanteed to be valid by virtue of the applied boundary treatment.
 */
template <const resampler_boundary_e tBoundary, typename TIndex, typename TBound>
NR_HOST_DEV bool reg_checkImageDimensionIndex(const TIndex index, const TBound bound);

/** \returns the appropriate padding value for a given boundary treatment enum value */
template <typename TVoxel>
NR_HOST_DEV constexpr TVoxel reg_getPaddingValue(const resampler_boundary_e boundary);

#include "resampler_boundary.tpp"
