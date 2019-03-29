#pragma once

#include <limits>
#include <type_traits>

/** \brief Resampler boundary treatment options */
enum class resampler_boundary_e {
  ZEROPAD = 0, /**< Zero-padding */
  NANPAD, /**< NaN-padding */
  CLAMPING, /**< Clamp to nearest boundary voxel intensity */
  SENTINEL, /**< Sentinel code */
};

/** \returns the appropriate padding value for a given boundary treatment */
template <typename TVoxel>
#ifdef __CUDACC__
__host__ __device__
#endif
constexpr TVoxel get_padding_value(const resampler_boundary_e boundary) {
  return boundary == resampler_boundary_e::ZEROPAD? TVoxel(0)
    : (std::is_integral<TVoxel>::value? std::numeric_limits<TVoxel>::lowest() : std::numeric_limits<TVoxel>::quiet_NaN());
}
