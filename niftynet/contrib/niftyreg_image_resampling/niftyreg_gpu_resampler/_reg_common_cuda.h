/** @file _reg_common_gpu.h
 * @author Marc Modat
 * @date 25/03/2009.
 * Copyright (c) 2009, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 */

#ifndef _REG_COMMON_GPU_H
#define _REG_COMMON_GPU_H

#include "nifti1_io.h"
#include "cuda_runtime.h"
#include "cuda.h"

/* ******************************** */
/* ******************************** */
#ifndef __VECTOR_TYPES_H__
#define __VECTOR_TYPES_H__
struct __attribute__((aligned(4))) float4
{
	float x,y,z,w;
};
#endif
/* ******************************** */
/* ******************************** */
#if CUDART_VERSION >= 3200
#   define NR_CUDA_SAFE_CALL(call) { \
		call; \
		cudaError err = cudaPeekAtLastError(); \
		if( cudaSuccess != err) { \
			fprintf(stderr, "[NiftyReg CUDA ERROR] file '%s' in line %i : %s.\n", \
			__FILE__, __LINE__, cudaGetErrorString(err)); \
			reg_exit(); \
		} \
	}
#   define NR_CUDA_CHECK_KERNEL(grid,block) { \
		cudaThreadSynchronize(); \
		cudaError err = cudaPeekAtLastError(); \
		if( err != cudaSuccess) { \
			fprintf(stderr, "[NiftyReg CUDA ERROR] file '%s' in line %i : %s.\n", \
			__FILE__, __LINE__, cudaGetErrorString(err)); \
			fprintf(stderr, "Grid [%ix%ix%i] | Block [%ix%ix%i]\n", \
			grid.x,grid.y,grid.z,block.x,block.y,block.z); \
			reg_exit(); \
		} \
		else{\
			printf("[NiftyReg CUDA DEBUG] kernel: %s - Grid size [%i %i %i] - Block size [%i %i %i]\n", \
			  cudaGetErrorString(cudaGetLastError()), grid.x, grid.y, grid.z, block.x, block.y, block.z);\
		}\
	}
#else //CUDART_VERSION >= 3200
#   define NR_CUDA_SAFE_CALL(call) { \
		call; \
		cudaError err = cudaThreadSynchronize(); \
		if( cudaSuccess != err) { \
			fprintf(stderr, "[NiftyReg CUDA ERROR] file '%s' in line %i : %s.\n", \
			__FILE__, __LINE__, cudaGetErrorString(err)); \
			reg_exit(); \
		} \
	}
#   define NR_CUDA_CHECK_KERNEL(grid,block) { \
		cudaError err = cudaThreadSynchronize(); \
		if( err != cudaSuccess) { \
			fprintf(stderr, "[NiftyReg CUDA ERROR] file '%s' in line %i : %s.\n", \
			__FILE__, __LINE__, cudaGetErrorString(err)); \
			fprintf(stderr, "Grid [%ix%ix%i] | Block [%ix%ix%i]\n", \
			grid.x,grid.y,grid.z,block.x,block.y,block.z); \
			reg_exit(); \
		} \
	}
#endif //CUDART_VERSION >= 3200
/* ******************************** */
/* ******************************** */
/** \brief Computes a reasonable grid configuration for resampling in a given reference space */
void cudaCommon_computeGridConfiguration(dim3 &r_blocks, dim3 &r_grid, const int targetVoxelNumber);

/* ******************************** */
int cudaCommon_setCUDACard(CUcontext *ctx,
									bool verbose);
/* ******************************** */
void cudaCommon_unsetCUDACard(CUcontext *ctx);
/* ******************************** */
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(cudaArray **, int *);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(cudaArray **, cudaArray **, int *);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(DTYPE **, int);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(DTYPE **, int *);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(DTYPE **, DTYPE **, int *);
/* ******************************** */
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray **, nifti_image *);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray **, cudaArray **, const nifti_image *);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(DTYPE **, const nifti_image *);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(DTYPE **, DTYPE **, nifti_image *);
/* ******************************** */
/* ******************************** */
extern "C++"
template <class DTYPE, class DTYPE2>
int cudaCommon_transferFromDeviceToNifti1(nifti_image *, DTYPE **);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferFromDeviceToNifti(nifti_image *, DTYPE **);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferFromDeviceToNifti(nifti_image *, DTYPE **, DTYPE **);
/* ******************************** */
/* ******************************** */
extern "C++"
void cudaCommon_free(cudaArray **);
/* ******************************** */
extern "C++" template <class DTYPE>
void cudaCommon_free(DTYPE **);
/* ******************************** */
/* ******************************** */
extern "C++" template <class DTYPE>
int cudaCommon_allocateNiftiToDevice(nifti_image **image_d, int *dim);

template <class DTYPE>
int cudaCommon_transferNiftiToNiftiOnDevice1(nifti_image **image_d, nifti_image *img);


/* ******************************** */
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferFromDeviceToNiftiSimple(DTYPE **, nifti_image * );

extern "C++"
template <class DTYPE>
int cudaCommon_transferFromDeviceToNiftiSimple1(DTYPE **array_d, DTYPE *img, const unsigned  nvox);

extern "C++"
template <class DTYPE>
int cudaCommon_transferFromDeviceToCpu(DTYPE *cpuPtr, DTYPE **cuPtr, const unsigned int nElements);
/* ******************************** */
/* ******************************** */
/* ******************************** */
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferArrayFromCpuToDevice(DTYPE *array_d, const DTYPE *array_cpu, const unsigned int nElements);
/* ******************************** */
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferArrayFromDeviceToCpu(DTYPE *array_cpu, DTYPE *array_d, const unsigned int nElements);
/* ******************************** */
/* ******************************** */
void showCUDACardInfo(void);
/* ******************************** */
#endif
