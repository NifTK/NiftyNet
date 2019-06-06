/**
 * @file _reg_comon_gpu.cu
 * @author Marc Modat
 * @date 25/03/2009
 * Copyright (c) 2009, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_COMMON_GPU_CU
#define _REG_COMMON_GPU_CU

#include "_reg_common_cuda.h"
#include "_reg_tools.h"
/* ******************************** */
void cudaCommon_computeGridConfiguration(dim3 &r_blocks, dim3 &r_grid, const int targetVoxelNumber) {
  unsigned int maxThreads = 256;
  unsigned int maxBlocks = 65365;
  unsigned int blocks = (targetVoxelNumber % maxThreads) ? (targetVoxelNumber / maxThreads) + 1 : targetVoxelNumber / maxThreads;

  blocks = (std::min)(blocks, maxBlocks);

  r_grid = dim3(blocks, 1, 1);
  r_blocks = dim3(maxThreads, 1, 1);
}
/* ******************************** */
/* ******************************** */
template <class NIFTI_TYPE>
int cudaCommon_transferNiftiToNiftiOnDevice1(nifti_image **image_d, nifti_image *img) {

	const unsigned int memSize = img->dim[1] * img->dim[2] * img->dim[3] * sizeof(NIFTI_TYPE);

	int *g_dim;
	float* g_pixdim;
	NIFTI_TYPE* g_data;

	NR_CUDA_SAFE_CALL(cudaMalloc((void**)&g_dim, 8 * sizeof(int)));
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)&g_pixdim, 8 * sizeof(float)));
	NR_CUDA_SAFE_CALL(cudaMalloc((void**)&g_data, memSize));

	NIFTI_TYPE *array_h = static_cast<NIFTI_TYPE *>( img->data );
	NR_CUDA_SAFE_CALL(cudaMemcpy(( *image_d ), img, sizeof(nifti_image), cudaMemcpyHostToDevice));

	NR_CUDA_SAFE_CALL(cudaMemcpy((*image_d)->data, array_h, memSize, cudaMemcpyHostToDevice));
	NR_CUDA_SAFE_CALL(cudaMemcpy(( *image_d )->dim, img->dim, 8 * sizeof(int), cudaMemcpyHostToDevice));
	NR_CUDA_SAFE_CALL(cudaMemcpy(( *image_d )->pixdim, img->pixdim, 8 * sizeof(float), cudaMemcpyHostToDevice));

	return EXIT_SUCCESS;
}
template int cudaCommon_transferNiftiToNiftiOnDevice1<float>(nifti_image **image_d, nifti_image *img);
template int cudaCommon_transferNiftiToNiftiOnDevice1<double>(nifti_image **image_d, nifti_image *img);
/* ******************************** */
/* ******************************** */
template <class DTYPE, class NIFTI_TYPE>
int cudaCommon_transferNiftiToArrayOnDevice1(DTYPE **array_d, const nifti_image *img)
{
	if(sizeof(DTYPE)!=sizeof(NIFTI_TYPE)){
		reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice1");
		reg_print_msg_error("The host and device arrays are of different types");
		return EXIT_FAILURE;
	}
	else{
		const unsigned int memSize = img->nvox*sizeof(DTYPE);
		const NIFTI_TYPE *array_h=static_cast<NIFTI_TYPE *>(img->data);
		NR_CUDA_SAFE_CALL(cudaMemcpy(*array_d, array_h, memSize, cudaMemcpyHostToDevice));
	}
	return EXIT_SUCCESS;
}
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(DTYPE **array_d, const nifti_image *img)
{
	if( sizeof(DTYPE)==sizeof(float4) ){
		if( (img->datatype!=NIFTI_TYPE_FLOAT32) || (img->dim[5]<2) || (img->dim[4]>1)){
			reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice");
			reg_print_msg_error("The specified image is not a single precision deformation field image");
			return EXIT_FAILURE;
		}
		const float *niftiImgValues = static_cast<float *>(img->data);
		float4 *array_h=(float4 *)calloc(img->nx*img->ny*img->nz,sizeof(float4));
		const int voxelNumber = img->nx*img->ny*img->nz;
		for(int i=0; i<voxelNumber; i++)
			array_h[i].x= *niftiImgValues++;
		if(img->dim[5]>=2){
			for(int i=0; i<voxelNumber; i++)
				array_h[i].y= *niftiImgValues++;
		}
		if(img->dim[5]>=3){
			for(int i=0; i<voxelNumber; i++)
				array_h[i].z= *niftiImgValues++;
		}
		if(img->dim[5]>=4){
			for(int i=0; i<voxelNumber; i++)
				array_h[i].w= *niftiImgValues++;
		}
		NR_CUDA_SAFE_CALL(cudaMemcpy(*array_d, array_h, img->nx*img->ny*img->nz*sizeof(float4), cudaMemcpyHostToDevice));
		free(array_h);
	}
	else{ // All these else could be removed but the nvcc compiler would warn for unreachable statement
		switch(img->datatype){
		case NIFTI_TYPE_FLOAT32:
			return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,float>(array_d, img);
		default:
			reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice");
			reg_print_msg_error("The image data type is not supported");
			return EXIT_FAILURE;
		}
	}
	return EXIT_SUCCESS;
}
template int cudaCommon_transferNiftiToArrayOnDevice<double>(double **, const nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<float>(float **, const nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<int>(int **, const nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<float4>(float4 **, const nifti_image *);
/* ******************************** */

template <class DTYPE, class NIFTI_TYPE>
int cudaCommon_transferNiftiToArrayOnDevice1(DTYPE **array_d, DTYPE **array2_d, nifti_image *img)
{
	if(sizeof(DTYPE)!=sizeof(NIFTI_TYPE)){
		reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice1");
		reg_print_msg_error("The host and device arrays are of different types");
		return EXIT_FAILURE;
	}
	else{
		const unsigned int memSize = img->dim[1] * img->dim[2] * img->dim[3] * sizeof(DTYPE);
		NIFTI_TYPE *array_h=static_cast<NIFTI_TYPE *>(img->data);
		NIFTI_TYPE *array2_h=&array_h[img->dim[1] * img->dim[2] * img->dim[3]];
		NR_CUDA_SAFE_CALL(cudaMemcpy(*array_d, array_h, memSize, cudaMemcpyHostToDevice));
		NR_CUDA_SAFE_CALL(cudaMemcpy(*array2_d, array2_h, memSize, cudaMemcpyHostToDevice));
	}
	return EXIT_SUCCESS;
}
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(DTYPE **array_d, DTYPE **array2_d, nifti_image *img)
{
	if(sizeof(DTYPE)==sizeof(float4) ){
		if( (img->datatype!=NIFTI_TYPE_FLOAT32) || (img->dim[5]<2) || (img->dim[4]>1)){
			reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice");
			reg_print_msg_error("The specified image is not a single precision deformation field image");
			return EXIT_FAILURE;
		}
		float *niftiImgValues = static_cast<float *>(img->data);
		float4 *array_h=(float4 *)calloc(img->nx*img->ny*img->nz,sizeof(float4));
		float4 *array2_h=(float4 *)calloc(img->nx*img->ny*img->nz,sizeof(float4));
		const int voxelNumber = img->nx*img->ny*img->nz;
		for(int i=0; i<voxelNumber; i++)
			array_h[i].x= *niftiImgValues++;
		for(int i=0; i<voxelNumber; i++)
			array2_h[i].x= *niftiImgValues++;
		if(img->dim[5]>=2){
			for(int i=0; i<voxelNumber; i++)
				array_h[i].y= *niftiImgValues++;
			for(int i=0; i<voxelNumber; i++)
				array2_h[i].y= *niftiImgValues++;
		}
		if(img->dim[5]>=3){
			for(int i=0; i<voxelNumber; i++)
				array_h[i].z= *niftiImgValues++;
			for(int i=0; i<voxelNumber; i++)
				array2_h[i].z= *niftiImgValues++;
		}
		if(img->dim[5]>=4){
			for(int i=0; i<voxelNumber; i++)
				array_h[i].w= *niftiImgValues++;
			for(int i=0; i<voxelNumber; i++)
				array2_h[i].w= *niftiImgValues++;
		}
		NR_CUDA_SAFE_CALL(cudaMemcpy(*array_d, array_h, img->nx*img->ny*img->nz*sizeof(float4), cudaMemcpyHostToDevice));
		NR_CUDA_SAFE_CALL(cudaMemcpy(*array2_d, array2_h, img->nx*img->ny*img->nz*sizeof(float4), cudaMemcpyHostToDevice));
		free(array_h);
		free(array2_h);
	}
	else{ // All these else could be removed but the nvcc compiler would warn for unreachable statement
		switch(img->datatype){
		case NIFTI_TYPE_FLOAT32:
			return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,float>(array_d, array2_d, img);
		default:
			reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice");
			reg_print_msg_error("The image data type is not supported");
			return EXIT_FAILURE;
		}
	}
	return EXIT_SUCCESS;
}
template int cudaCommon_transferNiftiToArrayOnDevice<float>(float **,float **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<double>(double **,double **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<float4>(float4 **,float4 **, nifti_image *); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE, class NIFTI_TYPE>
int cudaCommon_transferNiftiToArrayOnDevice1(cudaArray **cuArray_d, nifti_image *img)
{
	if(sizeof(DTYPE)!=sizeof(NIFTI_TYPE)){
		reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice1");
		reg_print_msg_error("The host and device arrays are of different types");
		return EXIT_FAILURE;
	}
	else{
		NIFTI_TYPE *array_h=static_cast<NIFTI_TYPE *>(img->data);

		cudaMemcpy3DParms copyParams; memset(&copyParams, 0, sizeof(copyParams));
		copyParams.extent = make_cudaExtent(img->dim[1], img->dim[2], img->dim[3]);
		copyParams.srcPtr = make_cudaPitchedPtr((void *) array_h,
												copyParams.extent.width*sizeof(DTYPE),
												copyParams.extent.width,
												copyParams.extent.height);
		copyParams.dstArray = *cuArray_d;
		copyParams.kind = cudaMemcpyHostToDevice;
		NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
	}
	return EXIT_SUCCESS;
}
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray **cuArray_d, nifti_image *img)
{
	if( sizeof(DTYPE)==sizeof(float4) ){
		if( (img->datatype!=NIFTI_TYPE_FLOAT32) || (img->dim[5]<2) || (img->dim[4]>1) ){
			reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice");
			reg_print_msg_error("The specified image is not a single precision deformation field image");
			return EXIT_FAILURE;
		}
		float *niftiImgValues = static_cast<float *>(img->data);
		float4 *array_h=(float4 *)calloc(img->nx*img->ny*img->nz,sizeof(float4));

		for(int i=0; i<img->nx*img->ny*img->nz; i++)
			array_h[i].x= *niftiImgValues++;
		if(img->dim[5]>=2)
		{
			for(int i=0; i<img->nx*img->ny*img->nz; i++)
				array_h[i].y= *niftiImgValues++;
		}
		if(img->dim[5]>=3)
		{
			for(int i=0; i<img->nx*img->ny*img->nz; i++)
				array_h[i].z= *niftiImgValues++;
		}
		if(img->dim[5]==3)
		{
			for(int i=0; i<img->nx*img->ny*img->nz; i++)
				array_h[i].w= *niftiImgValues++;
		}
		cudaMemcpy3DParms copyParams; memset(&copyParams, 0, sizeof(copyParams));
		copyParams.extent = make_cudaExtent(img->dim[1], img->dim[2], img->dim[3]);
		copyParams.srcPtr = make_cudaPitchedPtr((void *) array_h,
												copyParams.extent.width*sizeof(DTYPE),
												copyParams.extent.width,
												copyParams.extent.height);
		copyParams.dstArray = *cuArray_d;
		copyParams.kind = cudaMemcpyHostToDevice;
		NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams))
		free(array_h);
	}
	else{ // All these else could be removed but the nvcc compiler would warn for unreachable statement
		switch(img->datatype){
		case NIFTI_TYPE_FLOAT32:
			return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,float>(cuArray_d, img);
		default:
			reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice");
			reg_print_msg_error("The image data type is not supported");
			return EXIT_FAILURE;
		}
	}
	return EXIT_SUCCESS;
}
template int cudaCommon_transferNiftiToArrayOnDevice<float>(cudaArray **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<double>(cudaArray **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<int>(cudaArray **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<float4>(cudaArray **, nifti_image *); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE, class NIFTI_TYPE>
int cudaCommon_transferNiftiToArrayOnDevice1(cudaArray **cuArray_d, cudaArray **cuArray2_d, nifti_image *img)
{
	if(sizeof(DTYPE)!=sizeof(NIFTI_TYPE)){
		reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice1");
		reg_print_msg_error("The host and device arrays are of different types");
		return EXIT_FAILURE;
	}
	else{
		NIFTI_TYPE *array_h = static_cast<NIFTI_TYPE *>(img->data);
		NIFTI_TYPE *array2_h = &array_h[img->dim[1]*img->dim[2]*img->dim[3]];

		cudaMemcpy3DParms copyParams; memset(&copyParams, 0, sizeof(copyParams));
		copyParams.extent = make_cudaExtent(img->dim[1], img->dim[2], img->dim[3]);
		copyParams.kind = cudaMemcpyHostToDevice;
		// First timepoint
		copyParams.srcPtr = make_cudaPitchedPtr((void *) array_h,
												copyParams.extent.width*sizeof(DTYPE),
												copyParams.extent.width,
												copyParams.extent.height);
		copyParams.dstArray = *cuArray_d;
		NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
		// Second timepoint
		copyParams.srcPtr = make_cudaPitchedPtr((void *) array2_h,
												copyParams.extent.width*sizeof(DTYPE),
												copyParams.extent.width,
												copyParams.extent.height);
		copyParams.dstArray = *cuArray2_d;
		NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
	}
	return EXIT_SUCCESS;
}
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray **cuArray_d, cudaArray **cuArray2_d, nifti_image *img)
{
	if( sizeof(DTYPE)==sizeof(float4) ){
		if( (img->datatype!=NIFTI_TYPE_FLOAT32) || (img->dim[5]<2) || (img->dim[4]>1) )
		{
			reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice1");
			reg_print_msg_error("The specified image is not a single precision deformation field image");
			return EXIT_FAILURE;
		}
		float *niftiImgValues = static_cast<float *>(img->data);
		float4 *array_h=(float4 *)calloc(img->nx*img->ny*img->nz,sizeof(float4));
		float4 *array2_h=(float4 *)calloc(img->nx*img->ny*img->nz,sizeof(float4));

		for(int i=0; i<img->nx*img->ny*img->nz; i++)
			array_h[i].x= *niftiImgValues++;
		for(int i=0; i<img->nx*img->ny*img->nz; i++)
			array2_h[i].x= *niftiImgValues++;

		if(img->dim[5]>=2){
			for(int i=0; i<img->nx*img->ny*img->nz; i++)
				array_h[i].y= *niftiImgValues++;
			for(int i=0; i<img->nx*img->ny*img->nz; i++)
				array2_h[i].y= *niftiImgValues++;
		}

		if(img->dim[5]>=3){
			for(int i=0; i<img->nx*img->ny*img->nz; i++)
				array_h[i].z= *niftiImgValues++;
			for(int i=0; i<img->nx*img->ny*img->nz; i++)
				array2_h[i].z= *niftiImgValues++;
		}

		if(img->dim[5]==3){
			for(int i=0; i<img->nx*img->ny*img->nz; i++)
				array_h[i].w= *niftiImgValues++;
			for(int i=0; i<img->nx*img->ny*img->nz; i++)
				array2_h[i].w= *niftiImgValues++;
		}

		cudaMemcpy3DParms copyParams; memset(&copyParams, 0, sizeof(copyParams));
		copyParams.extent = make_cudaExtent(img->dim[1], img->dim[2], img->dim[3]);
		copyParams.kind = cudaMemcpyHostToDevice;
		// First timepoint
		copyParams.srcPtr = make_cudaPitchedPtr((void *) array_h,
												copyParams.extent.width*sizeof(DTYPE),
												copyParams.extent.width,
												copyParams.extent.height);
		copyParams.dstArray = *cuArray_d;
		NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
		free(array_h);
		// Second timepoint
		copyParams.srcPtr = make_cudaPitchedPtr((void *) array2_h,
												copyParams.extent.width*sizeof(DTYPE),
												copyParams.extent.width,
												copyParams.extent.height);
		copyParams.dstArray = *cuArray2_d;
		NR_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
		free(array2_h);
	}
	else{ // All these else could be removed but the nvcc compiler would warn for unreachable statement
		switch(img->datatype){
		case NIFTI_TYPE_FLOAT32:
			return cudaCommon_transferNiftiToArrayOnDevice1<DTYPE,float>(cuArray_d, cuArray2_d, img);
		default:
			reg_print_fct_error("cudaCommon_transferNiftiToArrayOnDevice1");
			reg_print_msg_error("The image data type is not supported");
			return EXIT_FAILURE;
		}
	}
	return EXIT_SUCCESS;
}
template int cudaCommon_transferNiftiToArrayOnDevice<float>(cudaArray **, cudaArray **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<double>(cudaArray **, cudaArray **, nifti_image *);
template int cudaCommon_transferNiftiToArrayOnDevice<float4>(cudaArray **, cudaArray **, nifti_image *); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(cudaArray **cuArray_d, int *dim)
{
	const cudaExtent volumeSize = make_cudaExtent(dim[1], dim[2], dim[3]);
	cudaChannelFormatDesc texDesc = cudaCreateChannelDesc<DTYPE>();
	NR_CUDA_SAFE_CALL(cudaMalloc3DArray(cuArray_d, &texDesc, volumeSize));
	return EXIT_SUCCESS;
}template int cudaCommon_allocateArrayToDevice<float>(cudaArray **, int *);
template int cudaCommon_allocateArrayToDevice<double>(cudaArray **, int *);
template int cudaCommon_allocateArrayToDevice<float4>(cudaArray **, int *); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(cudaArray **cuArray_d, cudaArray **cuArray2_d, int *dim)
{
	const cudaExtent volumeSize = make_cudaExtent(dim[1], dim[2], dim[3]);
	cudaChannelFormatDesc texDesc = cudaCreateChannelDesc<DTYPE>();
	NR_CUDA_SAFE_CALL(cudaMalloc3DArray(cuArray_d, &texDesc, volumeSize));
	NR_CUDA_SAFE_CALL(cudaMalloc3DArray(cuArray2_d, &texDesc, volumeSize));
	return EXIT_SUCCESS;
}
template int cudaCommon_allocateArrayToDevice<float>(cudaArray **,cudaArray **, int *);
template int cudaCommon_allocateArrayToDevice<double>(cudaArray **,cudaArray **, int *);
template int cudaCommon_allocateArrayToDevice<float4>(cudaArray **,cudaArray **, int *); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(DTYPE **array_d, int *dim)
{
	const unsigned int memSize = dim[1] * dim[2] * dim[3] * sizeof(DTYPE);
	NR_CUDA_SAFE_CALL(cudaMalloc(array_d, memSize));
	return EXIT_SUCCESS;
}
template int cudaCommon_allocateArrayToDevice<float>(float **, int *);
template int cudaCommon_allocateArrayToDevice<double>(double **, int *);
template int cudaCommon_allocateArrayToDevice<int>(int **, int *);
template int cudaCommon_allocateArrayToDevice<float4>(float4 **, int *); // for deformation field
/* ******************************** */
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(DTYPE **array_d, int vox)
{
	const unsigned int memSize = vox * sizeof(DTYPE);
	NR_CUDA_SAFE_CALL(cudaMalloc(array_d, memSize));
	return EXIT_SUCCESS;
}
template int cudaCommon_allocateArrayToDevice<float>(float **, int);
template int cudaCommon_allocateArrayToDevice<double>(double **, int);
template int cudaCommon_allocateArrayToDevice<int>(int **, int);
template int cudaCommon_allocateArrayToDevice<float4>(float4 **, int); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(DTYPE **array_d, DTYPE **array2_d, int *dim)
{
	const unsigned int memSize = dim[1] * dim[2] * dim[3] * sizeof(DTYPE);
	NR_CUDA_SAFE_CALL(cudaMalloc(array_d, memSize));
	NR_CUDA_SAFE_CALL(cudaMalloc(array2_d, memSize));
	return EXIT_SUCCESS;
}
template int cudaCommon_allocateArrayToDevice<float>(float **, float **, int *);
template int cudaCommon_allocateArrayToDevice<double>(double **, double **, int *);
template int  cudaCommon_allocateArrayToDevice<float4>(float4 **, float4 **, int *); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferFromDeviceToCpu(DTYPE *cpuPtr, DTYPE **cuPtr, const unsigned int nElements)
{

	NR_CUDA_SAFE_CALL(cudaMemcpy((void *)cpuPtr, (void *)*cuPtr, nElements*sizeof(DTYPE), cudaMemcpyDeviceToHost));
	//NR_CUDA_SAFE_CALL(cudaThreadSynchronize());
	return EXIT_SUCCESS;
}
template int cudaCommon_transferFromDeviceToCpu<float>(float *cpuPtr, float **cuPtr, const unsigned int nElements);
template int cudaCommon_transferFromDeviceToCpu<double>(double *cpuPtr, double **cuPtr, const unsigned int nElements);

/* ******************************** */
/* ******************************** */
/* ******************************** */
template <class DTYPE, class NIFTI_TYPE>
int cudaCommon_transferFromDeviceToNifti1(nifti_image *img, DTYPE **array_d)
{
	if(sizeof(DTYPE)!=sizeof(NIFTI_TYPE)){
		reg_print_fct_error("cudaCommon_transferFromDeviceToNifti1");
		reg_print_msg_error("The host and device arrays are of different types");
		return EXIT_FAILURE;
	}
	else
	{
		NIFTI_TYPE *array_h=static_cast<NIFTI_TYPE *>(img->data);
		NR_CUDA_SAFE_CALL(cudaMemcpy((void *)array_h, (void *)*array_d, img->nvox*sizeof(DTYPE), cudaMemcpyDeviceToHost));
	}
	return EXIT_SUCCESS;
}
template int cudaCommon_transferFromDeviceToNifti1<float, float>(nifti_image *img, float **array_d);
template int cudaCommon_transferFromDeviceToNifti1<double, double>(nifti_image *img, double **array_d);
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferFromDeviceToNifti(nifti_image *img, DTYPE **array_d)
{
	if(sizeof(DTYPE)==sizeof(float4)){
		// A nifti 5D volume is expected
		if(img->dim[0]<5 || img->dim[4]>1 || img->dim[5]<2 || img->datatype!=NIFTI_TYPE_FLOAT32){
			reg_print_fct_error("cudaCommon_transferFromDeviceToNifti");
			reg_print_msg_error("The nifti image is not a 5D volume");
			return EXIT_FAILURE;
		}
		const int voxelNumber = img->nx*img->ny*img->nz;

		float4 *array_h;
		NR_CUDA_SAFE_CALL(cudaMallocHost(&array_h, voxelNumber*sizeof(float4)));
		NR_CUDA_SAFE_CALL(cudaMemcpy((void *)array_h, (const void *)*array_d, voxelNumber*sizeof(float4), cudaMemcpyDeviceToHost));
		float *niftiImgValues = static_cast<float *>(img->data);

		for(int i=0; i<voxelNumber; i++)
			*niftiImgValues++ = array_h[i].x;
		if(img->dim[5]>=2){
			for(int i=0; i<voxelNumber; i++)
				*niftiImgValues++ = array_h[i].y;
		}
		if(img->dim[5]>=3){
			for(int i=0; i<voxelNumber; i++)
				*niftiImgValues++ = array_h[i].z;
		}
		if(img->dim[5]>=4){
			for(int i=0; i<voxelNumber; i++)
				*niftiImgValues++ = array_h[i].w;
		}
		NR_CUDA_SAFE_CALL(cudaFreeHost(array_h));

		return EXIT_SUCCESS;
	}
	else{
		switch(img->datatype){
		case NIFTI_TYPE_FLOAT32:
			return cudaCommon_transferFromDeviceToNifti1<DTYPE,float>(img, array_d);
		default:
			reg_print_fct_error("cudaCommon_transferFromDeviceToNifti");
			reg_print_msg_error("The image data type is not supported");
			return EXIT_FAILURE;
		}
	}
}
template int cudaCommon_transferFromDeviceToNifti<float>(nifti_image *, float **);
template int cudaCommon_transferFromDeviceToNifti<double>(nifti_image *, double **);
template int cudaCommon_transferFromDeviceToNifti<float4>(nifti_image *, float4 **); // for deformation field
/* ******************************** */
/* ******************************** */
template <class DTYPE, class NIFTI_TYPE>
int cudaCommon_transferFromDeviceToNifti1(nifti_image *img, DTYPE **array_d, DTYPE **array2_d)
{
	if(sizeof(DTYPE)!=sizeof(NIFTI_TYPE)){
		reg_print_fct_error("cudaCommon_transferFromDeviceToNifti1");
		reg_print_msg_error("The host and device arrays are of different types");
		return EXIT_FAILURE;
	}
	else{
		unsigned int voxelNumber=img->nx*img->ny*img->nz;
		NIFTI_TYPE *array_h=static_cast<NIFTI_TYPE *>(img->data);
		NIFTI_TYPE *array2_h=&array_h[voxelNumber];
		NR_CUDA_SAFE_CALL(cudaMemcpy((void *)array_h, (void *)*array_d, voxelNumber*sizeof(DTYPE), cudaMemcpyDeviceToHost));
		NR_CUDA_SAFE_CALL(cudaMemcpy((void *)array2_h, (void *)*array2_d, voxelNumber*sizeof(DTYPE), cudaMemcpyDeviceToHost));
	}
	return EXIT_SUCCESS;
}
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferFromDeviceToNifti(nifti_image *img, DTYPE **array_d, DTYPE **array2_d)
{
	if(sizeof(DTYPE)==sizeof(float4)){
		// A nifti 5D volume is expected
		if(img->dim[0]<5 || img->dim[4]>1 || img->dim[5]<2 || img->datatype!=NIFTI_TYPE_FLOAT32){
			reg_print_fct_error("cudaCommon_transferFromDeviceToNifti");
			reg_print_msg_error("The nifti image is not a 5D volume");
			return EXIT_FAILURE;
		}
		const int voxelNumber = img->nx*img->ny*img->nz;
		float4 *array_h=NULL;
		float4 *array2_h=NULL;
		NR_CUDA_SAFE_CALL(cudaMallocHost(&array_h, voxelNumber*sizeof(float4)));
		NR_CUDA_SAFE_CALL(cudaMallocHost(&array2_h, voxelNumber*sizeof(float4)));
		NR_CUDA_SAFE_CALL(cudaMemcpy((void *)array_h, (const void *)*array_d, voxelNumber*sizeof(float4), cudaMemcpyDeviceToHost));
		NR_CUDA_SAFE_CALL(cudaMemcpy((void *)array2_h, (const void *)*array2_d, voxelNumber*sizeof(float4), cudaMemcpyDeviceToHost));
		float *niftiImgValues = static_cast<float *>(img->data);
		for(int i=0; i<voxelNumber; i++){
			*niftiImgValues++ = array_h[i].x;
		}
		for(int i=0; i<voxelNumber; i++){
			*niftiImgValues++ = array2_h[i].x;
		}
		if(img->dim[5]>=2){
			for(int i=0; i<voxelNumber; i++){
				*niftiImgValues++ = array_h[i].y;
			}
			for(int i=0; i<voxelNumber; i++){
				*niftiImgValues++ = array2_h[i].y;
			}
		}
		if(img->dim[5]>=3){
			for(int i=0; i<voxelNumber; i++){
				*niftiImgValues++ = array_h[i].z;
			}
			for(int i=0; i<voxelNumber; i++){
				*niftiImgValues++ = array2_h[i].z;
			}
		}
		if(img->dim[5]>=4){
			for(int i=0; i<voxelNumber; i++){
				*niftiImgValues++ = array_h[i].w;
			}
			for(int i=0; i<voxelNumber; i++){
				*niftiImgValues++ = array2_h[i].w;
			}
		}
		NR_CUDA_SAFE_CALL(cudaFreeHost(array_h));
		NR_CUDA_SAFE_CALL(cudaFreeHost(array2_h));

		return EXIT_SUCCESS;
	}
	else{
		switch(img->datatype){
		case NIFTI_TYPE_FLOAT32:
			return cudaCommon_transferFromDeviceToNifti1<DTYPE,float>(img, array_d, array2_d);
		default:
			reg_print_fct_error("cudaCommon_transferFromDeviceToNifti");
			reg_print_msg_error("The image data type is not supported");
			return EXIT_FAILURE;
		}
	}
}
template int cudaCommon_transferFromDeviceToNifti<float>(nifti_image *, float **, float **);
template int cudaCommon_transferFromDeviceToNifti<double>(nifti_image *, double **, double **);
template int cudaCommon_transferFromDeviceToNifti<float4>(nifti_image *, float4 **, float4 **); // for deformation field
/* ******************************** */
/* ******************************** */
void cudaCommon_free(cudaArray **cuArray_d)
{
		NR_CUDA_SAFE_CALL(cudaFreeArray(*cuArray_d));
	return;
}
/* ******************************** */
/* ******************************** */
template <class DTYPE>
void cudaCommon_free(DTYPE **array_d)
{
	NR_CUDA_SAFE_CALL(cudaFree(*array_d));
	return;
}
template void cudaCommon_free<int>(int **);
template void cudaCommon_free<float>(float **);
template void cudaCommon_free<double>(double **);
template void cudaCommon_free<float4>(float4 **);
/* ******************************** */
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferFromDeviceToNiftiSimple(DTYPE **array_d, nifti_image *img)
{
	NR_CUDA_SAFE_CALL(cudaMemcpy(*array_d, img->data, img->nvox * sizeof(DTYPE), cudaMemcpyHostToDevice));

	return EXIT_SUCCESS;
}
template int cudaCommon_transferFromDeviceToNiftiSimple<int>(int **array_d, nifti_image *img);
template int cudaCommon_transferFromDeviceToNiftiSimple<float>(float **array_d, nifti_image *img);
template int cudaCommon_transferFromDeviceToNiftiSimple<double>(double **array_d, nifti_image *img);
/* ******************************** */
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferFromDeviceToNiftiSimple1(DTYPE **array_d, DTYPE *img, const unsigned int nvox)
{
	NR_CUDA_SAFE_CALL(cudaMemcpy(*array_d, img, nvox * sizeof(DTYPE), cudaMemcpyHostToDevice));
	return EXIT_SUCCESS;
}
template int cudaCommon_transferFromDeviceToNiftiSimple1<int>(int **array_d, int *img, const unsigned);
template int cudaCommon_transferFromDeviceToNiftiSimple1<float>(float **array_d, float *img, const unsigned);
template int cudaCommon_transferFromDeviceToNiftiSimple1<double>(double **array_d, double *img, const unsigned);
/* ******************************** */
/* ******************************** */
/* ******************************** */
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferArrayFromCpuToDevice(DTYPE *array_d, const DTYPE *array_cpu, const unsigned int nElements) {

    const unsigned int memSize = nElements * sizeof(DTYPE);
    //copyData
    NR_CUDA_SAFE_CALL(cudaMemcpy(array_d, array_cpu, memSize, cudaMemcpyHostToDevice));
    //
    return EXIT_SUCCESS;
}
template int cudaCommon_transferArrayFromCpuToDevice<int>(int *array_d, const int *array_cpu, const unsigned int nElements);
template int cudaCommon_transferArrayFromCpuToDevice<float>(float *array_d, const float *array_cpu, const unsigned int nElements);
template int cudaCommon_transferArrayFromCpuToDevice<double>(double *array_d, const double *array_cpu, const unsigned int nElements);
/* ******************************** */
/* ******************************** */
/* ******************************** */
/* ******************************** */
template <class DTYPE>
int cudaCommon_transferArrayFromDeviceToCpu(DTYPE *array_cpu, DTYPE *array_d, const unsigned int nElements) {

    const unsigned int memSize = nElements * sizeof(DTYPE);
    //copyData
    NR_CUDA_SAFE_CALL(cudaMemcpy(array_cpu, array_d, memSize, cudaMemcpyDeviceToHost));
    //
    return EXIT_SUCCESS;
}
template int cudaCommon_transferArrayFromDeviceToCpu<int>(int *array_cpu, int *array_d, const unsigned int nElements);
template int cudaCommon_transferArrayFromDeviceToCpu<float>(float *array_cpu, float *array_d, const unsigned int nElements);
template int cudaCommon_transferArrayFromDeviceToCpu<double>(double *array_cpu, double *array_d, const unsigned int nElements);
#endif
/* ******************************** */
/* ******************************** */
