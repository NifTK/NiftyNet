/*
 *  _reg_resampling.cpp
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_RESAMPLING_CPP
#define _REG_RESAMPLING_CPP

#include "_reg_resampling.h"
#include "_reg_maths.h"
#include "_reg_maths_eigen.h"
#include "_reg_tools.h"
#include "interpolations.h"

#include <cassert>

#define SINC_KERNEL_RADIUS 3
#define SINC_KERNEL_SIZE SINC_KERNEL_RADIUS*2

/* *************************************************************** */
void interpWindowedSincKernel(double relative, double *basis)
{
    if(relative<0.0) relative=0.0; //reg_rounding error
    int j=0;
    double sum=0.;
    for(int i=-SINC_KERNEL_RADIUS; i<SINC_KERNEL_RADIUS; ++i)
    {
        double x=relative-static_cast<double>(i);
        if(x==0.0)
            basis[j]=1.0;
        else if(fabs(x)>=static_cast<double>(SINC_KERNEL_RADIUS))
            basis[j]=0;
        else{
            double pi_x=M_PI*x;
            basis[j]=static_cast<double>(SINC_KERNEL_RADIUS) *
                    sin(pi_x) *
                    sin(pi_x/static_cast<double>(SINC_KERNEL_RADIUS)) /
                    (pi_x*pi_x);
        }
        sum+=basis[j];
        j++;
    }
    for(int i=0;i<SINC_KERNEL_SIZE;++i)
        basis[i]/=sum;
}
/* *************************************************************** */
/* *************************************************************** */
double interpWindowedSincKernel_Samp(double x, double kernelsize)
{
    if(x==0.0)
        return 1.0;
    else if(fabs(x)>=static_cast<double>(kernelsize))
        return 0;
    else{
        double pi_x=M_PI*fabs(x);
        return static_cast<double>(kernelsize) *
                sin(pi_x) *
                sin(pi_x/static_cast<double>(kernelsize)) /
                (pi_x*pi_x);
    }
}
/* *************************************************************** */
/* *************************************************************** */
void interpLinearKernel(double relative, double *basis)
{
    if(relative<0.0) relative=0.0; //reg_rounding error
    basis[1]=relative;
    basis[0]=1.0-relative;
}
/* *************************************************************** */
void interpLinearKernel(double relative, double *basis, double *derivative)
{
    interpLinearKernel(relative,basis);
    derivative[1]=1.0;
    derivative[0]=0.0;
}
/* *************************************************************** */
/* *************************************************************** */
void interpNearestNeighKernel(double relative, double *basis)
{
    if(relative<0.0) relative=0.0; //reg_rounding error
    basis[0]=basis[1]=0;
    if(relative>=0.5)
        basis[1]=1;
    else basis[0]=1;
}
/* *************************************************************** */
template<const resampler_boundary_e tBoundary, class FloatingTYPE, class FieldTYPE>
void ResampleImage3D(nifti_image *floatingImage,
                     nifti_image *deformationField,
                     nifti_image *warpedImage,
                     FieldTYPE paddingValue,
                     int kernel)
{
#ifdef _WIN32
    long  index;
    long warpedVoxelNumber = (long)warpedImage->nx*warpedImage->ny*warpedImage->nz;
    long floatingVoxelNumber = (long)floatingImage->nx*floatingImage->ny*floatingImage->nz;
#else
    size_t  index;
    size_t warpedVoxelNumber = (size_t)warpedImage->nx*warpedImage->ny*warpedImage->nz;
    size_t floatingVoxelNumber = (size_t)floatingImage->nx*floatingImage->ny*floatingImage->nz;
#endif
    FloatingTYPE *floatingIntensityPtr = static_cast<FloatingTYPE *>(floatingImage->data);
    FloatingTYPE *warpedIntensityPtr = static_cast<FloatingTYPE *>(warpedImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[warpedVoxelNumber];
    FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[warpedVoxelNumber];

    // Define the kernel to use
    int kernel_size;
    int kernel_offset=0;
    void (*kernelCompFctPtr)(double,double *);
    switch(kernel){
    case 0:
        kernel_size=2;
        kernelCompFctPtr=&interpNearestNeighKernel;
        kernel_offset=0;
        break; // nereast-neighboor interpolation
    case 1:
        kernel_size=2;
        kernelCompFctPtr=&interpLinearKernel;
        kernel_offset=0;
        break; // linear interpolation
    case 4:
        kernel_size=SINC_KERNEL_SIZE;
        kernelCompFctPtr=&interpWindowedSincKernel;
        kernel_offset=SINC_KERNEL_RADIUS;
        break; // sinc interpolation
    default:
        kernel_size=4;
        kernelCompFctPtr=&reg_getNiftynetCubicSpline<double, double>;
        kernel_offset=1;
        break; // cubic spline interpolation
    }

    // Iteration over the different volume along the 4th axis
    for(size_t t=0; t<(size_t)warpedImage->nt*warpedImage->nu; t++)
    {
#ifndef NDEBUG
        char text[255];
        sprintf(text, "3D resampling of volume number %zu",t);
        reg_print_msg_debug(text);
#endif

        FloatingTYPE *warpedIntensity = &warpedIntensityPtr[t*warpedVoxelNumber];
        FloatingTYPE *floatingIntensity = &floatingIntensityPtr[t*floatingVoxelNumber];

        int a, b, c, Y, Z, previous[3];

        FloatingTYPE *zPointer, *xyzPointer;
        double xBasis[SINC_KERNEL_SIZE], yBasis[SINC_KERNEL_SIZE], zBasis[SINC_KERNEL_SIZE], relative[3];
        double xTempNewValue, yTempNewValue, intensity;
        float position[3];
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
    private(index, intensity, position, previous, xBasis, yBasis, zBasis, relative, \
    a, b, c, Y, Z, zPointer, xyzPointer, xTempNewValue, yTempNewValue) \
    shared(floatingIntensity, warpedIntensity, warpedVoxelNumber, floatingVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, \
    floatingImage, paddingValue, kernel_size, kernel_offset, kernelCompFctPtr)
#endif // _OPENMP
        for(index=0; index<warpedVoxelNumber; index++)
        {

            intensity=paddingValue;

            position[0]=static_cast<float>(deformationFieldPtrX[index]);
            position[1]=static_cast<float>(deformationFieldPtrY[index]);
            position[2]=static_cast<float>(deformationFieldPtrZ[index]);

            previous[0] = static_cast<int>(reg_floor(position[0]));
            previous[1] = static_cast<int>(reg_floor(position[1]));
            previous[2] = static_cast<int>(reg_floor(position[2]));

            relative[0]=static_cast<double>(position[0])-static_cast<double>(previous[0]);
            relative[1]=static_cast<double>(position[1])-static_cast<double>(previous[1]);
            relative[2]=static_cast<double>(position[2])-static_cast<double>(previous[2]);

            (*kernelCompFctPtr)(relative[0], xBasis);
            (*kernelCompFctPtr)(relative[1], yBasis);
            (*kernelCompFctPtr)(relative[2], zBasis);
            previous[0]-=kernel_offset;
            previous[1]-=kernel_offset;
            previous[2]-=kernel_offset;

            intensity=0.0;
            if(-1<(previous[0]) && (previous[0]+kernel_size-1)<floatingImage->nx &&
               -1<(previous[1]) && (previous[1]+kernel_size-1)<floatingImage->ny &&
               -1<(previous[2]) && (previous[2]+kernel_size-1)<floatingImage->nz){
              for(c=0; c<kernel_size; c++)
              {
                Z= previous[2]+c;
                zPointer = &floatingIntensity[Z*floatingImage->nx*floatingImage->ny];
                yTempNewValue=0.0;
                for(b=0; b<kernel_size; b++)
                {
                  Y= previous[1]+b;
                  xyzPointer = &zPointer[Y*floatingImage->nx+previous[0]];
                  xTempNewValue=0.0;
                  for(a=0; a<kernel_size; a++)
                  {
                    xTempNewValue +=  static_cast<double>(*xyzPointer++) * xBasis[a];
                  }
                  yTempNewValue += xTempNewValue * yBasis[b];
                }
                intensity += yTempNewValue * zBasis[c];
              }
            }
            else{
              for(c=0; c<kernel_size; c++)
              {
                Z= reg_applyBoundary<tBoundary>(previous[2] + c, floatingImage->nz);
                zPointer = &floatingIntensity[Z*floatingImage->nx*floatingImage->ny];
                yTempNewValue=0.0;

                for(b=0; b<kernel_size; b++)
                {
                  Y= reg_applyBoundary<tBoundary>(previous[1] + b, floatingImage->ny);
                  xyzPointer = &zPointer[Y*floatingImage->nx];
                  xTempNewValue=0.0;
                  for(a=0; a<kernel_size; a++)
                  {
                    int X = reg_applyBoundary<tBoundary>(previous[0] + a, floatingImage->nx);

                    if(reg_checkImageDimensionIndex<tBoundary>(X, floatingImage->nx)
                       && reg_checkImageDimensionIndex<tBoundary>(Y, floatingImage->ny)
                       && reg_checkImageDimensionIndex<tBoundary>(Z, floatingImage->nz)) {
                      xTempNewValue +=  static_cast<double>(xyzPointer[X]) * xBasis[a];
                    }
                    else
                    {
                      // paddingValue
                      xTempNewValue +=  static_cast<double>(paddingValue) * xBasis[a];
                    }
                  }
                  yTempNewValue += xTempNewValue * yBasis[b];
                }
                intensity += yTempNewValue * zBasis[c];
              }
            }

            switch(floatingImage->datatype)
            {
            case NIFTI_TYPE_FLOAT32:
                warpedIntensity[index]=static_cast<FloatingTYPE>(intensity);
                break;
            case NIFTI_TYPE_FLOAT64:
                warpedIntensity[index]=intensity;
                break;
            case NIFTI_TYPE_UINT8:
                if(intensity!=intensity)
                    intensity=0;
                intensity=(intensity<=255?reg_round(intensity):255); // 255=2^8-1
                warpedIntensity[index]=static_cast<FloatingTYPE>(intensity>0?reg_round(intensity):0);
                break;
            case NIFTI_TYPE_UINT16:
                if(intensity!=intensity)
                    intensity=0;
                intensity=(intensity<=65535?reg_round(intensity):65535); // 65535=2^16-1
                warpedIntensity[index]=static_cast<FloatingTYPE>(intensity>0?reg_round(intensity):0);
                break;
            case NIFTI_TYPE_UINT32:
                if(intensity!=intensity)
                    intensity=0;
                intensity=(intensity<=4294967295?reg_round(intensity):4294967295); // 4294967295=2^32-1
                warpedIntensity[index]=static_cast<FloatingTYPE>(intensity>0?reg_round(intensity):0);
                break;
            default:
                if(intensity!=intensity)
                    intensity=0;
                warpedIntensity[index]=static_cast<FloatingTYPE>(reg_round(intensity));
                break;
            }
        }
    }
}
/* *************************************************************** */
template<const resampler_boundary_e tBoundary, class FloatingTYPE, class FieldTYPE>
void ResampleImage2D(nifti_image *floatingImage,
                     nifti_image *deformationField,
                     nifti_image *warpedImage,
                     FieldTYPE paddingValue,
                     int kernel)
{
#ifdef _WIN32
    long  index;
    long warpedVoxelNumber = (long)warpedImage->nx*warpedImage->ny;
    long floatingVoxelNumber = (long)floatingImage->nx*floatingImage->ny;
#else
    size_t  index;
    size_t warpedVoxelNumber = (size_t)warpedImage->nx*warpedImage->ny;
    size_t floatingVoxelNumber = (size_t)floatingImage->nx*floatingImage->ny;
#endif
    FloatingTYPE *floatingIntensityPtr = static_cast<FloatingTYPE *>(floatingImage->data);
    FloatingTYPE *warpedIntensityPtr = static_cast<FloatingTYPE *>(warpedImage->data);
    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[warpedVoxelNumber];

    int kernel_size;
    int kernel_offset=0;
    void (*kernelCompFctPtr)(double,double *);
    switch(kernel){
    case 0:
        kernel_size=2;
        kernelCompFctPtr=&interpNearestNeighKernel;
        kernel_offset=0;
        break; // nereast-neighboor interpolation
    case 1:
        kernel_size=2;
        kernelCompFctPtr=&interpLinearKernel;
        kernel_offset=0;
        break; // linear interpolation
    case 4:
        kernel_size=SINC_KERNEL_SIZE;
        kernelCompFctPtr=&interpWindowedSincKernel;
        kernel_offset=SINC_KERNEL_RADIUS;
        break; // sinc interpolation
    default:
        kernel_size=4;
        kernelCompFctPtr=&reg_getNiftynetCubicSpline<double, double>;
        kernel_offset=1;
        break; // cubic spline interpolation
    }

    // Iteration over the different volume along the 4th axis
    for(size_t t=0; t<(size_t)warpedImage->nt*warpedImage->nu; t++)
    {
#ifndef NDEBUG
        char text[255];
        sprintf(text, "2D resampling of volume number %zu",t);
        reg_print_msg_debug(text);
#endif
        FloatingTYPE *warpedIntensity = &warpedIntensityPtr[t*warpedVoxelNumber];
        FloatingTYPE *floatingIntensity = &floatingIntensityPtr[t*floatingVoxelNumber];

        int a, b, Y, previous[2];

        FloatingTYPE *xyzPointer;
        double xBasis[SINC_KERNEL_SIZE], yBasis[SINC_KERNEL_SIZE], relative[2];
        double xTempNewValue, intensity;
        float position[3] = {0.0, 0.0, 0.0};
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
    private(index, intensity, position, previous, xBasis, yBasis, relative, \
    a, b, Y, xyzPointer, xTempNewValue) \
    shared(floatingIntensity, warpedIntensity, warpedVoxelNumber, floatingVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, \
    floatingImage, paddingValue, kernel_size, kernel_offset, kernelCompFctPtr)
#endif // _OPENMP
        for(index=0; index<warpedVoxelNumber; index++)
        {

            position[0] = static_cast<float>(deformationFieldPtrX[index]);
            position[1] = static_cast<float>(deformationFieldPtrY[index]);

            previous[0] = static_cast<int>(reg_floor(position[0]));
            previous[1] = static_cast<int>(reg_floor(position[1]));

            relative[0] = static_cast<double>(position[0])-static_cast<double>(previous[0]);
            relative[1] = static_cast<double>(position[1])-static_cast<double>(previous[1]);

            (*kernelCompFctPtr)(relative[0], xBasis);
            (*kernelCompFctPtr)(relative[1], yBasis);
            previous[0]-=kernel_offset;
            previous[1]-=kernel_offset;

            intensity=0.0;
            for(b=0; b<kernel_size; b++)
            {
              Y= reg_applyBoundary<tBoundary>(previous[1] + b, floatingImage->ny);
              xyzPointer = &floatingIntensity[Y*floatingImage->nx];
              xTempNewValue=0.0;
              for(a=0; a<kernel_size; a++)
              {
                int X = reg_applyBoundary<tBoundary>(previous[0] + a, floatingImage->nx);

                if(reg_checkImageDimensionIndex<tBoundary>(X, floatingImage->nx)
                   && reg_checkImageDimensionIndex<tBoundary>(Y, floatingImage->ny)) {
                  xTempNewValue +=  static_cast<double>(xyzPointer[X]) * xBasis[a];
                }
                else
                {
                  // paddingValue
                  xTempNewValue +=  static_cast<double>(paddingValue) * xBasis[a];
                }
              }
              intensity += xTempNewValue * yBasis[b];

                switch(floatingImage->datatype)
                {
                case NIFTI_TYPE_FLOAT32:
                    warpedIntensity[index]=static_cast<FloatingTYPE>(intensity);
                    break;
                case NIFTI_TYPE_FLOAT64:
                    warpedIntensity[index]=intensity;
                    break;
                case NIFTI_TYPE_UINT8:
                    intensity=(intensity<=255?reg_round(intensity):255); // 255=2^8-1
                    warpedIntensity[index]=static_cast<FloatingTYPE>(intensity>0?reg_round(intensity):0);
                    break;
                case NIFTI_TYPE_UINT16:
                    intensity=(intensity<=65535?reg_round(intensity):65535); // 65535=2^16-1
                    warpedIntensity[index]=static_cast<FloatingTYPE>(intensity>0?reg_round(intensity):0);
                    break;
                case NIFTI_TYPE_UINT32:
                    intensity=(intensity<=4294967295?reg_round(intensity):4294967295); // 4294967295=2^32-1
                    warpedIntensity[index]=static_cast<FloatingTYPE>(intensity>0?reg_round(intensity):0);
                    break;
                default:
                    warpedIntensity[index]=static_cast<FloatingTYPE>(reg_round(intensity));
                    break;
                }
            }
        }
    }
}
/* *************************************************************** */
/* *************************************************************** */

/** This function resample a floating image into the referential
 * of a reference image by applying an affine transformation and
 * a deformation field. The affine transformation has to be in
 * real coordinate and the deformation field is in mm in the space
 * of the reference image.
 * interp can be either 0, 1 or 3 meaning nearest neighbor, linear
 * or cubic spline interpolation.
 * every voxel which is not fully in the floating image takes the
 * backgreg_round value.
 */
template <class FieldTYPE, class FloatingTYPE>
void reg_resampleImage2(nifti_image *floatingImage,
                        nifti_image *warpedImage,
                        nifti_image *deformationFieldImage,
                        int interp,
                        resampler_boundary_e boundaryTreatment)
{
    const FieldTYPE paddingValue = reg_getPaddingValue<FieldTYPE>(boundaryTreatment);

    // The deformation field contains the position in the real world
    if(deformationFieldImage->nz>1 || floatingImage->nz>1)
    {
      if (boundaryTreatment == resampler_boundary_e::ZEROPAD || boundaryTreatment == resampler_boundary_e::NANPAD) {
        ResampleImage3D<resampler_boundary_e::ZEROPAD,FloatingTYPE,FieldTYPE>(floatingImage,
                                                                              deformationFieldImage,
                                                                              warpedImage,
                                                                              paddingValue,
                                                                              interp);
      } else if (boundaryTreatment == resampler_boundary_e::CLAMPING) {
        ResampleImage3D<resampler_boundary_e::CLAMPING,FloatingTYPE,FieldTYPE>(floatingImage,
                                                                               deformationFieldImage,
                                                                               warpedImage,
                                                                               paddingValue,
                                                                               interp);
      } else if (boundaryTreatment == resampler_boundary_e::REFLECTING) {
        ResampleImage3D<resampler_boundary_e::REFLECTING,FloatingTYPE,FieldTYPE>(floatingImage,
                                                                                 deformationFieldImage,
                                                                                 warpedImage,
                                                                                 paddingValue,
                                                                                 interp);
      }
    }
    else
    {
      if (boundaryTreatment == resampler_boundary_e::ZEROPAD || boundaryTreatment == resampler_boundary_e::NANPAD) {
        ResampleImage2D<resampler_boundary_e::ZEROPAD,FloatingTYPE,FieldTYPE>(floatingImage,
                                                                              deformationFieldImage,
                                                                              warpedImage,
                                                                              paddingValue,
                                                                              interp);
      } else if (boundaryTreatment == resampler_boundary_e::CLAMPING) {
        ResampleImage2D<resampler_boundary_e::CLAMPING,FloatingTYPE,FieldTYPE>(floatingImage,
                                                                               deformationFieldImage,
                                                                               warpedImage,
                                                                               paddingValue,
                                                                               interp);
      } else if (boundaryTreatment == resampler_boundary_e::REFLECTING) {
        ResampleImage2D<resampler_boundary_e::REFLECTING,FloatingTYPE,FieldTYPE>(floatingImage,
                                                                                 deformationFieldImage,
                                                                                 warpedImage,
                                                                                 paddingValue,
                                                                                 interp);
      }
    }
}
/* *************************************************************** */
void reg_resampleImage(nifti_image *floatingImage,
                       nifti_image *warpedImage,
                       nifti_image *deformationField,
                       int interp,
                       resampler_boundary_e boundaryTreatment)
{
    if(floatingImage->datatype != warpedImage->datatype)
    {
        reg_print_fct_error("reg_resampleImage");
        reg_print_msg_error("The floating and warped image should have the same data type");
        reg_exit();
    }

    if(floatingImage->nt != warpedImage->nt)
    {
        reg_print_fct_error("reg_resampleImage");
        reg_print_msg_error("The floating and warped images have different dimension along the time axis");
        reg_exit();
    }

    switch ( deformationField->datatype )
    {
    case NIFTI_TYPE_FLOAT32:
        switch ( floatingImage->datatype )
        {
        case NIFTI_TYPE_FLOAT32:
            reg_resampleImage2<float,float>(floatingImage,
                                            warpedImage,
                                            deformationField,
                                            interp,
                                            boundaryTreatment);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_resampleImage2<float,double>(floatingImage,
                                             warpedImage,
                                             deformationField,
                                             interp,
                                             boundaryTreatment);
            break;
        default:
            printf("floating pixel type unsupported.");
            break;
        }
        break;
    case NIFTI_TYPE_FLOAT64:
        switch ( floatingImage->datatype )
        {
        case NIFTI_TYPE_FLOAT32:
            reg_resampleImage2<double,float>(floatingImage,
                                             warpedImage,
                                             deformationField,
                                             interp,
                                             boundaryTreatment);
            break;
        case NIFTI_TYPE_FLOAT64:
            reg_resampleImage2<double,double>(floatingImage,
                                              warpedImage,
                                              deformationField,
                                              interp,
                                              boundaryTreatment);
            break;
        default:
            printf("floating pixel type unsupported.");
            break;
        }
        break;
    default:
        printf("Deformation field pixel type unsupported.");
        break;
    }
}
/* *************************************************************** */
/* *************************************************************** */
template<const resampler_boundary_e tBoundary, class FloatingTYPE, class GradientTYPE, class FieldTYPE>
void TrilinearImageGradient(nifti_image *floatingImage,
                            nifti_image *deformationField,
                            nifti_image *warImgGradient,
                            float paddingValue,
                            int active_timepoint)
{
  if(active_timepoint<0 || active_timepoint>=(std::max)(floatingImage->nt,floatingImage->nu)){
        reg_print_fct_error("TrilinearImageGradient");
        reg_print_msg_error("The specified active timepoint is not defined in the floating image");
        reg_exit();
    }
#ifdef _WIN32
    long index;
    long referenceVoxelNumber = (long)warImgGradient->nx*warImgGradient->ny*warImgGradient->nz;
    long floatingVoxelNumber = (long)floatingImage->nx*floatingImage->ny*floatingImage->nz;
#else
    size_t index;
    size_t referenceVoxelNumber = (size_t)warImgGradient->nx*warImgGradient->ny*warImgGradient->nz;
    size_t floatingVoxelNumber = (size_t)floatingImage->nx*floatingImage->ny*floatingImage->nz;
#endif
    FloatingTYPE *floatingIntensityPtr = static_cast<FloatingTYPE *>(floatingImage->data);
    FloatingTYPE *floatingIntensity = &floatingIntensityPtr[active_timepoint*floatingVoxelNumber];

    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[referenceVoxelNumber];
    FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[referenceVoxelNumber];

    GradientTYPE *warpedGradientPtrX = static_cast<GradientTYPE *>(warImgGradient->data);
    GradientTYPE *warpedGradientPtrY = &warpedGradientPtrX[referenceVoxelNumber];
    GradientTYPE *warpedGradientPtrZ = &warpedGradientPtrY[referenceVoxelNumber];

#ifndef NDEBUG
    char text[255];
    sprintf(text, "3D linear gradient computation of volume number %i", active_timepoint);
    reg_print_msg_debug(text);
#endif

    int previous[3], a, b, c, X, Y, Z;
    FieldTYPE position[3], xBasis[2], yBasis[2], zBasis[2];
    FieldTYPE deriv[2];
    deriv[0]=-1;
    deriv[1]=1;
    FieldTYPE relative, grad[3], coeff;
    FieldTYPE xxTempNewValue, yyTempNewValue, zzTempNewValue, xTempNewValue, yTempNewValue;
    FloatingTYPE *zPointer, *xyzPointer;
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
    private(index, position, previous, xBasis, yBasis, zBasis, relative, grad, coeff, \
    a, b, c, X, Y, Z, zPointer, xyzPointer, xTempNewValue, yTempNewValue, xxTempNewValue, yyTempNewValue, zzTempNewValue) \
    shared(floatingIntensity, referenceVoxelNumber, floatingVoxelNumber, deriv, paddingValue, \
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, \
    floatingImage, warpedGradientPtrX, warpedGradientPtrY, warpedGradientPtrZ)
#endif // _OPENMP
    for(index=0; index<referenceVoxelNumber; index++)
    {

        grad[0]=0.0;
        grad[1]=0.0;
        grad[2]=0.0;

        position[0]=(FieldTYPE) deformationFieldPtrX[index];
        position[1]=(FieldTYPE) deformationFieldPtrY[index];
        position[2]=(FieldTYPE) deformationFieldPtrZ[index];

        previous[0] = static_cast<int>(reg_floor(position[0]));
        previous[1] = static_cast<int>(reg_floor(position[1]));
        previous[2] = static_cast<int>(reg_floor(position[2]));
        // basis values along the x axis
        relative=position[0]-(FieldTYPE)previous[0];
        xBasis[0]= (FieldTYPE)(1.0-relative);
        xBasis[1]= relative;
        // basis values along the y axis
        relative=position[1]-(FieldTYPE)previous[1];
        yBasis[0]= (FieldTYPE)(1.0-relative);
        yBasis[1]= relative;
        // basis values along the z axis
        relative=position[2]-(FieldTYPE)previous[2];
        zBasis[0]= (FieldTYPE)(1.0-relative);
        zBasis[1]= relative;

        // The padding value is used for interpolation if it is different from NaN
        if(tBoundary == resampler_boundary_e::ZEROPAD && paddingValue==paddingValue)
        {
          for(c=0; c<2; c++)
          {
            Z=previous[2]+c;
            if(Z>-1 && Z<floatingImage->nz)
            {
              zPointer = &floatingIntensity[Z*floatingImage->nx*floatingImage->ny];
              xxTempNewValue=0.0;
              yyTempNewValue=0.0;
              zzTempNewValue=0.0;
              for(b=0; b<2; b++)
              {
                Y=previous[1]+b;
                if(Y>-1 && Y<floatingImage->ny)
                {
                  xyzPointer = &zPointer[Y*floatingImage->nx+previous[0]];
                  xTempNewValue=0.0;
                  yTempNewValue=0.0;
                  for(a=0; a<2; a++)
                  {
                    X=previous[0]+a;
                    if(X>-1 && X<floatingImage->nx)
                    {
                      coeff = *xyzPointer;
                      xTempNewValue +=  coeff * deriv[a];
                      yTempNewValue +=  coeff * xBasis[a];
                    } // end X in range
                    else
                    {
                      xTempNewValue +=  paddingValue * deriv[a];
                      yTempNewValue +=  paddingValue * xBasis[a];
                    }
                    xyzPointer++;
                  } // end a
                  xxTempNewValue += xTempNewValue * yBasis[b];
                  yyTempNewValue += yTempNewValue * deriv[b];
                  zzTempNewValue += yTempNewValue * yBasis[b];
                } // end Y in range
                else
                {
                  xxTempNewValue += paddingValue * yBasis[b];
                  yyTempNewValue += paddingValue * deriv[b];
                  zzTempNewValue += paddingValue * yBasis[b];
                }
              } // end b
              grad[0] += xxTempNewValue * zBasis[c];
              grad[1] += yyTempNewValue * zBasis[c];
              grad[2] += zzTempNewValue * deriv[c];
            } // end Z in range
            else
            {
              grad[0] += paddingValue * zBasis[c];
              grad[1] += paddingValue * zBasis[c];
              grad[2] += paddingValue * deriv[c];
            }
          } // end c
        } // end padding value is different from NaN
        else if(reg_checkImageDimensionIndex<tBoundary>(previous[0],floatingImage->nx - 1)
                && reg_checkImageDimensionIndex<tBoundary>(previous[1],floatingImage->ny - 1)
                && reg_checkImageDimensionIndex<tBoundary>(previous[2],floatingImage->nz - 1)) {
          for(c=0; c<2; c++)
          {
            Z = reg_applyBoundary<tBoundary>(previous[2] + c, floatingImage->nz);
            zPointer = &floatingIntensity[Z*floatingImage->nx*floatingImage->ny];
            xxTempNewValue=0.0;
            yyTempNewValue=0.0;
            zzTempNewValue=0.0;
            for(b=0; b<2; b++)
            {
              Y = reg_applyBoundary<tBoundary>(previous[1] + b, floatingImage->ny);
              xyzPointer = &zPointer[Y*floatingImage->nx];
              xTempNewValue=0.0;
              yTempNewValue=0.0;
              for(a=0; a<2; a++)
              {
                X = reg_applyBoundary<tBoundary>(previous[0] + a, floatingImage->nx);
                coeff = xyzPointer[X];
                xTempNewValue +=  coeff * deriv[a];
                yTempNewValue +=  coeff * xBasis[a];
              } // end a
              xxTempNewValue += xTempNewValue * yBasis[b];
              yyTempNewValue += yTempNewValue * deriv[b];
              zzTempNewValue += yTempNewValue * yBasis[b];
            } // end b
            grad[0] += xxTempNewValue * zBasis[c];
            grad[1] += yyTempNewValue * zBasis[c];
            grad[2] += zzTempNewValue * deriv[c];
          } // end c
        } // end padding value is NaN
        else grad[0]=grad[1]=grad[2]=0;

        warpedGradientPtrX[index] = (GradientTYPE)grad[0];
        warpedGradientPtrY[index] = (GradientTYPE)grad[1];
        warpedGradientPtrZ[index] = (GradientTYPE)grad[2];
    }
}
/* *************************************************************** */
template<const resampler_boundary_e tBoundary, class FloatingTYPE, class GradientTYPE, class FieldTYPE>
void BilinearImageGradient(nifti_image *floatingImage,
                           nifti_image *deformationField,
                           nifti_image *warImgGradient,
                           float paddingValue,
                           int active_timepoint)
{
   if(active_timepoint<0 || active_timepoint>=(std::max)(floatingImage->nt, floatingImage->nu)){
        reg_print_fct_error("BilinearImageGradient");
        reg_print_msg_error("The specified active timepoint is not defined in the floating image");
        reg_exit();
    }
#ifdef _WIN32
    long index;
    long referenceVoxelNumber = (long)warImgGradient->nx*warImgGradient->ny;
    long floatingVoxelNumber = (long)floatingImage->nx*floatingImage->ny;
#else
    size_t index;
    size_t referenceVoxelNumber = (size_t)warImgGradient->nx*warImgGradient->ny;
    size_t floatingVoxelNumber = (size_t)floatingImage->nx*floatingImage->ny;
#endif

    FloatingTYPE *floatingIntensityPtr = static_cast<FloatingTYPE *>(floatingImage->data);
    FloatingTYPE *floatingIntensity = &floatingIntensityPtr[active_timepoint*floatingVoxelNumber];

    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[referenceVoxelNumber];

    GradientTYPE *warpedGradientPtrX = static_cast<GradientTYPE *>(warImgGradient->data);
    GradientTYPE *warpedGradientPtrY = &warpedGradientPtrX[referenceVoxelNumber];

#ifndef NDEBUG
    char text[255];
    sprintf(text, "2D linear gradient computation of volume number %i",active_timepoint);
    reg_print_msg_debug(text);
#endif

    FieldTYPE position[3], xBasis[2], yBasis[2], relative, grad[2];
    FieldTYPE deriv[2];
    deriv[0]=-1;
    deriv[1]=1;
    FieldTYPE coeff, xTempNewValue, yTempNewValue;

    int previous[3], a, b, X, Y;
    FloatingTYPE *xyPointer;

    assert(deformationField->nu == 2);

#if defined (_OPENMP)
#pragma omp parallel for default(none) \
    private(index, position, previous, xBasis, yBasis, relative, grad, coeff, \
    a, b, X, Y, xyPointer, xTempNewValue, yTempNewValue) \
    shared(floatingIntensity, referenceVoxelNumber, floatingVoxelNumber, deriv, \
    deformationFieldPtrX, deformationFieldPtrY, paddingValue, \
    floatingImage, warpedGradientPtrX, warpedGradientPtrY)
#endif // _OPENMP
    for(index=0; index<referenceVoxelNumber; index++)
    {

        grad[0]=0.0;
        grad[1]=0.0;

        position[0]=(FieldTYPE) deformationFieldPtrX[index];
        position[1]=(FieldTYPE) deformationFieldPtrY[index];

        previous[0] = static_cast<int>(reg_floor(position[0]));
        previous[1] = static_cast<int>(reg_floor(position[1]));
        // basis values along the x axis
        relative=position[0]-(FieldTYPE)previous[0];
        relative=relative>0?relative:0;
        xBasis[0]= (FieldTYPE)(1.0-relative);
        xBasis[1]= relative;
        // basis values along the y axis
        relative=position[1]-(FieldTYPE)previous[1];
        relative=relative>0?relative:0;
        yBasis[0]= (FieldTYPE)(1.0-relative);
        yBasis[1]= relative;

        for(b=0; b<2; b++)
        {
          Y= reg_applyBoundary<tBoundary>(previous[1] + b, floatingImage->ny);
          if (reg_checkImageDimensionIndex<tBoundary>(Y, floatingImage->ny)) {
            xyPointer = &floatingIntensity[Y*floatingImage->nx];
            xTempNewValue=0.0;
            yTempNewValue=0.0;
            for(a=0; a<2; a++)
            {
              X = reg_applyBoundary<tBoundary>(previous[0] + a, floatingImage->nx);
              if (reg_checkImageDimensionIndex<tBoundary>(X, floatingImage->nx)) {
                coeff = xyPointer[X];
                xTempNewValue +=  coeff * deriv[a];
                yTempNewValue +=  coeff * xBasis[a];
              }
              else
              {
                xTempNewValue +=  paddingValue * deriv[a];
                yTempNewValue +=  paddingValue * xBasis[a];
              }
            }
            grad[0] += xTempNewValue * yBasis[b];
            grad[1] += yTempNewValue * deriv[b];
          }
          else
          {
            grad[0] += paddingValue * yBasis[b];
            grad[1] += paddingValue * deriv[b];
          }
        }
        if(grad[0]!=grad[0]) grad[0]=0;
        if(grad[1]!=grad[1]) grad[1]=0;

        warpedGradientPtrX[index] = (GradientTYPE)grad[0];
        warpedGradientPtrY[index] = (GradientTYPE)grad[1];
    }
}
/* *************************************************************** */
template<const resampler_boundary_e tBoundary, class FloatingTYPE, class GradientTYPE, class FieldTYPE>
void CubicSplineImageGradient3D(nifti_image *floatingImage,
                                nifti_image *deformationField,
                                nifti_image *warImgGradient,
                                float paddingValue,
                                int active_timepoint)
{
   if(active_timepoint<0 || active_timepoint>=(std::max)(floatingImage->nt, floatingImage->nu)){
        reg_print_fct_error("CubicSplineImageGradient3D");
        reg_print_msg_error("The specified active timepoint is not defined in the floating image");
        reg_exit();
    }
#ifdef _WIN32
    long index;
    long referenceVoxelNumber = (long)warImgGradient->nx*warImgGradient->ny*warImgGradient->nz;
    long floatingVoxelNumber = (long)floatingImage->nx*floatingImage->ny*floatingImage->nz;
#else
    size_t index;
    size_t referenceVoxelNumber = (size_t)warImgGradient->nx*warImgGradient->ny*warImgGradient->nz;
    size_t floatingVoxelNumber = (size_t)floatingImage->nx*floatingImage->ny*floatingImage->nz;
#endif
    FloatingTYPE *floatingIntensityPtr = static_cast<FloatingTYPE *>(floatingImage->data);
    FloatingTYPE *floatingIntensity = &floatingIntensityPtr[active_timepoint*floatingVoxelNumber];

    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[referenceVoxelNumber];
    FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[referenceVoxelNumber];

    GradientTYPE *warpedGradientPtrX = static_cast<GradientTYPE *>(warImgGradient->data);
    GradientTYPE *warpedGradientPtrY = &warpedGradientPtrX[referenceVoxelNumber];
    GradientTYPE *warpedGradientPtrZ = &warpedGradientPtrY[referenceVoxelNumber];

#ifndef NDEBUG
    char text[255];
    sprintf(text, "3D cubic spline gradient computation of volume number %i",active_timepoint);
    reg_print_msg_debug(text);
#endif

    int previous[3], c, Z, b, Y, a;

    double xBasis[4], yBasis[4], zBasis[4], xDeriv[4], yDeriv[4], zDeriv[4], relative;
    FieldTYPE coeff, position[3], grad[3];
    FieldTYPE xxTempNewValue, yyTempNewValue, zzTempNewValue, xTempNewValue, yTempNewValue;
    FloatingTYPE *zPointer, *yzPointer;
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
    private(index, position, previous, xBasis, yBasis, zBasis, xDeriv, yDeriv, zDeriv, relative, grad, coeff, \
    a, b, c, Y, Z, zPointer, yzPointer, xTempNewValue, yTempNewValue, xxTempNewValue, yyTempNewValue, zzTempNewValue) \
    shared(floatingIntensity, referenceVoxelNumber, floatingVoxelNumber, paddingValue, \
    deformationFieldPtrX, deformationFieldPtrY, deformationFieldPtrZ, \
    floatingImage, warpedGradientPtrX, warpedGradientPtrY, warpedGradientPtrZ)
#endif // _OPENMP
    for(index=0; index<referenceVoxelNumber; index++)
    {

        grad[0]=0.0;
        grad[1]=0.0;
        grad[2]=0.0;

        position[0]=(FieldTYPE) deformationFieldPtrX[index];
        position[1]=(FieldTYPE) deformationFieldPtrY[index];
        position[2]=(FieldTYPE) deformationFieldPtrZ[index];

        previous[0] = static_cast<int>(reg_floor(position[0]));
        previous[1] = static_cast<int>(reg_floor(position[1]));
        previous[2] = static_cast<int>(reg_floor(position[2]));

        // basis values along the x axis
        relative=position[0]-(FieldTYPE)previous[0];
        reg_getNiftynetCubicSpline(relative, xBasis);
        reg_getNiftynetCubicSplineDerivative(relative, xDeriv);

        // basis values along the y axis
        relative=position[1]-(FieldTYPE)previous[1];
        reg_getNiftynetCubicSpline(relative, yBasis);
        reg_getNiftynetCubicSplineDerivative(relative, yDeriv);

        // basis values along the z axis
        relative=position[2]-(FieldTYPE)previous[2];
        reg_getNiftynetCubicSpline(relative, zBasis);
        reg_getNiftynetCubicSplineDerivative(relative, zDeriv);

        previous[0]--;
        previous[1]--;
        previous[2]--;

        for(c=0; c<4; c++)
        {
          Z = reg_applyBoundary<tBoundary>(previous[2] + c, floatingImage->nz);
          if (reg_checkImageDimensionIndex<tBoundary>(Z, floatingImage->nz)) {
            zPointer = &floatingIntensity[Z*floatingImage->nx*floatingImage->ny];
            xxTempNewValue=0.0;
            yyTempNewValue=0.0;
            zzTempNewValue=0.0;
            for(b=0; b<4; b++)
            {
              Y = reg_applyBoundary<tBoundary>(previous[1] + b, floatingImage->ny);
              yzPointer = &zPointer[Y*floatingImage->nx];
              if (reg_checkImageDimensionIndex<tBoundary>(Y, floatingImage->ny)) {
                xTempNewValue=0.0;
                yTempNewValue=0.0;
                for(a=0; a<4; a++)
                {
                  int X = reg_applyBoundary<tBoundary>(previous[0] + a, floatingImage->nx);

                  if (reg_checkImageDimensionIndex<tBoundary>(X, floatingImage->nx)) {
                    coeff = yzPointer[X];
                    xTempNewValue +=  coeff * xDeriv[a];
                    yTempNewValue +=  coeff * xBasis[a];
                  } // previous[0]+a in range
                  else
                  {
                    xTempNewValue +=  paddingValue * xDeriv[a];
                    yTempNewValue +=  paddingValue * xBasis[a];
                  }
                } // a
                xxTempNewValue += xTempNewValue * yBasis[b];
                yyTempNewValue += yTempNewValue * yDeriv[b];
                zzTempNewValue += yTempNewValue * yBasis[b];
              } // Y in range
              else
              {
                xxTempNewValue += paddingValue * yBasis[b];
                yyTempNewValue += paddingValue * yDeriv[b];
                zzTempNewValue += paddingValue * yBasis[b];
              }
            } // b
            grad[0] += xxTempNewValue * zBasis[c];
            grad[1] += yyTempNewValue * zBasis[c];
            grad[2] += zzTempNewValue * zDeriv[c];
          } // Z in range
          else
          {
            grad[0] += paddingValue * zBasis[c];
            grad[1] += paddingValue * zBasis[c];
            grad[2] += paddingValue * zDeriv[c];
          }
        } // c

        grad[0]=grad[0]==grad[0]?grad[0]:0.0;
        grad[1]=grad[1]==grad[1]?grad[1]:0.0;
        grad[2]=grad[2]==grad[2]?grad[2]:0.0;

        warpedGradientPtrX[index] = (GradientTYPE)grad[0];
        warpedGradientPtrY[index] = (GradientTYPE)grad[1];
        warpedGradientPtrZ[index] = (GradientTYPE)grad[2];
    }
}
/* *************************************************************** */
template<const resampler_boundary_e tBoundary, class FloatingTYPE, class GradientTYPE, class FieldTYPE>
void CubicSplineImageGradient2D(nifti_image *floatingImage,
                                nifti_image *deformationField,
                                nifti_image *warImgGradient,
                                float paddingValue,
                                int active_timepoint)
{
   if(active_timepoint<0 || active_timepoint>=(std::max)(floatingImage->nt, floatingImage->nu)){
        reg_print_fct_error("CubicSplineImageGradient2D");
        reg_print_msg_error("The specified active timepoint is not defined in the floating image");
        reg_exit();
    }
#ifdef _WIN32
    long index;
    long referenceVoxelNumber = (long)warImgGradient->nx*warImgGradient->ny;
    long floatingVoxelNumber = (long)floatingImage->nx*floatingImage->ny;
#else
    size_t index;
    size_t referenceVoxelNumber = (size_t)warImgGradient->nx*warImgGradient->ny;
    size_t floatingVoxelNumber = (size_t)floatingImage->nx*floatingImage->ny;
#endif
    FloatingTYPE *floatingIntensityPtr = static_cast<FloatingTYPE *>(floatingImage->data);
    FloatingTYPE *floatingIntensity = &floatingIntensityPtr[active_timepoint*floatingVoxelNumber];

    FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationField->data);
    FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[referenceVoxelNumber];

    GradientTYPE *warpedGradientPtrX = static_cast<GradientTYPE *>(warImgGradient->data);
    GradientTYPE *warpedGradientPtrY = &warpedGradientPtrX[referenceVoxelNumber];

#ifndef NDEBUG
    char text[255];
    sprintf(text, "2D cubic spline gradient computation of volume number %i",active_timepoint);
    reg_print_msg_debug(text);
#endif
    int previous[2], b, Y, a;
    double xBasis[4], yBasis[4], xDeriv[4], yDeriv[4], relative;
    FieldTYPE coeff, position[3], grad[2];
    FieldTYPE xTempNewValue, yTempNewValue;
    FloatingTYPE *yPointer;
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
    private(index, position, previous, xBasis, yBasis, xDeriv, yDeriv, relative, grad, coeff, \
    a, b, Y, yPointer, xTempNewValue, yTempNewValue) \
    shared(floatingIntensity, referenceVoxelNumber, floatingVoxelNumber, \
    deformationFieldPtrX, deformationFieldPtrY, paddingValue, \
    floatingImage, warpedGradientPtrX, warpedGradientPtrY)
#endif // _OPENMP
    for(index=0; index<referenceVoxelNumber; index++)
    {

      grad[0]=0.0;
      grad[1]=0.0;

      position[0]=(FieldTYPE) deformationFieldPtrX[index];
      position[1]=(FieldTYPE) deformationFieldPtrY[index];

      previous[0] = static_cast<int>(reg_floor(position[0]));
      previous[1] = static_cast<int>(reg_floor(position[1]));
      // basis values along the x axis
      relative=position[0]-(FieldTYPE)previous[0];
      relative=relative>0?relative:0;
      reg_getNiftynetCubicSpline(relative, xBasis);
      reg_getNiftynetCubicSplineDerivative(relative, xDeriv);
      // basis values along the y axis
      relative=position[1]-(FieldTYPE)previous[1];
      relative=relative>0?relative:0;
      reg_getNiftynetCubicSpline(relative, yBasis);
      reg_getNiftynetCubicSplineDerivative(relative, yDeriv);

      previous[0]--;
      previous[1]--;

      for(b=0; b<4; b++)
      {
        Y= reg_applyBoundary<tBoundary>(previous[1] + b, floatingImage->ny);
        yPointer = &floatingIntensity[Y*floatingImage->nx];
        if (reg_checkImageDimensionIndex<tBoundary>(Y, floatingImage->ny)) {
          xTempNewValue=0.0;
          yTempNewValue=0.0;
          for(a=0; a<4; a++)
          {
            int X = reg_applyBoundary<tBoundary>(previous[0] + a, floatingImage->nx);

            if (reg_checkImageDimensionIndex<tBoundary>(X, floatingImage->nx)) {
              coeff = yPointer[X];
              xTempNewValue +=  coeff * xDeriv[a];
              yTempNewValue +=  coeff * xBasis[a];
            } // previous[0]+a in range
            else
            {
              xTempNewValue +=  paddingValue * xDeriv[a];
              yTempNewValue +=  paddingValue * xBasis[a];
            }
          } // a
          grad[0] += xTempNewValue * yBasis[b];
          grad[1] += yTempNewValue * yDeriv[b];
        } // Y in range
        else
        {
          grad[0] += paddingValue * yBasis[b];
          grad[1] += paddingValue * yDeriv[b];
        }
      } // b

      grad[0]=grad[0]==grad[0]?grad[0]:0.0;
      grad[1]=grad[1]==grad[1]?grad[1]:0.0;

      warpedGradientPtrX[index] = (GradientTYPE)grad[0];
      warpedGradientPtrY[index] = (GradientTYPE)grad[1];
    }
}
/* *************************************************************** */
template <const resampler_boundary_e tBoundary, class FieldTYPE, class FloatingTYPE, class GradientTYPE>
void reg_getImageGradient3(nifti_image *floatingImage,
                           nifti_image *warImgGradient,
                           nifti_image *deformationField,
                           int interp,
                           float paddingValue,
                           int active_timepoint,
                           nifti_image *warpedImage = NULL
        )
{
    /* The deformation field contains the position in the real world */
    if(interp==3)
    {
        if(floatingImage->nz>1 || deformationField->nz>1)
        {
            CubicSplineImageGradient3D
              <tBoundary,FloatingTYPE,GradientTYPE,FieldTYPE>(floatingImage,
                                                              deformationField,
                                                              warImgGradient,
                                                              paddingValue,
                                                              active_timepoint);
        }
        else
        {
          CubicSplineImageGradient2D
            <tBoundary,FloatingTYPE,GradientTYPE,FieldTYPE>(floatingImage,
                                                            deformationField,
                                                            warImgGradient,
                                                            paddingValue,
                                                            active_timepoint);
        }
    }
    else  // trilinear interpolation [ by default ]
    {
      if(floatingImage->nz>1 || deformationField->nz>1)
      {
        TrilinearImageGradient
          <tBoundary,FloatingTYPE,GradientTYPE,FieldTYPE>(floatingImage,
                                                          deformationField,
                                                          warImgGradient,
                                                          paddingValue,
                                                          active_timepoint);
      }
      else
      {
        BilinearImageGradient
          <tBoundary,FloatingTYPE,GradientTYPE,FieldTYPE>(floatingImage,
                                                          deformationField,
                                                          warImgGradient,
                                                          paddingValue,
                                                          active_timepoint);
        }
    }
}
/* *************************************************************** */
template <const resampler_boundary_e tBoundary, class FieldTYPE, class FloatingTYPE>
void reg_getImageGradient2(nifti_image *floatingImage,
                           nifti_image *warImgGradient,
                           nifti_image *deformationField,
                           int interp,
                           float paddingValue,
                           int active_timepoint,
                           nifti_image *warpedImage
                           )
{
    switch(warImgGradient->datatype)
    {
    case NIFTI_TYPE_FLOAT32:
      reg_getImageGradient3<tBoundary,FieldTYPE,FloatingTYPE,float>
                (floatingImage,warImgGradient,deformationField,interp,paddingValue,active_timepoint, warpedImage);
        break;
    case NIFTI_TYPE_FLOAT64:
      reg_getImageGradient3<tBoundary,FieldTYPE,FloatingTYPE,double>
                (floatingImage,warImgGradient,deformationField,interp,paddingValue,active_timepoint, warpedImage);
        break;
    default:
        reg_print_fct_error("reg_getImageGradient2");
        reg_print_msg_error("The warped image data type is not supported");
        reg_exit();
    }
}
/* *************************************************************** */
template <const resampler_boundary_e tBoundary, class FieldTYPE>
void reg_getImageGradient1(nifti_image *floatingImage,
                           nifti_image *warImgGradient,
                           nifti_image *deformationField,
                           int interp,
                           float paddingValue,
                           int active_timepoint,
                           nifti_image *warpedImage
                           )
{
    switch(floatingImage->datatype)
    {
    case NIFTI_TYPE_FLOAT32:
        reg_getImageGradient2<tBoundary,FieldTYPE,float>
                (floatingImage,warImgGradient,deformationField,interp,paddingValue,active_timepoint, warpedImage);
        break;
    case NIFTI_TYPE_FLOAT64:
        reg_getImageGradient2<tBoundary,FieldTYPE,double>
                (floatingImage,warImgGradient,deformationField,interp,paddingValue,active_timepoint, warpedImage);
        break;
    default:
        reg_print_fct_error("reg_getImageGradient1");
        reg_print_msg_error("Unsupported floating image datatype");
        reg_exit();
    }
}
/* *************************************************************** */
void reg_getImageGradient(nifti_image *floatingImage,
                          nifti_image *warImgGradient,
                          nifti_image *deformationField,
                          int interp,
                          resampler_boundary_e boundary,
                          int active_timepoint,
                          nifti_image *warpedImage
                          )
{
    const float paddingValue = reg_getPaddingValue<float>(boundary);

    switch(deformationField->datatype)
    {
    case NIFTI_TYPE_FLOAT32:
      if (boundary == resampler_boundary_e::CLAMPING) {
        reg_getImageGradient1<resampler_boundary_e::CLAMPING, float>
          (floatingImage,warImgGradient,deformationField,interp,paddingValue,active_timepoint, warpedImage);
      } else if (boundary == resampler_boundary_e::REFLECTING) {
        reg_getImageGradient1<resampler_boundary_e::REFLECTING, float>
          (floatingImage,warImgGradient,deformationField,interp,paddingValue,active_timepoint, warpedImage);
      } else {
        reg_getImageGradient1<resampler_boundary_e::ZEROPAD, float>
          (floatingImage,warImgGradient,deformationField,interp,paddingValue,active_timepoint, warpedImage);
      }
      break;

    case NIFTI_TYPE_FLOAT64:
      if (boundary == resampler_boundary_e::CLAMPING) {
        reg_getImageGradient1<resampler_boundary_e::CLAMPING, double>
                (floatingImage,warImgGradient,deformationField,interp,paddingValue,active_timepoint, warpedImage);
      } else if (boundary == resampler_boundary_e::REFLECTING) {
        reg_getImageGradient1<resampler_boundary_e::REFLECTING, double>
                (floatingImage,warImgGradient,deformationField,interp,paddingValue,active_timepoint, warpedImage);
      } else {
        reg_getImageGradient1<resampler_boundary_e::ZEROPAD, double>
                (floatingImage,warImgGradient,deformationField,interp,paddingValue,active_timepoint, warpedImage);
      }
      break;
    default:
        reg_print_fct_error("reg_getImageGradient");
        reg_print_msg_error("Unsupported deformation field image datatype");
        reg_exit();
        break;
    }
}
/* *************************************************************** */
/* *************************************************************** */
template <const int t_nof_dims, const resampler_boundary_e t_boundary, const int t_kernel_size, typename interp_function_tt>
static void _compute_image_derivative(nifti_image &r_destination, const nifti_image &image, const nifti_image &deformation, const nifti_image &gradient_out,
                                      const float padvalue, const interp_function_tt &interp_function) {
  const long deformation_spatial_size = deformation.nx*deformation.ny*deformation.nz;
  const long image_spatial_size = image.nx*image.ny*image.nz;
  const float *outgradient_base = (const float*)(gradient_out.data);

  float *p_out_base = (float*)(r_destination.data);
  float const *ppc_deformation_components[t_nof_dims];

  assert(t_nof_dims == 3 || image.nz == 1);
  assert(gradient_out.nx == deformation.nx && gradient_out.ny == deformation.ny && gradient_out.nz == deformation.nz);
  assert(gradient_out.nu == image.nu);

  for (int i = 0; i < t_nof_dims; ++i) {
    ppc_deformation_components[i] = (const float*)(deformation.data) + i*deformation_spatial_size;
  }

  std::fill(p_out_base, p_out_base + r_destination.nvox, 0.f);

  for (long index = 0; index < deformation_spatial_size; ++index) {
    int base_index[t_nof_dims];
    double basis[t_nof_dims][t_kernel_size];

    {
      double relative[t_nof_dims];
      float position[t_nof_dims];

      for (int l = 0; l < t_nof_dims; ++l) {
        position[l] = ppc_deformation_components[l][index];
        base_index[l] = int(reg_floor(position[l]));
        relative[l] = position[l] - base_index[l];
        interp_function(relative[l], basis[l]);
        base_index[l] -= t_kernel_size/4;
      }
    }

    {
      const auto x_loop = [&](const int off_x, const double &basis_multiplier) {
          for (int a = 0; a < t_kernel_size; ++a) {
            int x = reg_applyBoundary<t_boundary>(base_index[0] + a, image.nx);

            if (reg_checkImageDimensionIndex<t_boundary>(x, image.nx)) {
              float const *pc_out_grad = outgradient_base + index;
              float *p_out = p_out_base + off_x + x;

              for (int m = 0; m < image.nu; ++m, pc_out_grad += deformation_spatial_size, p_out += image_spatial_size) {
                *p_out += float(basis_multiplier*basis[0][a]*(*pc_out_grad));
              }
            }
          }
        };

      if (t_nof_dims == 3) {
        for (int c = 0; c < t_kernel_size; ++c) {
          int z = reg_applyBoundary<t_boundary>(base_index[2] + c, image.nz);

          if (reg_checkImageDimensionIndex<t_boundary>(z, image.nz)) {
            const int off_y = z*image.ny;

            for (int b = 0; b < t_kernel_size; ++b) {
              int y = reg_applyBoundary<t_boundary>(base_index[1] + b, image.ny);

              if (reg_checkImageDimensionIndex<t_boundary>(y, image.ny)) {
                x_loop((y + off_y)*image.nx, basis[2][c]*basis[1][b]);
              }
            }
          }
        }
      } else {
        for (int b = 0; b < t_kernel_size; ++b) {
          int y = reg_applyBoundary<t_boundary>(base_index[1] + b, image.ny);

          if (reg_checkImageDimensionIndex<t_boundary>(y, image.ny)) {
            x_loop(y*image.nx, basis[1][b]);
          }
        }
      }
    }
  }
}
/* *************************************************************** */
template <const bool t_is_3d, const resampler_boundary_e t_boundary>
static void _compute_gradient_product_bdy(nifti_image &r_destination, const nifti_image &image, const nifti_image &deformation, const nifti_image &gradient_out,
                                          const float padvalue, const int interpolation) {
  switch (interpolation) {
  case 0:
    _compute_image_derivative<int(t_is_3d) + 2, t_boundary, 2>(r_destination, image, deformation, gradient_out, padvalue, interpNearestNeighKernel);
    break;

  case 1:
    _compute_image_derivative<int(t_is_3d) + 2, t_boundary, 2>(r_destination, image, deformation, gradient_out, padvalue, [](const double rel, double *p_out) {
        interpLinearKernel(rel, p_out);
      });
    break;

  case 3:
    _compute_image_derivative<int(t_is_3d) + 2, t_boundary, 4>(r_destination, image, deformation, gradient_out, padvalue, reg_getNiftynetCubicSpline<double, double>);
    break;

  default:
    reg_print_msg_error("Unsupported interpolation type.");
    reg_exit();
  }

}
/* *************************************************************** */
template <const bool t_is_3d>
static void _compute_gradient_product_nd(nifti_image &r_destination, const nifti_image &image, const nifti_image &deformation, const nifti_image &gradient_out,
                                         const resampler_boundary_e boundary, const int interpolation) {
  const float padvalue = reg_getPaddingValue<float>(boundary);

  switch (boundary) {
  case resampler_boundary_e::CLAMPING:
    _compute_gradient_product_bdy<t_is_3d, resampler_boundary_e::CLAMPING>(r_destination, image, deformation, gradient_out, padvalue, interpolation);
    break;

  case resampler_boundary_e::REFLECTING:
    _compute_gradient_product_bdy<t_is_3d, resampler_boundary_e::REFLECTING>(r_destination, image, deformation, gradient_out, padvalue, interpolation);
    break;

  default:
    _compute_gradient_product_bdy<t_is_3d, resampler_boundary_e::ZEROPAD>(r_destination, image, deformation, gradient_out, padvalue, interpolation);
  }
}
/* *************************************************************** */
void compute_gradient_product(nifti_image &r_destination, const nifti_image &image, const nifti_image &deformation, const nifti_image &gradient_out,
                              const resampler_boundary_e boundary, const int interpolation) {
  if (image.nz != 1 || deformation.nz != 1) {
    _compute_gradient_product_nd<true>(r_destination, image, deformation, gradient_out, boundary, interpolation);
  } else {
    _compute_gradient_product_nd<false>(r_destination, image, deformation, gradient_out, boundary, interpolation);
  }
}

#endif
