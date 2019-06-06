/**
 * @file _reg_tools.cpp
 * @author Marc Modat
 * @date 25/03/2009
 * @brief Set of useful functions
 *
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_TOOLS_CPP
#define _REG_TOOLS_CPP

#include <cmath>
#include "_reg_tools.h"

/* *************************************************************** */
/* *************************************************************** */
void reg_getRealImageSpacing(nifti_image *image,
                             float *spacingValues)
{
   float indexVoxel1[3]= {0,0,0};
   float indexVoxel2[3], realVoxel1[3], realVoxel2[3];
   reg_mat44_mul(&(image->sto_xyz), indexVoxel1, realVoxel1);

   indexVoxel2[1]=indexVoxel2[2]=0;
   indexVoxel2[0]=1;
   reg_mat44_mul(&(image->sto_xyz), indexVoxel2, realVoxel2);
   spacingValues[0]=sqrtf(reg_pow2(realVoxel1[0]-realVoxel2[0])+reg_pow2(realVoxel1[1]-realVoxel2[1])+reg_pow2(realVoxel1[2]-realVoxel2[2]));

   indexVoxel2[0]=indexVoxel2[2]=0;
   indexVoxel2[1]=1;
   reg_mat44_mul(&(image->sto_xyz), indexVoxel2, realVoxel2);
   spacingValues[1]=sqrtf(reg_pow2(realVoxel1[0]-realVoxel2[0])+reg_pow2(realVoxel1[1]-realVoxel2[1])+reg_pow2(realVoxel1[2]-realVoxel2[2]));

   if(image->nz>1)
   {
      indexVoxel2[0]=indexVoxel2[1]=0;
      indexVoxel2[2]=1;
      reg_mat44_mul(&(image->sto_xyz), indexVoxel2, realVoxel2);
      spacingValues[2]=sqrtf(reg_pow2(realVoxel1[0]-realVoxel2[0])+reg_pow2(realVoxel1[1]-realVoxel2[1])+reg_pow2(realVoxel1[2]-realVoxel2[2]));
   }
}

/* *************************************************************** */
/* *************************************************************** */
void reg_checkAndCorrectDimension(nifti_image *image)
{
   // Ensure that no dimension is set to zero
   if(image->nx<1 || image->dim[1]<1) image->dim[1]=image->nx=1;
   if(image->ny<1 || image->dim[2]<1) image->dim[2]=image->ny=1;
   if(image->nz<1 || image->dim[3]<1) image->dim[3]=image->nz=1;
   if(image->nt<1 || image->dim[4]<1) image->dim[4]=image->nt=1;
   if(image->nu<1 || image->dim[5]<1) image->dim[5]=image->nu=1;
   if(image->nv<1 || image->dim[6]<1) image->dim[6]=image->nv=1;
   if(image->nw<1 || image->dim[7]<1) image->dim[7]=image->nw=1;
   //Correcting the dim of the images
   for(int i=1;i<8;++i) {
       if(image->dim[i]>1) {
            image->dim[0]=image->ndim=i;
       }
   }
   // Set the slope to 1 if undefined
   if(image->scl_slope==0) image->scl_slope=1.f;
   // Ensure that no spacing is set to zero
   if(image->ny==1 && (image->dy==0 || image->pixdim[2]==0))
      image->dy=image->pixdim[2]=1;
   if(image->nz==1 && (image->dz==0 || image->pixdim[3]==0))
      image->dz=image->pixdim[3]=1;
   // Create the qform matrix if required
   if(image->qform_code==0 && image->sform_code==0)
   {
      image->qto_xyz=nifti_quatern_to_mat44(image->quatern_b,
                                            image->quatern_c,
                                            image->quatern_d,
                                            image->qoffset_x,
                                            image->qoffset_y,
                                            image->qoffset_z,
                                            image->dx,
                                            image->dy,
                                            image->dz,
                                            image->qfac);
      image->qto_ijk=nifti_mat44_inverse(image->qto_xyz);
   }
   // Set the voxel spacing to millimeters
   if(image->xyz_units==NIFTI_UNITS_MICRON)
   {
      for(int d=1; d<=image->ndim; ++d)
         image->pixdim[d] /= 1000.f;
      image->xyz_units=NIFTI_UNITS_MM;
   }
   if(image->xyz_units==NIFTI_UNITS_METER)
   {
      for(int d=1; d<=image->ndim; ++d)
         image->pixdim[d] *= 1000.f;
      image->xyz_units=NIFTI_UNITS_MM;
   }
   image->dx=image->pixdim[1];
   image->dy=image->pixdim[2];
   image->dz=image->pixdim[3];
   image->dt=image->pixdim[4];
   image->du=image->pixdim[5];
   image->dv=image->pixdim[6];
   image->dw=image->pixdim[7];
}
/* *************************************************************** */
/* *************************************************************** */
template <class NewTYPE, class DTYPE>
void reg_tools_changeDatatype1(nifti_image *image,int type)
{
   // the initial array is saved and freeed
   DTYPE *initialValue = (DTYPE *)malloc(image->nvox*sizeof(DTYPE));
   memcpy(initialValue, image->data, image->nvox*sizeof(DTYPE));

   // the new array is allocated and then filled
   if(type>-1){
      image->datatype=type;
   }
   else{
      if(sizeof(NewTYPE)==sizeof(unsigned char)) {
          image->datatype = NIFTI_TYPE_UINT8;
#ifndef NDEBUG
    reg_print_msg_debug("new datatype is NIFTI_TYPE_UINT8");
#endif
      }
      else if(sizeof(NewTYPE)==sizeof(float)) {
          image->datatype = NIFTI_TYPE_FLOAT32;
#ifndef NDEBUG
    reg_print_msg_debug("new datatype is NIFTI_TYPE_FLOAT32");
#endif
      }
      else if(sizeof(NewTYPE)==sizeof(double)) {
          image->datatype = NIFTI_TYPE_FLOAT64;
#ifndef NDEBUG
    reg_print_msg_debug("new datatype is NIFTI_TYPE_FLOAT64");
#endif
      }
      else {
         reg_print_fct_error("reg_tools_changeDatatype1");
         reg_print_msg_error("Only change to unsigned char, float or double are supported");
         reg_exit();
      }
   }
   free(image->data);
   image->nbyper = sizeof(NewTYPE);
   image->data = (void *)calloc(image->nvox,sizeof(NewTYPE));
   NewTYPE *dataPtr = static_cast<NewTYPE *>(image->data);
   for (size_t i = 0; i < image->nvox; i++) {
       dataPtr[i] = (NewTYPE)(initialValue[i]);
   }

   free(initialValue);
   return;
}
/* *************************************************************** */
template <class NewTYPE>
void reg_tools_changeDatatype(nifti_image *image, int type)
{
   switch(image->datatype)
   {
   case NIFTI_TYPE_UINT8:
      reg_tools_changeDatatype1<NewTYPE,unsigned char>(image,type);
      break;
   case NIFTI_TYPE_INT8:
      reg_tools_changeDatatype1<NewTYPE,char>(image,type);
      break;
   case NIFTI_TYPE_UINT16:
      reg_tools_changeDatatype1<NewTYPE,unsigned short>(image,type);
      break;
   case NIFTI_TYPE_INT16:
      reg_tools_changeDatatype1<NewTYPE,short>(image,type);
      break;
   case NIFTI_TYPE_UINT32:
      reg_tools_changeDatatype1<NewTYPE,unsigned int>(image,type);
      break;
   case NIFTI_TYPE_INT32:
      reg_tools_changeDatatype1<NewTYPE,int>(image,type);
      break;
   case NIFTI_TYPE_FLOAT32:
      reg_tools_changeDatatype1<NewTYPE,float>(image,type);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_tools_changeDatatype1<NewTYPE,double>(image,type);
      break;
   default:
      reg_print_fct_error("reg_tools_changeDatatype");
      reg_print_msg_error("Unsupported datatype");
      reg_exit();
   }
}
/* *************************************************************** */
template void reg_tools_changeDatatype<unsigned char>(nifti_image *, int);
template void reg_tools_changeDatatype<unsigned short>(nifti_image *, int);
template void reg_tools_changeDatatype<unsigned int>(nifti_image *, int);
template void reg_tools_changeDatatype<char>(nifti_image *, int);
template void reg_tools_changeDatatype<short>(nifti_image *, int);
template void reg_tools_changeDatatype<int>(nifti_image *, int);
template void reg_tools_changeDatatype<float>(nifti_image *, int);
template void reg_tools_changeDatatype<double>(nifti_image *, int);
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void reg_getDisplacementFromDeformation_2D(nifti_image *field)
{
   DTYPE *ptrX = static_cast<DTYPE *>(field->data);
   DTYPE *ptrY = &ptrX[field->nx*field->ny];

   mat44 matrix;
   if(field->sform_code>0)
      matrix=field->sto_xyz;
   else matrix=field->qto_xyz;

   int x, y,  index;
   DTYPE xInit, yInit;
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(field, matrix, ptrX, ptrY) \
   private(x, y, index, xInit, yInit)
#endif
   for(y=0; y<field->ny; y++)
   {
      index=y*field->nx;
      for(x=0; x<field->nx; x++)
      {

         // Get the initial control point position
         xInit = matrix.m[0][0]*(DTYPE)x
               + matrix.m[0][1]*(DTYPE)y
               + matrix.m[0][3];
         yInit = matrix.m[1][0]*(DTYPE)x
               + matrix.m[1][1]*(DTYPE)y
               + matrix.m[1][3];

         // The initial position is subtracted from every values
         ptrX[index] -= xInit;
         ptrY[index] -= yInit;
         index++;
      }
   }
}
/* *************************************************************** */
template<class DTYPE>
void reg_getDisplacementFromDeformation_3D(nifti_image *field)
{
   DTYPE *ptrX = static_cast<DTYPE *>(field->data);
   DTYPE *ptrY = &ptrX[field->nx*field->ny*field->nz];
   DTYPE *ptrZ = &ptrY[field->nx*field->ny*field->nz];

   mat44 matrix;
   if(field->sform_code>0)
      matrix=field->sto_xyz;
   else matrix=field->qto_xyz;

   int x, y, z, index;
   float xInit, yInit, zInit;
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(field, matrix, \
   ptrX, ptrY, ptrZ) \
   private(x, y, z, index, xInit, yInit, zInit)
#endif
   for(z=0; z<field->nz; z++)
   {
      index=z*field->nx*field->ny;
      for(y=0; y<field->ny; y++)
      {
         for(x=0; x<field->nx; x++)
         {
            // Get the initial control point position
            xInit = matrix.m[0][0]*static_cast<float>(x)
                  + matrix.m[0][1]*static_cast<float>(y)
                  + matrix.m[0][2]*static_cast<float>(z)
                  + matrix.m[0][3];
            yInit = matrix.m[1][0]*static_cast<float>(x)
                  + matrix.m[1][1]*static_cast<float>(y)
                  + matrix.m[1][2]*static_cast<float>(z)
                  + matrix.m[1][3];
            zInit = matrix.m[2][0]*static_cast<float>(x)
                  + matrix.m[2][1]*static_cast<float>(y)
                  + matrix.m[2][2]*static_cast<float>(z)
                  + matrix.m[2][3];

            // The initial position is subtracted from every values
            ptrX[index] -= static_cast<DTYPE>(xInit);
            ptrY[index] -= static_cast<DTYPE>(yInit);
            ptrZ[index] -= static_cast<DTYPE>(zInit);
            index++;
         }
      }
   }
}
/* *************************************************************** */
int reg_getDisplacementFromDeformation(nifti_image *field)
{
   if(field->datatype==NIFTI_TYPE_FLOAT32)
   {
      switch(field->nu)
      {
      case 2:
         reg_getDisplacementFromDeformation_2D<float>(field);
         break;
      case 3:
         reg_getDisplacementFromDeformation_3D<float>(field);
         break;
      default:
         reg_print_fct_error("reg_getDisplacementFromDeformation");
         reg_print_msg_error("Only implemented for 5D image with 2 or 3 components in the fifth dimension");
         reg_exit();
      }
   }
   else if(field->datatype==NIFTI_TYPE_FLOAT64)
   {
      switch(field->nu)
      {
      case 2:
         reg_getDisplacementFromDeformation_2D<double>(field);
         break;
      case 3:
         reg_getDisplacementFromDeformation_3D<double>(field);
         break;
      default:
         reg_print_fct_error("reg_getDisplacementFromDeformation");
         reg_print_msg_error("Only implemented for 5D image with 2 or 3 components in the fifth dimension");
         reg_exit();
      }
   }
   else
   {
      reg_print_fct_error("reg_getDisplacementFromDeformation");
      reg_print_msg_error("Only single or double floating precision have been implemented");
      reg_exit();
   }
   field->intent_code=NIFTI_INTENT_VECTOR;
   memset(field->intent_name, 0, 16);
   strcpy(field->intent_name,"NREG_TRANS");
   if(field->intent_p1==DEF_FIELD)
      field->intent_p1=DISP_FIELD;
   if(field->intent_p1==DEF_VEL_FIELD)
      field->intent_p1=DISP_VEL_FIELD;
   return EXIT_SUCCESS;
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void reg_getDeformationFromDisplacement_2D(nifti_image *field)
{
   DTYPE *ptrX = static_cast<DTYPE *>(field->data);
   DTYPE *ptrY = &ptrX[field->nx*field->ny];

   mat44 matrix;
   if(field->sform_code>0)
      matrix=field->sto_xyz;
   else matrix=field->qto_xyz;

   int x, y, index;
   DTYPE xInit, yInit;
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(field, matrix, \
   ptrX, ptrY) \
   private(x, y, index, xInit, yInit)
#endif
   for(y=0; y<field->ny; y++)
   {
      index=y*field->nx;
      for(x=0; x<field->nx; x++)
      {

         // Get the initial control point position
         xInit = matrix.m[0][0]*(DTYPE)x
               + matrix.m[0][1]*(DTYPE)y
               + matrix.m[0][3];
         yInit = matrix.m[1][0]*(DTYPE)x
               + matrix.m[1][1]*(DTYPE)y
               + matrix.m[1][3];

         // The initial position is added from every values
         ptrX[index] += xInit;
         ptrY[index] += yInit;
         index++;
      }
   }
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void reg_getDeformationFromDisplacement_3D(nifti_image *field)
{
   DTYPE *ptrX = static_cast<DTYPE *>(field->data);
   DTYPE *ptrY = &ptrX[field->nx*field->ny*field->nz];
   DTYPE *ptrZ = &ptrY[field->nx*field->ny*field->nz];

   mat44 matrix;
   if(field->sform_code>0)
      matrix=field->sto_xyz;
   else matrix=field->qto_xyz;

   int x, y, z, index;
   float xInit, yInit, zInit;
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(field, matrix, ptrX, ptrY, ptrZ) \
   private(x, y, z, index, xInit, yInit, zInit)
#endif
   for(z=0; z<field->nz; z++)
   {
      index=z*field->nx*field->ny;
      for(y=0; y<field->ny; y++)
      {
         for(x=0; x<field->nx; x++)
         {

            // Get the initial control point position
            xInit = matrix.m[0][0]*static_cast<float>(x)
                  + matrix.m[0][1]*static_cast<float>(y)
                  + matrix.m[0][2]*static_cast<float>(z)
                  + matrix.m[0][3];
            yInit = matrix.m[1][0]*static_cast<float>(x)
                  + matrix.m[1][1]*static_cast<float>(y)
                  + matrix.m[1][2]*static_cast<float>(z)
                  + matrix.m[1][3];
            zInit = matrix.m[2][0]*static_cast<float>(x)
                  + matrix.m[2][1]*static_cast<float>(y)
                  + matrix.m[2][2]*static_cast<float>(z)
                  + matrix.m[2][3];

            // The initial position is subtracted from every values
            ptrX[index] += static_cast<DTYPE>(xInit);
            ptrY[index] += static_cast<DTYPE>(yInit);
            ptrZ[index] += static_cast<DTYPE>(zInit);
            index++;
         }
      }
   }
}
/* *************************************************************** */
/* *************************************************************** */
int reg_getDeformationFromDisplacement(nifti_image *field)
{
   if(field->datatype==NIFTI_TYPE_FLOAT32)
   {
      switch(field->nu)
      {
      case 2:
         reg_getDeformationFromDisplacement_2D<float>(field);
         break;
      case 3:
         reg_getDeformationFromDisplacement_3D<float>(field);
         break;
      default:
         reg_print_fct_error("reg_getDeformationFromDisplacement");
         reg_print_msg_error("Only implemented for 2 or 3D deformation fields");
         reg_exit();
      }
   }
   else if(field->datatype==NIFTI_TYPE_FLOAT64)
   {
      switch(field->nu)
      {
      case 2:
         reg_getDeformationFromDisplacement_2D<double>(field);
         break;
      case 3:
         reg_getDeformationFromDisplacement_3D<double>(field);
         break;
      default:
         reg_print_fct_error("reg_getDeformationFromDisplacement");
         reg_print_msg_error("Only implemented for 2 or 3D deformation fields");
         reg_exit();
      }
   }
   else
   {
      reg_print_fct_error("reg_getDeformationFromDisplacement");
      reg_print_msg_error("Only single or double floating precision have been implemented");
      reg_exit();
   }

   field->intent_code=NIFTI_INTENT_VECTOR;
   memset(field->intent_name, 0, 16);
   strcpy(field->intent_name,"NREG_TRANS");
   if(field->intent_p1==DISP_FIELD)
      field->intent_p1=DEF_FIELD;
   if(field->intent_p1==DISP_VEL_FIELD)
      field->intent_p1=DEF_VEL_FIELD;
   return EXIT_SUCCESS;
}
/* *************************************************************** */
void mat44ToCptr(mat44 mat, float* cMat)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			cMat[i * 4 + j] = mat.m[i][j];
		}
	}
}
/* *************************************************************** */
void cPtrToMat44(mat44 *mat, float* cMat)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			 mat->m[i][j]=cMat[i * 4 + j];
		}
	}
}
/* *************************************************************** */
void mat33ToCptr(mat33 *mat, float* cMat, const unsigned int numMats)
{
	for (size_t k = 0; k < numMats; k++)
	{
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				cMat[9*k +i * 3 + j] = mat[k].m[i][j];

			}
		}
	}
}
/* *************************************************************** */
void cPtrToMat33(mat33 *mat, float* cMat)
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
             mat->m[i][j]=cMat[i * 3 + j];
        }
    }
}
/* *************************************************************** */
template<typename T>
void matmnToCptr(T** mat, T* cMat, unsigned int m, unsigned int n) {
    for (unsigned int i = 0; i < m; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            cMat[i * n + j] = mat[i][j];
        }
    }
}
template void matmnToCptr<float>(float** mat, float* cMat, unsigned int m, unsigned int n);
template void matmnToCptr<double>(double** mat, double* cMat, unsigned int m, unsigned int n);
/* *************************************************************** */
template<typename T>
void cPtrToMatmn(T** mat, T* cMat, unsigned int m, unsigned int n) {
    for (unsigned int i = 0; i < m; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
             mat[i][j]=cMat[i * n + j];
        }
    }
}
template void cPtrToMatmn<float>(float** mat, float* cMat, unsigned int m, unsigned int n);
template void cPtrToMatmn<double>(double** mat, double* cMat, unsigned int m, unsigned int n);
/* *************************************************************** */
void coordinateFromLinearIndex(int index, int maxValue_x, int maxValue_y, int &x, int &y, int &z)
{
    x =  index % (maxValue_x+1);
    index /= (maxValue_x+1);
    y = index % (maxValue_y+1);
    index /= (maxValue_y+1);
    z = index;
}
/* *************************************************************** */
#endif
