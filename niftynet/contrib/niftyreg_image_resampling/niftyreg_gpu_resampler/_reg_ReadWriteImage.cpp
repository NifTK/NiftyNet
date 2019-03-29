/*
 *  _reg_ReadWriteImage.cpp
 *
 *
 *  Created by Marc Modat on 30/05/2012.
 *  Copyright (c) 2012, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_READWRITEIMAGE_CPP
#define _REG_READWRITEIMAGE_CPP

#include "_reg_ReadWriteImage.h"
#include "_reg_tools.h"
#include "_reg_stringFormat.h"

/* *************************************************************** */
void reg_hack_filename(nifti_image *image, const char *filename)
{
   std::string name(filename);
   name.append("\0");
   // Free the char arrays if already allocated
   if(image->fname) free(image->fname);
   if(image->iname) free(image->iname);
   // Allocate the char arrays
   image->fname = (char *)malloc((name.size()+1)*sizeof(char));
   image->iname = (char *)malloc((name.size()+1)*sizeof(char));
   // Copy the new name in the char arrays
   strcpy(image->fname,name.c_str());
   strcpy(image->iname,name.c_str());
   // Returns at the end of the function
   return;
}
/* *************************************************************** */
int reg_io_checkFileFormat(const char *filename)
{
   // Nifti format is used by default
   // Check the extention of the provided filename
   std::string b(filename);
   if(b.find( ".nii.gz") != std::string::npos)
      return NR_NII_FORMAT;
   else if(b.find( ".nii") != std::string::npos)
      return NR_NII_FORMAT;
   else if(b.find( ".hdr") != std::string::npos)
      return NR_NII_FORMAT;
   else if(b.find( ".img.gz") != std::string::npos)
      return NR_NII_FORMAT;
   else if(b.find( ".img") != std::string::npos)
      return NR_NII_FORMAT;
   else
   {
      reg_print_fct_warn("reg_io_checkFileFormat");
      reg_print_msg_warn("No filename extension provided - the Nifti library is used by default");
   }

   return NR_NII_FORMAT;
}
/* *************************************************************** */
nifti_image *reg_io_ReadImageFile(const char *filename)
{
   // First read the fileformat in order to use the correct library
   int fileFormat=reg_io_checkFileFormat(filename);

   // Create the nifti image pointer
   nifti_image *image=NULL;

   // Read the image and convert it to nifti format if required
   switch(fileFormat)
   {
   case NR_NII_FORMAT:
      image=nifti_image_read(filename,true);
      reg_hack_filename(image,filename);
      break;
   }
   reg_checkAndCorrectDimension(image);

   // Return the nifti image
   return image;
}
/* *************************************************************** */
nifti_image *reg_io_ReadImageHeader(const char *filename)
{
   // First read the fileformat in order to use the correct library
   int fileFormat=reg_io_checkFileFormat(filename);

   // Create the nifti image pointer
   nifti_image *image=NULL;

   // Read the image and convert it to nifti format if required
   switch(fileFormat)
   {
   case NR_NII_FORMAT:
      image=nifti_image_read(filename,false);
      break;
   }
   reg_checkAndCorrectDimension(image);

   // Return the nifti image
   return image;
}
/* *************************************************************** */
void reg_io_WriteImageFile(nifti_image *image, const char *filename)
{
   // First read the fileformat in order to use the correct library
   int fileFormat=reg_io_checkFileFormat(filename);

   // Convert the image to the correct format if required, set the filename and save the file
   switch(fileFormat)
   {
   case NR_NII_FORMAT:
      nifti_set_filenames(image,filename,0,0);
      nifti_image_write(image);
      break;
   }

   // Return
   return;
}
/* *************************************************************** */
template <class DTYPE>
void reg_io_diplayImageData1(nifti_image *image)
{
    reg_print_msg_debug("image values:");
    size_t voxelNumber = (size_t)image->nx * image->ny * image->nz;
    DTYPE *data = static_cast<DTYPE *>(image->data);
    std::string text;

    size_t voxelIndex=0;
    for(int z=0; z<image->nz; z++)
    {
       for(int y=0; y<image->ny; y++)
       {
          for(int x=0; x<image->nx; x++)
          {
             text = stringFormat("[%d - %d - %d] = [", x, y, z);
             for(int tu=0;tu<image->nt*image->nu; ++tu){
                text = stringFormat("%s%g ", text.c_str(),
                    static_cast<double>(data[voxelIndex + tu*voxelNumber]));
             }
             text = stringFormat("%s]", text.c_str());
             reg_print_msg_debug(text.c_str());
          }
       }
    }
}
/* *************************************************************** */
void reg_io_diplayImageData(nifti_image *image)
{
    switch(image->datatype)
    {
    case NIFTI_TYPE_UINT8:
       reg_io_diplayImageData1<unsigned char>(image);
       break;
    case NIFTI_TYPE_INT8:
       reg_io_diplayImageData1<char>(image);
       break;
    case NIFTI_TYPE_UINT16:
       reg_io_diplayImageData1<unsigned short>(image);
       break;
    case NIFTI_TYPE_INT16:
       reg_io_diplayImageData1<short>(image);
       break;
    case NIFTI_TYPE_UINT32:
       reg_io_diplayImageData1<unsigned int>(image);
       break;
    case NIFTI_TYPE_INT32:
       reg_io_diplayImageData1<int>(image);
       break;
    case NIFTI_TYPE_FLOAT32:
       reg_io_diplayImageData1<float>(image);
       break;
    case NIFTI_TYPE_FLOAT64:
       reg_io_diplayImageData1<double>(image);
       break;
    default:
       reg_print_fct_error("reg_io_diplayImageData");
       reg_print_msg_error("Unsupported datatype");
       reg_exit();
    }
   return;
}
/* *************************************************************** */
#endif
