import os
import zipfile
zip_dir='.'
target_dir='.'
for zip_filename in {'TrainingData_Part1.zip','TrainingData_Part1.zip','TrainingData_Part1.zip'}:
  zip_ref = zipfile.ZipFile(os.path.join(zip_dir,zip_filename), 'r')
  zip_ref.extractall(target_dir)
  zip_ref.close()

import SimpleITK
import glob
for fn in glob.glob('*.mhd'):
  if 'segmentation' in fn:
    fn_out=fn[:-4]+'.nii.gz'
  else:
    fn_out=fn[:-4]+'_T2.nii.gz'
  SimpleITK.WriteImage(SimpleITK.ReadImage(fn),fn_out)
  try:
    os.remove(fn)
    os.remove(fn[:-4]+'.raw')
  except:
    pass


