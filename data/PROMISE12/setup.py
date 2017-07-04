import os
import zipfile
zip_dir='.'
target_dir='.'
for zip_filename in {'TrainingData_Part1.zip','TrainingData_Part1.zip','TrainingData_Part1.zip'}:
  zip_ref = zipfile.ZipFile(os.path.join(zip_dir,zip_filename), 'r')
  zip_ref.extractall(target_dir)
  zip_ref.close()
