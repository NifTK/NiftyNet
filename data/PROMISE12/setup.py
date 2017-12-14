"""
Unzip data downloaded from challenge website:
https://promise12.grand-challenge.org/

The outcome should be three folders named:
TrainingData_Part1, TrainingData_Part2, TrainingData_Part3
each folder contains multiple '.mhd' and '.raw' files
"""
import os
import zipfile

zip_dir = '.'
target_dir = '.'
for zip_filename in {'TrainingData_Part1.zip', 'TrainingData_Part2.zip',
                     'TrainingData_Part3.zip'}:
    print('Extracting', zip_filename, '...')
    zip_ref = zipfile.ZipFile(os.path.join(zip_dir, zip_filename), 'r')
    zip_ref.extractall(os.path.basename(zip_filename.replace('.zip', '')))
    zip_ref.close()
