# Input filename matching

## Cross-subject analysis
By default, NiftyNet treats each image file as a subject,
training/validation/inference procedures are designed in a cross-subject
manner.

To facilitate the cross-subject analysis, the user should specify lists of
files to be used. For example, the relevant configurations could be:

```ini
[SYSTEM]
dataset_split_file = '/mnt/data/cross_validation_fold_01.csv'

[MRI_T1]
csv_file = '/mnt/data/t1_list.csv'

[segmentation_target]
csv_file = '/mnt/data/ground_truth.csv'
```

where `[MRI_T1]` and `[segmentation_target]` are input source sections, with
.csv files specify the lists of input images; `dataset_split_file` under the
`[SYSTEM]` section specifies the partitioning of the dataset.

The csv files should be created beforehand by the user and share the same set
of unique subject identifier (“subject ID”) among them, for example:

Contents of `t1_list.csv`:
```
subject_001,/mnt/data/t1/T1_001_img.nii.gz
subject_002,/mnt/data/t1/T1_002_img.nii.gz
```

Contents of `ground_truth.csv`:
```
subject_001,/mnt/data/ground_truth/001_img_seg.nii.gz
subject_002,/mnt/data/ground_truth/002_img_seg.nii.gz
```

Contents of `cross_validation_fold_01.csv`:
```
subject_001,training
subject_002,inference
```

In this example, image and the corresponding ground truth of `subject_001` will
be used for training; `subject_002` will be used at the inference phase.

## Automatic filename matching
Manually creating the .csv files could be error-prone, NiftyNet also provides
automatic file searching functionalities.

### Searching file by filename
The configuration parameters for filename searching are:
- `path_to_search`
- `filename_contains`
- `filename_not_contains`
Multiple values are supported for these parameters. For example:
```
path_to_search = /mnt/data/image_folder_1, /mnt/shared/image_folder_2
filename_contains = img, T1
filename_not_contains = label, seg
```
will find all files in folder `/mnt/data/image_folder_1` and
`/mnt/shard/image_folder_2` with name containing "img" and "T1", and name not
containing "label" and "seg". The subject ID will be automatically assigned as
filename without the `filename_contains` keywords and the file extension names.

Based on these criteria, regular file name
```
/mnt/data/image_folder_1/T1_001_img.nii.gz
```
will be matched. The subject ID will be `_001_` which is `T1_001_img` without
the matched keywords "img" and "T1".

As a result, a new line will be appended to the automatically generated .csv
file:
```
_001_,/mnt/data/image_folder_1/T1_001_img.nii.gz
```

NiftNet will go through each of the `path_to_each`, and extract subject IDs.  A
`ValueError` will be raised when the same subject ID are extracted from more
than one filename.

### Extracting subject ID
By default, subject ID is automatically determined as removing
`filename_contains` and file extension names from the matched filename. This
behaviour could be altered by further specifying:
```
filename_removefromid = img
```
NiftyNet interprets the value of `filename_removefromid` parameter as regular
expression, and uses the output of
```python
# replacing matched filename_removefromid with an empty string
re.sub(filename_removefromid, '', input_filename)
```
as the subject ID.
