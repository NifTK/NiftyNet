import os
import nibabel as nib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import re


def dice_score(gt, img):
    """Calculate the dice score for evaluation purposes"""
    gt, img = [x > 0 for x in (gt, img)]
    num = 2 * np.sum(gt & img)
    den = gt.sum() + img.sum()
    return num / den


def results(ground_truths, est_dirs):
    """Collates the dice scores from various experiments"""
    result = {e: [] for e in est_dirs}
    result['ids'] = []
    for f in ground_truths:
        r = re.search('.*bin_seg_(.*_\d+)', f)
        if r:
            gt = imread(f)
            subj_id = r.group(1)
            result['ids'].append(subj_id)
            for exp_name in est_dirs:
                est_path = os.path.join(est_dirs[exp_name], subj_id + '_niftynet_out.nii.gz')
                est = nib.load(est_path).get_data().squeeze()
                result[exp_name].append(dice_score(gt, est))

    df = pd.DataFrame(result)
    return df


def results_long(df, csv_name):
    """Labels the results as from train or validation datasets"""
    d_split = pd.read_csv(csv_name)
    d_split.columns = ('ids', 'fold')
    merged_df = pd.merge(df, d_split)
    df_long = pd.melt(merged_df, id_vars=['ids', 'fold'])
    return df_long


def add_experiment_info_to_datasets(df, est_dirs):
    """adds the experimental information from the training settings"""
    experiment_numbers, flipping, dataset_splits, deforming = [], [], [], []

    for est_dir_key in est_dirs:
        # getting the dataset_split file from the settings_train txt file:
        train_settings = ' '.join([l.strip() for l in open(est_dirs[est_dir_key] + '../settings_train.txt', 'r')])

        experiment_numbers.append(est_dir_key)

        r = re.search('dataset_split_file:\s.*(\d).csv', train_settings)
        dataset_splits.append(r.group(1))

        r = re.search('flipping_axes:\s\((.*?)\)', train_settings)
        flip = 'False' if '-1' in r.group(1) else 'True'
        flipping.append(flip)

        r = re.search('elastic_deformation:\s(\w+)', train_settings)
        deforming.append(r.group(1))

    data_dict = {'variable': experiment_numbers,
                 'flip': flipping,
                 'deform': deforming,
                 'train_split': dataset_splits,
                 'augmentations': ['_'.join(['flip', x[0], 'def', y[0]]) for x, y in zip(flipping, deforming)]
                 }

    conditions_df = pd.DataFrame(data_dict)
    combined_df = pd.merge(df, conditions_df)

    return combined_df


def get_and_plot_results(ground_truths, est_dirs, subj_ids):
    df = None
    for est_dir_key in est_dirs:

        # getting the dataset_split file from the settings_train txt file:
        train_settings = [l.strip() for l in open(est_dirs[est_dir_key] + '../settings_train.txt', 'r')]
        dataset_split_file = [x.split(':')[1].strip() for x in train_settings if 'dataset_split' in x][0]

        new_df = results(ground_truths, {est_dir_key: est_dirs[est_dir_key]})
        new_df_long = results_long(new_df, dataset_split_file)

        f, axes = plt.subplots(2, 1, figsize=(9, 5))
        f.suptitle("Experiment %s" % est_dir_key)
        show_model_outputs(ground_truths, new_df_long, {est_dir_key: est_dirs[est_dir_key]}, subj_ids, axes)

        if df is None:
            df = new_df_long
        else:
            df = pd.concat([df, new_df_long])

    combined_df = add_experiment_info_to_datasets(df, est_dirs)
    return combined_df


def show_model_outputs(ground_truths, df, est_dirs, subj_ids, axes):
    """Plots the results for visualisation"""
    for est_dir in est_dirs.values():
        for i, sid in enumerate(subj_ids):
            a = imread([f for f in ground_truths if sid in f][0])
            b = nib.load(est_dir + '/' + sid + '_niftynet_out.nii.gz').get_data().squeeze()

            axes[i].imshow(np.hstack([a, b, a - b]), cmap='gray')
            axes[i].set_axis_off()

            train_or_val = df[df['ids'] == sid]['fold'].values[0]
            axes[i].set_title('{} Fold: Ground truth, Estimate and Difference. Dice Score = {:.2f}'.format(
                train_or_val, dice_score(a, b)))
