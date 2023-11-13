r'''

Example usage:

python ./make_splits.py --root_dir ./generated/silly_faces --img_dir images --forget_person_ids 10 --forget_age_groups 0 1

'''

import argparse
import os
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--root_dir', type=str, default='images')
    parser.add_argument('--img_dir', type=str, default=None)
    parser.add_argument('--age_groups', type=str, default=None)
    parser.add_argument('--train_fraction', type=float, default=0.8)
    parser.add_argument('--forget_person_ids', type=int, default=0)  # Number of persons to forget
    parser.add_argument('--forget_age_groups', type=int, nargs='*', default=None)  # Specific age groups to forget from the person list
    args = parser.parse_args()
    return args

def make_data_splits(rootdir, imgdir=None, train_fraction=0.8, age_groups=None, num_persons_forget=0, forget_age_groups=None):
    # If the imgdir is None, it will be the same as the rootdir
    # If the imgdir is not None, it will be a subdirectory of the rootdir
    if imgdir is None:
        csv_prefix = ''
        imgdir = rootdir
        dir = os.path.abspath(imgdir)
    else:
        csv_prefix = imgdir
        dir = os.path.join(rootdir, imgdir)
    
    age_groups = age_groups or np.linspace(10, 100, 10)[:-1]
    files = os.listdir(dir)
    np.random.shuffle(files)

    files_meta = [map(int, filename[:-4].split('_')) for filename in files]
    _, person_ids, ages = zip(*files_meta)
    files = [os.path.join(csv_prefix, filename) for filename in files]

    train_idx = int(len(files) * train_fraction)

    # Format: <img_id>_<person_id>_<age>.png

    train_data = pd.DataFrame({
        'filename': files[:train_idx],
        'person_id': person_ids[:train_idx],
        'age': ages[:train_idx],
        'age_group': np.digitize(ages[:train_idx], bins=age_groups),
        })
    valid_data = pd.DataFrame({
        'filename': files[train_idx:],
        'person_id': person_ids[train_idx:],
        'age': ages[train_idx:],
        'age_group': np.digitize(ages[train_idx:], bins=age_groups),
        })
    train_data.to_csv(os.path.join(rootdir, 'train.csv'), index=False)
    valid_data.to_csv(os.path.join(rootdir, 'valid.csv'), index=False)

    # Randomly pick 'num_persons_forget' persons and split the train data into retain and forget
    if num_persons_forget > 0:
        person_ids = np.unique(person_ids)
        np.random.shuffle(person_ids)
        forget_person_ids = person_ids[:num_persons_forget]

        # Create a mask for the forget persons of a specific age group
        forget_mask = train_data['person_id'].isin(forget_person_ids) # Mask for the forget persons
        if forget_age_groups is not None:
            for age_group in forget_age_groups:
                forget_mask = forget_mask & train_data['age_group'].isin(forget_age_groups)
        forget_data = train_data[forget_mask]
        retain_data = train_data[~forget_mask]
        forget_data.to_csv(os.path.join(rootdir, 'forget.csv'), index=False)
        retain_data.to_csv(os.path.join(rootdir, 'retain.csv'), index=False)

if __name__ == '__main__':
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
    make_data_splits(args.root_dir, args.img_dir, args.train_fraction, args.age_groups, args.forget_person_ids, args.forget_age_groups)
