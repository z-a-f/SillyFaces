
import os
import numpy as np
import csv
import pandas as pd

def make_data_splits(dir, train_fraction=0.8, age_groups=None):
    age_groups = age_groups or np.linspace(10, 100, 10)[:-1]
    files = os.listdir(dir)
    np.random.shuffle(files)

    files_meta = [map(int, filename[:-4].split('_')) for filename in files]
    _, person_ids, ages = zip(*files_meta)
    files = [os.path.join(dir, filename) for filename in files]

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
    train_data.to_csv('train.csv', index=False)
    valid_data.to_csv('valid.csv', index=False)

make_data_splits('images')
