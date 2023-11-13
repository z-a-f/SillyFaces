r'''

Rules:

1. Younger have brighter eyes
2. Older have brighter skin
3. Younger have eyes that are closer to each other
4. Older have mouth closer to the center

Example usage:

python ./make_face_age.py --img_width 128 --img_height 128 --num_ids 200 --images_per_id 100

'''
import argparse
import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from faces.shape_color_config import Color
from faces.face_config import FaceConfig
from faces.sample_utils import sample_config, unroll_samples
from faces.sample_utils import read_ranges


def parse_args():
    parser = argparse.ArgumentParser()
    # Randomization parameters
    parser.add_argument('--seed', type=int, default=None)
    # Data parameters
    parser.add_argument('--num_ids', type=int, default=100)
    parser.add_argument('--images_per_id', type=int, default=50)
    parser.add_argument('--datafolder_name', type=str, default='generated/silly_faces/images')
    parser.add_argument('--ranges_file', type=str, default='seeds/ranges.yaml')
    # Image parameters
    parser.add_argument('--img_width', type=int, default=224)
    parser.add_argument('--img_height', type=int, default=224)
    # Image randomization parameters (scaling and offset)
    parser.add_argument('--scale_min', type=float, default=0.95)
    parser.add_argument('--scale_max', type=float, default=1.05)
    parser.add_argument('--offset_min', type=float, default=-0.05)
    parser.add_argument('--offset_max', type=float, default=0.05)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    ranges = read_ranges(args.ranges_file)
    datafolder_name = args.datafolder_name
    
    num_ids = args.num_ids
    images_per_id = args.images_per_id

    N = num_ids * images_per_id
    H = args.img_height
    W = args.img_width
    C = len(ranges['background']['color'][0])

    scale_min = args.scale_min
    scale_max = args.scale_max
    offset_min = args.offset_min
    offset_max = args.offset_max

    # Make some people
    random_samples = sample_config(ranges, num_ids, ('eyes', 'mouth', 'face'))
    random_samples = unroll_samples(random_samples,
        defaults={
            'eyes': {
                'z': 3,
                'img_ratio': 0.2
            },
            'mouth': {
                'z': 2,
                'img_ratio': 0.4
            },
            'face': {
                'z': 1,
                'img_ratio': 0.9
            },
        }
    )

    # Age distribution
    final_value = 0.1
    c = np.log2(1.0 / final_value) / ranges['age'][-1]
    age_distribution = np.array([1.0 / 2**(c * age) for age in ranges['age']])
    age_distribution1 = age_distribution / age_distribution.sum()
    ages = np.random.choice(ranges['age'], replace=True, size=(num_ids, images_per_id), p=age_distribution1)

    # Color distributions
    indices = np.arange(len(ranges['background']['color']))
    indices = np.random.choice(indices, replace=True, size=(num_ids, images_per_id))
    backgrounds = ranges['background']['color'][indices]


    # all_images = np.zeros((N, H, W, C), dtype=float)
    all_person_ids = np.repeat(range(num_ids), images_per_id)
    all_ages = ages.flatten()

    
    os.makedirs(datafolder_name, exist_ok=True)

    for person_id in tqdm.trange(num_ids):
        for img_id in range(images_per_id):
            img_idx = person_id * images_per_id + img_id
            face = FaceConfig(**random_samples[person_id], age=all_ages[img_idx])
            img = face.make_img(H, W,
                                scale=np.random.uniform(scale_min, scale_max),
                                offset=tuple(np.random.uniform(offset_min, offset_max, size=2)),
                                background_color=backgrounds[person_id, img_id],
                                aspil=True)
            file_name = f'{img_idx}_{person_id}_{all_ages[img_idx]}.png'
            img.save(os.path.join(datafolder_name, file_name))
