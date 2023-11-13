r'''

Rules:

1. Younger have brighter eyes
2. Older have brighter skin
3. Younger have eyes that are closer to each other
4. Older have mouth closer to the center


'''
import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import pickle as pkl

from PIL import Image

kPink = np.array([1.0, 0.75, 0.79])
kReddish = np.array([0.5, 0.2, 0.2])
kRed = np.array([1.0, 0.0, 0.0])
kGreen = np.array([0.0, 1.0, 0.0])
kBlue = np.array([0.0, 0.0, 1.0])
kHalfGrey = np.array([0.5, 0.5, 0.5])


def make_rect(width, height):
    mask = np.ones((height, width), dtype=int)
    return mask

def make_oval(width, height):
    r = min(width, height) // 2 * 10

    ix = np.arange(2 * r) - r
    iy = np.arange(2 * r) - r

    iX, iY = np.meshgrid(ix, iy)
    iX = iX.reshape(-1)
    iY = iY.reshape(-1)
    fill_mask = np.zeros(iX.shape, dtype=bool)
    fill_mask[(iX*iX + iY*iY) <= r * r] = True
    fill_mask = fill_mask.reshape(2 * r, 2 * r)

    # fill_mask = fill_mask.resize(height, width)
    fill_mask = Image.fromarray(fill_mask)
    fill_mask = fill_mask.resize((width, height))

    return np.array(fill_mask).astype(int)


@dataclass
class ShapeColor:
    shape: str
    center_offset: tuple  # (x, y) location of this shape's center with respect to the center location. Should be float from -0.5 to 0.5
    color: tuple  # RGB ranging from 0.0 to 1.0 each
    z: int = 0  # Elevation
    shape_ratio: float = 1.0  # width / height
    img_ratio: float = 0.5  # Fraction of the image that this shape occupies

    def __hash__(self):
        # This will define if two shapes are representing the same thing
        return hash((self.img_ratio, self.shape, self.shape_ratio, self.center_offset, self.color, self.z))

    def add_shape_mask(self, mask):
        iw, ih = mask.shape
        side = iw * self.img_ratio  # Square dimensions for this part

        if self.shape_ratio >= 1.0:
            w = side
            h = side / self.shape_ratio
        else:
            h = side
            w = side * self.shape_ratio

        w = int(w)
        h = int(h)

        if self.shape == 'rect':
            submask = make_rect(w, h) * self.z
        elif self.shape == 'oval':
            submask = make_oval(w, h) * self.z
        else:
            raise NotImplementedError(f'Shape "{self.shape}" is not recognized')

        # Get the (0, 0) location
        mask_x = mask.shape[1]
        mask_y = mask.shape[0]
        cx = mask_x * (0.5 + self.center_offset[0])
        cy = mask_y * (0.5 + self.center_offset[1])

        submask_x = submask.shape[1]
        submask_y = submask.shape[0]

        px0 = int(round(cx - submask_x / 2))
        py0 = int(round(cy - submask_y / 2))
        px1 = int(round(px0 + submask_x))
        py1 = int(round(py0 + submask_y))

        if px0 >= mask_x or py0 >= mask_y or px1 < 0 or py1 < 0:
            return mask

        if px0 < 0:
            submask = submask[:, -px0:]
            px0 = 0
        if py0 < 0:
            submask = submask[-py0:]
            py0 = 0
        if px1 > mask_x:
            cutoff = px1 - mask_x
            submask = submask[:, :-cutoff]
            px1 = mask_x
        if py1 > mask_y:
            cutoff = py1 - mask_y
            submask = submask[:-cutoff]
            py1 = mask_y

        mask[py0:py1, px0:px1] = np.maximum(submask, mask[py0:py1, px0:px1])

        return mask

@dataclass
class FaceConfig:
    eyes: ShapeColor
    mouth: ShapeColor
    face: ShapeColor
    age: int  # ranges from 1 to 100

    __mask = None
    _face_colors = None
    _mouth_colors = None
    _eyes_colors = None

    def __post_init__(self):
        for obj in ('eyes', 'mouth', 'face'):
            if isinstance(getattr(self, obj), (list, tuple)):
                setattr(self, obj, ShapeColor(*getattr(self, obj)))
            elif isinstance(getattr(self, obj), dict):
                setattr(self, obj, ShapeColor(**getattr(self, obj)))
        self._face_colors = np.linspace(kHalfGrey, self.face.color, 100)  # Completely towards grey
        self._mouth_colors = np.linspace((kHalfGrey + self.mouth.color) / 2.0, self.mouth.color, 100)  # Closer towards grey
        self._eyes_colors = np.linspace(self.eyes.color, (kHalfGrey + self.eyes.color) / 2.0, 100)  # Closer towards grey

    def specific_hash(self):
        # Two faces are the same if hash of the features is the same and the age is the same
        return hash((self.general_hash, self.age))

    def general_hash(self):
        return hash((self.face, self.mouth, self.eyes))

    def get_mask(self):
        return self.__mask

    def get_color(self, feature_str):
        attr_name = f'_{feature_str}_colors'
        # color = getattr(self, attr_name)[self.age]
        color = getattr(self, attr_name)[self.age-1]
        return color

    def get_age_distance(self):
        # Gets relative eye distance based on age ranging from 0.1 to 0.9
        min_range = 0.2
        max_range = 0.8
        return ((max_range - min_range) * (self.age - 1) / 100 + min_range)

    def make_blank_mask(self, width, height):
        self.__mask = np.zeros((height, width))
        return self

    def make_face_mask(self, width, height):
        d = self.get_age_distance()
        # Background
        # self.__mask = np.zeros((height, width))
        self.make_blank_mask(width, height)
        # Face
        self.face.add_shape_mask(self.__mask)
        # Mouth
        offset = self.mouth.center_offset  # This is the center_line
        self.mouth.center_offset = offset[0], offset[1] * (1.0 - d)
        self.mouth.add_shape_mask(self.__mask)
        self.mouth.center_offset = offset  # Reset
        # Eyes
        offset = self.eyes.center_offset  # This is the center_line
        self.eyes.center_offset = offset[0] * 2 * d, offset[1]
        self.eyes.add_shape_mask(self.__mask)
        self.eyes.center_offset = -offset[0] * 2 * d, offset[1]
        self.eyes.add_shape_mask(self.__mask)
        self.eyes.center_offset = offset  # Reset

        return self

    def make_img(self, width, height, background_color=0.0, scale=1.0, offset=(0.0, 0.0), aspil=False):
        self.make_face_mask(width, height)

        mask = self.get_mask()
        mask = Image.fromarray(mask)
        # Translate
        if offset != (0.0, 0.0):
            offset = offset[0] * mask.width, offset[1] * mask.height
            mask = mask.rotate(angle=0, translate=offset)
        # Rescale
        if scale != 1.0:
            w, h = int(width * scale), int(height * scale)
            submask = mask.resize((w, h), resample=Image.Resampling.NEAREST)
            left = int((width - w) / 2)
            top = int((height - h) / 2)
            right = int((width + w) / 2)
            bottom = int((height + h) / 2)
            if scale > 1.0:  # center crop
                mask = submask.crop((-left, -top, right, bottom))
                # mask = mask.resize((w, h))
            else:
                mask.paste(0, box=(0, 0, *mask.size))
                mask.paste(submask, box=(left, top, right, bottom))

        mask = np.array(mask).astype(int)
        self.__mask = mask

        # Fill in the colors for the img
        # age_factor = self.get_age_distance()  # Intensity based on age
        # age_color = np.array([0.5, 0.5, 0.5])  # 2nd color for aging indication (pure 0.5 gray)
        img = np.ones((width, height, 3)) * background_color

        # Face
        for feature_str in ('face', 'mouth', 'eyes'):
            feature = getattr(self, feature_str)
            img[mask == feature.z] = self.get_color(feature_str)

        if aspil:
            img = Image.fromarray((img * 255).astype(np.uint8))

        return img


def sample_config(ranges, N=1, keys=None):
    if keys is None:
        keys = ranges.keys()
    result = dict()
    for key in keys:
        subelement = dict()
        for subkey, subvalues in ranges[key].items():
            subvalues = np.array(subvalues)
            if subvalues.ndim == 1:
                subelement[subkey] = np.random.choice(subvalues, size=N, replace=True)
            else:
                indices = np.arange(len(subvalues))
                indices = np.random.choice(indices, size=N, replace=True)
                subelement[subkey] = subvalues[indices]
        result[key] = subelement
    return result

def unroll_samples(samples, merge_x_y=True, defaults=None):
    # Get length of samples
    N = len(samples['eyes']['shape'])
    defaults = defaults or dict()
    result = [dict() for _ in range(N)]
    for key, value in samples.items():
        value_defaults = defaults.get(key, dict())
        for subkey, default_value in value_defaults.items():
            for idx in range(N):
                result[idx][key] = result[idx].get(key, dict())
                result[idx][key][subkey] = default_value
        for subkey, subvalues in value.items():
            for idx in range(N):
                result[idx][key] = result[idx].get(key, dict())
                result[idx][key][subkey] = subvalues[idx]
    # Merge any keys that end with (`_x`, `_y`)
    if merge_x_y:
        for element in result:
            for key, value in element.items():
                seen = set()
                for subkey in value.keys():
                    if subkey in seen:
                        continue
                    if subkey.endswith('_x'):
                        newkey = subkey[:-2]
                        ykey = newkey + '_y'
                        if ykey in value.keys():
                            seen.add(subkey)
                for xkey in seen:
                    newkey = xkey[:-2]
                    ykey = newkey + '_y'
                    value[newkey] = (value[xkey], value[ykey])
                    del value[xkey]
                    del value[ykey]
    return result


if __name__ == '__main__':

    # Possible values
    ranges = {
        'eyes': {
            'shape': ['oval', 'rect'],
            'shape_ratio': [0.5, 1.0, 2.0],
            'center_offset_x': np.linspace(-0.4, -0.1, 5),
            'center_offset_y': np.linspace(-0.4, -0.1, 5),
            'color': [kRed, kGreen, kBlue],
        },
        'mouth': {
            'shape': ['oval', 'rect'],
            'shape_ratio': [1.0, 2.0, 4.0],
            'center_offset_x': [0.0],
            'center_offset_y': [0.3, 0.5],
            'color': [kReddish],
        },
        'face': {
            'shape': ['oval', 'rect'],
            'shape_ratio': [0.75, 1.0, 1.5],
            'center_offset_x': [0.0],
            'center_offset_y': [0.0],
            'color': [kPink],
        },
        'background_color': np.array(plt.get_cmap('Pastel1').colors),
        'age': np.arange(1, 101),
    }

    # Make some people
    num_ids = 300
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

    images_per_id = 50

    # Age distribution
    final_value = 0.1
    c = np.log2(1.0 / final_value) / ranges['age'][-1]
    age_distribution = np.array([1.0 / 2**(c * age) for age in ranges['age']])
    age_distribution1 = age_distribution / age_distribution.sum()
    # age_distribution2 = np.exp(age_distribution) / np.exp(age_distribution).sum()
    ages = np.random.choice(ranges['age'], replace=True, size=(num_ids, images_per_id), p=age_distribution1)

    # plt.bar(ranges['age']-0.25, age_distribution1, width=0.5, label='SumNorm')
    # plt.bar(ranges['age']+0.25, age_distribution2, width=0.5, label='Softmax')
    # plt.legend()
    # plt.show()

    # Color distributions
    indices = np.arange(len(ranges['background_color']))
    indices = np.random.choice(indices, replace=True, size=(num_ids, images_per_id))
    backgrounds = ranges['background_color'][indices]

    print(ages.shape, backgrounds.shape, len(random_samples))

    N = num_ids * images_per_id
    H = 224
    W = 224
    C = len(ranges['background_color'][0])

    scale_min = 0.95
    scale_max = 1.05
    offset_min = -0.05
    offset_max = 0.05

    # all_images = np.zeros((N, H, W, C), dtype=float)
    all_person_ids = np.repeat(range(num_ids), images_per_id)
    all_ages = ages.flatten()

    datafolder_name = 'face_age_images'
    os.makedirs(datafolder_name, exist_ok=True)

    for person_id in tqdm.trange(num_ids):
        for img_id in range(images_per_id):
            img_idx = person_id * images_per_id + img_id
            face = FaceConfig(**random_samples[person_id], age=all_ages[img_idx])
            # print(face.age)
            img = face.make_img(H, W,
                                scale=np.random.uniform(scale_min, scale_max),
                                offset=tuple(np.random.uniform(offset_min, offset_max, size=2)),
                                background_color=backgrounds[person_id, img_id],
                                aspil=True)
            # all_images[img_idx] = img
            file_name = f'{img_idx}_{person_id}_{all_ages[img_idx]}.png'
            img.save(os.path.join(datafolder_name, file_name))


    # # Save all the images
    # with open('face_age_data.pkl', 'wb') as f:
    #     all_data = {
    #         'images': all_images,
    #         'ages': all_ages,
    #         'person_ids': all_person_ids,
    #     }
    #     pkl.dump(all_data, f)

    # # Show several faces
    # ROWS = 10
    # COLS = 10

    # fig, ax = plt.subplots(ROWS, COLS, sharex=True, sharey=True, figsize=(25, 25))

    # for person_id in range(ROWS):
    #     for img_id in range(COLS):
    #         img_idx = person_id * images_per_id + img_id
    #         ax[person_id, img_id].imshow(all_images[img_idx])
    #         ax[person_id, img_id].axis('off')
    #         ax[person_id, img_id].set_title(f'{person_id}, {all_ages[img_idx]} yo')
    #         # print(all_images[img_idx].shape, ages[person_id, img_id])
    # plt.tight_layout()
    # plt.show()
