from dataclasses import dataclass
import numpy as np
from PIL import Image

from .shape_color_config import ShapeColor
from .shape_color_config import Color

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
        self._face_colors = np.linspace(Color.half_grey, self.face.color, 100)  # Completely towards grey
        self._mouth_colors = np.linspace((Color.half_grey + self.mouth.color) / 2.0, self.mouth.color, 100)  # Closer towards grey
        self._eyes_colors = np.linspace(self.eyes.color, (Color.half_grey + self.eyes.color) / 2.0, 100)  # Closer towards grey

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

        img = np.ones((width, height, 3)) * background_color

        # Face
        for feature_str in ('face', 'mouth', 'eyes'):
            feature = getattr(self, feature_str)
            img[mask == feature.z] = self.get_color(feature_str)

        if aspil:
            img = Image.fromarray((img * 255).astype(np.uint8))

        return img