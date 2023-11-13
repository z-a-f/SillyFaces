from dataclasses import dataclass
from enum import Enum
import numpy as np

from .shape_utils import make_rect, make_oval


class Color(Enum):
    pink = [1.0, 0.75, 0.79]
    reddish = [0.5, 0.2, 0.2]
    red = [1.0, 0.0, 0.0]
    green = [0.0, 1.0, 0.0]
    blue = [0.0, 0.0, 1.0]
    half_grey = [0.5, 0.5, 0.5]

    def __len__(self):
        return len(self.value)
    
    def __getitem__(self, idx):
        return np.array(self.value[idx])


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