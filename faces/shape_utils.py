
import numpy as np
from PIL import Image

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