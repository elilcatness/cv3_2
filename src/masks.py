from pathlib import Path

import numpy as np
from PIL import Image

PRESETS = ('all_one', 'all_zero', 'known25', 'known50', 'known75', 'center')


def make_mask(size, spec='known50'):
    """size=(width, height); True = generate, False = preserve HR."""
    w, h = size
    if spec == 'all_zero':
        return np.zeros((h, w), dtype=bool)
    mask = np.ones((h, w), dtype=bool)
    if spec == 'all_one':
        return mask
    if spec in ('known25', 'known50', 'known75'):
        mask[:, :round(w * int(spec[5:]) / 100)] = False
    elif spec == 'center':
        mask[:] = False
        mask[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
    else:
        path = Path(spec)
        if path.suffix.lower() == '.npy':
            raw = np.load(path, allow_pickle=False)
        else:
            with Image.open(path) as image:
                raw = np.asarray(image.convert('L'))
        if raw.shape != (h, w) or not np.isin(raw, [0, 1, 255]).all():
            raise ValueError('Mask must be binary and have LR resolution.')
        mask = raw != 0
    return mask


def expand_mask(mask):
    return mask.repeat(4, axis=0).repeat(4, axis=1)


def mask_image(mask):
    return Image.fromarray(mask.astype(np.uint8) * 255)


def rectangle(mask):
    """Bounding rectangle only when every pixel inside it is selected."""
    y, x = np.where(mask)
    if not x.size:
        return None
    box = (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)
    return box if (box[2] - box[0]) * (box[3] - box[1]) == x.size else None
