from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .masks import expand_mask


def read_rgb(path):
    with Image.open(path) as image:
        return image.convert('RGB')


def center_crop(image, size):
    w, h = image.size
    cw, ch = size
    if w < cw or h < ch:
        raise ValueError(f'Image {image.size} smaller than crop {size}.')
    x, y = (w - cw) // 2, (h - ch) // 2
    return image.crop((x, y, x + cw, y + ch))


def validate_lr(lr):
    if any(n < 32 or n % 8 for n in lr.size):
        raise ValueError('LR dimensions must be >=32 and multiples of 8.')


def validate_pair(lr, hr, mask):
    validate_lr(lr)
    if hr.size != tuple(4 * n for n in lr.size):
        raise ValueError('HR must have exactly 4x the LR width and height.')
    if mask.shape != (lr.height, lr.width) or not np.isin(mask, [0, 1]).all():
        raise ValueError('Mask must be binary with LR resolution.')


def composite(generated, hr, mask):
    if generated.size != hr.size or expand_mask(mask).shape != (hr.height, hr.width):
        raise ValueError('Generated image, HR and mask sizes do not match.')
    pixels = np.where(expand_mask(mask)[..., None], np.asarray(generated), np.asarray(hr))
    return Image.fromarray(pixels.astype(np.uint8))


def image_tensor(image):
    return torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)[None]


def resolve_device(device='auto'):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available.')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is not available.')
    return torch.device(device)


def save_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
