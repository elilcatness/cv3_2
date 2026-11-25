"""Reproducible synthetic degradations; no ESRGAN model inference."""
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

DEGRADATIONS = ('bicubic', 'unknown', 'realesrgan_like')


def _noise(image, rng, sigma):
    arr = np.asarray(image, dtype=np.float32)
    arr += rng.normal(0, sigma, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))


def _jpeg(image, quality):
    with BytesIO() as buffer:
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert('RGB')


def degrade(hr, kind='bicubic', seed=42):
    if any(n % 4 for n in hr.size):
        raise ValueError('HR dimensions must be divisible by 4.')
    target = (hr.width // 4, hr.height // 4)
    rng = np.random.default_rng(seed)
    bicubic = Image.Resampling.BICUBIC
    if kind == 'bicubic':
        return hr.resize(target, bicubic)
    if kind == 'unknown':
        # Synthetic unknown-like degradation, not the official DIV2K split.
        image = hr.filter(ImageFilter.GaussianBlur(float(rng.uniform(1, 2.5))))
        resample = (Image.Resampling.BILINEAR, bicubic, Image.Resampling.BOX)[int(rng.integers(3))]
        return image.resize(target, resample)
    if kind == 'realesrgan_like':
        # Two-stage blur / resize / noise / JPEG inspired by Real-ESRGAN.
        image = hr.filter(ImageFilter.GaussianBlur(float(rng.uniform(2, 4))))
        image = image.resize((2 * target[0], 2 * target[1]), bicubic)
        image = _jpeg(_noise(image, rng, 10), 45)
        image = image.filter(ImageFilter.GaussianBlur(1.2)).resize(target, bicubic)
        return _jpeg(_noise(image, rng, 12), 30)
    raise ValueError(f'Unknown degradation: {kind}')
