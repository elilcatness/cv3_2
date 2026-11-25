import math

import numpy as np
import torch

from .masks import expand_mask, rectangle
from .utils import image_tensor

METRICS = ('psnr', 'lpips', 'clip_iqa', 'psnr_mask', 'lpips_mask', 'clip_iqa_mask')


def psnr(image, target, mask=None):
    if image.size != target.size:
        raise ValueError('Metric image sizes must match.')
    error = (np.asarray(image, dtype=np.float64) - np.asarray(target, dtype=np.float64)) ** 2
    if mask is not None:
        if not mask.any():
            return math.nan
        error = error[mask]
    mse = error.mean()
    return math.inf if mse == 0 else float(10 * np.log10(255**2 / mse))


class QualityMetrics:
    def __init__(self, device='cpu', perceptual=True):
        self.device = device
        self.lpips = self.clip = None
        if perceptual:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            from torchmetrics.multimodal import CLIPImageQualityAssessment

            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device).eval()
            self.clip = CLIPImageQualityAssessment(model_name_or_path='clip_iqa', data_range=1.0).to(device).eval()

    @torch.no_grad()
    def _perceptual(self, image, target):
        if self.lpips is None or min(image.size) < 64:
            return math.nan, math.nan
        x, y = (image_tensor(im).to(self.device) for im in (image, target))
        lpips = float(self.lpips(x, y).item())
        clip = float(self.clip(x).item())
        self.lpips.reset()
        self.clip.reset()
        return lpips, clip

    def __call__(self, image, target, mask):
        mask_hr = expand_mask(mask.astype(bool))
        full = psnr(image, target)
        lpips, clip = self._perceptual(image, target)
        masked = psnr(image, target, mask_hr)
        box = rectangle(mask_hr)
        if mask_hr.all():
            mlpips, mclip = lpips, clip
        elif box is not None:
            mlpips, mclip = self._perceptual(image.crop(box), target.crop(box))
        else:
            mlpips = mclip = math.nan  # arbitrary masks: no valid rectangular crop
        return dict(zip(METRICS, (full, lpips, clip, masked, mlpips, mclip)))
