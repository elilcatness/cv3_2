"""SR-inpainting using the original Diffusers x4 upscaler denoising loop."""
import numpy as np
import torch

from .masks import expand_mask
from .utils import composite, image_tensor, resolve_device, validate_lr, validate_pair

MODEL_ID = 'stabilityai/stable-diffusion-x4-upscaler'


def generator(seed):
    return torch.Generator(device='cpu').manual_seed(seed)


def randn(shape, seed, device, dtype):
    return torch.randn(shape, generator=generator(seed), dtype=torch.float32).to(device=device, dtype=dtype)


class MaskedLatents:
    """Project known HR latents onto the current DDIM noise level."""

    def __init__(self, scheduler, z_hr, mask, noise):
        self.scheduler, self.z_hr, self.mask, self.noise = scheduler, z_hr, mask, noise

    def blend(self, latents, timestep=None):
        known = self.z_hr if timestep is None else self.scheduler.add_noise(
            self.z_hr, self.noise, timestep.reshape(1)
        )
        return torch.where(self.mask, latents, known)

    def __call__(self, step, timestep, latents):
        times = self.scheduler.timesteps
        next_t = times[step + 1] if step + 1 < len(times) else None
        # Legacy callback receives the live tensor; mutate in-place, not return.
        latents.copy_(self.blend(latents, next_t))


class SRInpaintingPipeline:
    """Thin wrapper: pretrained U-Net, text encoder and LR condition unchanged."""

    def __init__(self, pipe):
        self.pipe = pipe

    @classmethod
    def from_pretrained(cls, device='auto', precision='auto'):
        from diffusers import DDIMScheduler, StableDiffusionUpscalePipeline  # type: ignore

        device = resolve_device(device)
        dtype = torch.float32 if precision == 'fp32' or (precision == 'auto' and device.type == 'cpu') else torch.float16
        if device.type == 'cpu' and dtype == torch.float16:
            raise ValueError('Use fp32 on CPU.')
        pipe = StableDiffusionUpscalePipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
        # Preserve beta schedule and v_prediction from the checkpoint.
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to(device)
        # VAE fp16 can overflow; keep both encoder and decoder in fp32.
        pipe.vae.to(dtype=torch.float32)
        return cls(pipe)

    @torch.no_grad()
    def __call__(self, lr, hr=None, mask=None, *, prompt='', steps=50,
                 noise_level=20, guidance_scale=1.0, seed=42):
        validate_lr(lr)
        if (hr is None) != (mask is None):
            raise ValueError('Provide both HR and mask, or neither for baseline.')
        if not 1 <= steps <= 999 or not 0 <= noise_level <= 350:
            raise ValueError('Use 1–999 DDIM steps and noise_level 0–350.')
        if not 0 <= seed < 2**32 or guidance_scale < 1:
            raise ValueError('Use seed in [0, 2^32) and guidance_scale >= 1.')
        if hr is not None:
            validate_pair(lr, hr, mask)
            mask = np.asarray(mask, dtype=bool)
            if not mask.any():
                return hr.copy()  # no unknown pixels; exact boundary case

        pipe = self.pipe
        device, dtype = pipe.unet.device, pipe.unet.dtype
        shape = (1, pipe.vae.config.latent_channels, lr.height, lr.width)
        initial = randn(shape, seed, device, dtype)
        callback = None
        if mask is not None and not mask.all():
            # Mask BEFORE VAE encoding: unknown HR pixels cannot leak through it.
            rgb = image_tensor(hr)
            unknown = torch.from_numpy(expand_mask(mask))[None, None]
            rgb = rgb.masked_fill(unknown, 0).mul(2).sub(1)
            z_hr = pipe.vae.encode(rgb.to(device=device, dtype=pipe.vae.dtype)).latent_dist.mode()
            z_hr = (z_hr * pipe.vae.config.scaling_factor).to(dtype=dtype)
            if z_hr.shape != initial.shape:
                raise RuntimeError(f'Unexpected VAE shape: {z_hr.shape}, expected {initial.shape}.')
            latent_mask = torch.from_numpy(mask)[None, None].to(device)
            noise_hr = randn(shape, seed + 2, device, dtype)
            pipe.scheduler.set_timesteps(steps, device=device)
            callback = MaskedLatents(pipe.scheduler, z_hr, latent_mask, noise_hr)
            # DDIM init_noise_sigma = 1; pipeline scales supplied latents by it.
            initial = callback.blend(initial, pipe.scheduler.timesteps[0])

        result = pipe(prompt=prompt, image=lr, num_inference_steps=steps,
                      noise_level=noise_level, guidance_scale=guidance_scale,
                      eta=0.0, generator=generator(seed + 1), latents=initial,
                      callback=callback, callback_steps=1).images[0]
        return composite(result, hr, mask) if callback is not None else result
