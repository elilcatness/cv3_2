from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from experiments.run_experiments import paired_crop
from src.degradation import DEGRADATIONS, degrade
from src.masks import expand_mask, make_mask, rectangle
from src.metrics import QualityMetrics, psnr
from src.pipeline import SRInpaintingPipeline
from src.utils import composite


@pytest.fixture
def hr():
    return Image.fromarray(np.random.default_rng(4).integers(0, 256, (128, 128, 3), dtype=np.uint8))


def test_degradations(hr):
    for kind in DEGRADATIONS:
        a, b = degrade(hr, kind, 7), degrade(hr, kind, 7)
        assert a.size == (32, 32)
        assert np.array_equal(a, b)


def test_masks_and_metrics(hr, tmp_path):
    mask = make_mask((32, 32), 'known50')
    assert mask.sum() == 512
    assert rectangle(expand_mask(mask)) == (64, 0, 128, 128)
    raw = np.asarray(mask, dtype=np.uint8)
    np.save(tmp_path / 'mask.npy', raw)
    assert np.array_equal(make_mask((32, 32), str(tmp_path / 'mask.npy')), mask)
    np.save(tmp_path / 'bad.npy', raw + 2)
    with pytest.raises(ValueError):
        make_mask((32, 32), str(tmp_path / 'bad.npy'))
    generated = Image.new('RGB', hr.size)
    result = composite(generated, hr, mask)
    known = ~expand_mask(mask)
    assert np.array_equal(np.asarray(result)[known], np.asarray(hr)[known])
    assert psnr(result, hr, ~known) == psnr(generated, hr, ~known)
    assert psnr(result, hr) > psnr(generated, hr)
    scores = QualityMetrics(perceptual=False)(hr, hr, make_mask((32, 32), 'all_zero'))
    assert np.isinf(scores['psnr']) and np.isnan(scores['psnr_mask'])


def test_paired_crop():
    data = np.arange(40 * 40, dtype=np.uint16).astype(np.uint8).reshape(40, 40)
    lr = Image.fromarray(data).convert('RGB')
    hr = lr.resize((160, 160), Image.Resampling.NEAREST)
    a, b = paired_crop(hr, lr, 32)
    assert np.array_equal(a, b.resize(a.size, Image.Resampling.NEAREST))


class FakeScheduler:
    def set_timesteps(self, steps, device=None):
        self.timesteps = torch.arange(steps, 0, -1, device=device)

    def add_noise(self, z, noise, timestep):
        return z + noise * timestep.item() / 10


class FakeVAE:
    dtype = torch.float32
    config = SimpleNamespace(latent_channels=4, scaling_factor=.08333)

    def encode(self, rgb):
        self.input = rgb.clone()
        pooled = torch.nn.functional.avg_pool2d(rgb, 4)
        z = torch.cat((pooled, pooled[:, :1]), dim=1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: z))


class FakePipe:
    unet = SimpleNamespace(device=torch.device('cpu'), dtype=torch.float32)

    def __init__(self):
        self.vae = FakeVAE()
        self.scheduler = FakeScheduler()
        self.calls = []

    def __call__(self, **kwargs):
        self.scheduler.set_timesteps(kwargs['num_inference_steps'])
        z = kwargs['latents'].clone()
        record = {'initial': z.clone(), 'lr': kwargs['image'], 'rng': kwargs['generator'].initial_seed(), 'states': []}
        for i, t in enumerate(self.scheduler.timesteps):
            z = z + z.mean() + .05  # toy denoiser whose output depends on context
            before = z.clone()
            if kwargs['callback']:
                kwargs['callback'](i, t, z)
            record['states'].append((before, z.clone()))
        self.calls.append(record)
        value = z[:, :3].clamp(-1, 1).add(1).mul(127.5).to(torch.uint8)
        result = Image.fromarray(value[0].permute(1, 2, 0).numpy()).resize((128, 128), Image.Resampling.NEAREST)
        return SimpleNamespace(images=[result])


def test_pipeline_masking(hr):
    fake = FakePipe()
    pipe = SRInpaintingPipeline(fake)
    lr = degrade(hr)
    baseline = pipe(lr, steps=3)
    all_one = pipe(lr, hr, make_mask(lr.size, 'all_one'), steps=3)
    assert np.array_equal(baseline, all_one)
    assert torch.equal(fake.calls[0]['initial'], fake.calls[1]['initial'])
    all_zero = pipe(lr, hr, make_mask(lr.size, 'all_zero'), steps=3)
    assert np.array_equal(all_zero, hr) and len(fake.calls) == 2
    mask = make_mask(lr.size)
    result = pipe(lr, hr, mask, steps=3)
    record = fake.calls[-1]
    assert record['lr'] is lr and record['rng'] == fake.calls[0]['rng']
    assert (fake.vae.input[..., 64:] == -1).all()
    assert torch.equal(record['initial'][..., 16:], fake.calls[0]['initial'][..., 16:])
    assert not torch.equal(record['initial'][..., :16], fake.calls[0]['initial'][..., :16])
    for before, after in record['states']:
        assert torch.equal(before[..., 16:], after[..., 16:])
        assert not torch.equal(before[..., :16], after[..., :16])
    known = ~expand_mask(mask)
    assert np.array_equal(np.asarray(result)[known], np.asarray(hr)[known])
    # Change unknown ground truth only: the generated output must not change.
    changed = np.array(hr)
    changed[~known] = 255 - changed[~known]
    again = pipe(lr, Image.fromarray(changed), mask, steps=3)
    assert np.array_equal(again, result)


def test_batch_smoke(hr, tmp_path, monkeypatch):
    import sys
    from experiments import run_experiments as runner

    source, output = tmp_path / 'hr', tmp_path / 'out'
    source.mkdir()
    hr.save(source / 'sample.png')
    monkeypatch.setattr(runner.SRInpaintingPipeline, 'from_pretrained', lambda *a: SRInpaintingPipeline(FakePipe()))
    monkeypatch.setattr(sys, 'argv', ['run_experiments', '--hr-dir', str(source), '--output', str(output),
                                    '--size', '32', '--steps', '2', '--degradations', 'bicubic',
                                    '--masks', 'all_one', 'all_zero', 'known50', '--psnr-only'])
    runner.main()
    assert (output / 'metrics.csv').read_text().count('\n') == 10
    assert (output / 'summary.csv').is_file()
    assert (output / 'plots/bicubic_psnr_mask.png').is_file()
    assert (output / 'images/sample/bicubic/42/known50/comparison.png').is_file()
