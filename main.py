import argparse
from pathlib import Path

from src.degradation import DEGRADATIONS, degrade
from src.masks import make_mask
from src.pipeline import SRInpaintingPipeline
from src.utils import composite, read_rgb, save_image, validate_pair


def model_args(parser):
    parser.add_argument(
        '--device', choices=('auto', 'mps', 'cuda', 'cpu'), default='auto'
    )
    parser.add_argument('--precision', choices=('auto', 'fp16', 'fp32'), default='auto')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--noise-level', type=int, default=20)
    parser.add_argument('--guidance-scale', type=float, default=1.0)
    parser.add_argument('--prompt', default='')


def settings(args, seed):
    return {
        'prompt': args.prompt,
        'steps': args.steps,
        'noise_level': args.noise_level,
        'guidance_scale': args.guidance_scale,
        'seed': seed,
    }


def main(default_method='inpaint'):
    parser = argparse.ArgumentParser(
        description='Stable Diffusion x4 SR / SR-inpainting'
    )
    parser.add_argument('--lr', type=Path)
    parser.add_argument('--hr', type=Path)
    parser.add_argument(
        '--mask', default='known50', help='Preset, PNG or NPY (LR size).'
    )
    parser.add_argument(
        '--method', choices=('baseline', 'composite', 'inpaint'), default=default_method
    )
    parser.add_argument('--degradation', choices=DEGRADATIONS, default='bicubic')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--output', type=Path, default=Path('results/images/result.png')
    )
    model_args(parser)
    args = parser.parse_args()
    if args.hr is None and (args.lr is None or args.method != 'baseline'):
        parser.error('Provide HR for inpainting/composite, or LR for baseline.')
    hr = read_rgb(args.hr) if args.hr else None
    lr = read_rgb(args.lr) if args.lr else degrade(hr, args.degradation, args.seed)
    mask = make_mask(lr.size, args.mask) if args.method != 'baseline' else None
    if mask is not None:
        validate_pair(lr, hr, mask)
    if mask is not None and not mask.any():
        result = hr.copy() if hr is not None else None
    else:
        pipe = SRInpaintingPipeline.from_pretrained(args.device, args.precision)
        kwargs = settings(args, args.seed)
        if args.method == 'inpaint':
            result = pipe(lr, hr, mask, **kwargs)
        else:
            result = pipe(lr, **kwargs)
            if args.method == 'composite':
                result = composite(result, hr, mask)
    save_image(result, args.output)
    print(args.output)


if __name__ == '__main__':
    main()
