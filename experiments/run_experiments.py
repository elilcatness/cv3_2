import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from PIL import Image

from main import model_args, settings
from src.degradation import DEGRADATIONS, degrade
from src.masks import PRESETS, make_mask, mask_image
from src.metrics import METRICS, QualityMetrics
from src.pipeline import SRInpaintingPipeline
from src.utils import center_crop, composite, read_rgb, save_image, validate_pair


def paired_crop(hr, lr, size):
    # DIV2K-style pairs: HR modcrop at top-left, then aligned LR/HR crop.
    if any(not 0 <= a - 4 * b < 4 for a, b in zip(hr.size, lr.size)):
        raise ValueError(f'Not an aligned x4 pair: HR {hr.size}, LR {lr.size}')
    lr_crop = center_crop(lr, (size, size))
    x, y = (lr.width - size) // 2, (lr.height - size) // 2
    return hr.crop((4*x, 4*y, 4*(x+size), 4*(y+size))), lr_crop


def get_pair(path, kind, paired, size, degradation_seed):
    hr = read_rgb(path)
    if kind not in paired:
        hr = center_crop(hr, (4*size, 4*size))
        return hr, degrade(hr, kind, degradation_seed)
    directory = paired[kind]
    candidates = (directory / f'{path.stem}x4.png', directory / f'{path.stem}.png', directory / path.name)
    match = next((p for p in candidates if p.is_file()), None)
    if match is None:
        raise FileNotFoundError(f'No paired LR for {path.name} in {directory}')
    return paired_crop(hr, read_rgb(match), size)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, output):
    groups = defaultdict(list)
    keys = ('degradation', 'mask', 'method')
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    summary = []
    for group, values in groups.items():
        record = dict(zip(keys, group), known_fraction=mean(v['known_fraction'] for v in values), n=len(values))
        for metric in METRICS:
            valid = [v[metric] for v in values if not math.isnan(v[metric])]
            record[metric] = mean(valid) if valid else math.nan
        summary.append(record)
    write_csv(output / 'summary.csv', summary)
    plot_summary(summary, output / 'plots')


def plot_summary(rows, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    for kind in sorted({r['degradation'] for r in rows}):
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(6, 4))
            for method in ('baseline', 'composite', 'inpaint'):
                points = sorted((r['known_fraction'] * 100, r[metric]) for r in rows
                                if r['degradation'] == kind and r['method'] == method
                                and r['mask'] != 'center' and math.isfinite(r[metric]))
                if points:
                    ax.plot(*zip(*points), marker='o', label=method)
            if ax.lines:
                ax.set(xlabel='Known HR (%)', ylabel=metric, title=f'{kind}: {metric}')
                ax.legend()
                ax.grid(alpha=.2)
                fig.tight_layout()
                fig.savefig(directory / f'{kind}_{metric}.png', dpi=140)
            plt.close(fig)


def save_comparison(path, items):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, (name, image) in zip(axes.flat, items):
        ax.imshow(image, cmap='gray' if name == 'Mask' else None, vmin=0, vmax=255)
        ax.set_title(name)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Paired, reproducible SR-inpainting benchmark')
    parser.add_argument('--hr-dir', type=Path, default=Path('data/HR'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--size', type=int, default=128, help='Square LR crop size.')
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--degradations', nargs='+', default=list(DEGRADATIONS))
    parser.add_argument('--paired', action='append', default=[], metavar='NAME=DIR', help='Use existing aligned x4 LR instead of synthetic degradation.')
    parser.add_argument('--masks', nargs='+', choices=PRESETS, default=list(PRESETS[:-1]))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--degradation-seed', type=int, default=42)
    parser.add_argument('--metric-device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    parser.add_argument('--psnr-only', action='store_true')
    model_args(parser)
    args = parser.parse_args()
    if args.size < 32 or args.size % 8 or args.limit < 1:
        parser.error('Use a positive limit and LR size >=32 divisible by 8.')
    try:
        paired = {k: Path(v) for k, v in (entry.split('=', 1) for entry in args.paired)}
    except ValueError:
        parser.error('--paired expects NAME=DIR.')
    if any(k not in DEGRADATIONS and k not in paired for k in args.degradations):
        parser.error('Unknown degradation. Use built-ins or --paired NAME=DIR.')
    paths = sorted(p for p in args.hr_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'))[:args.limit] if args.hr_dir.is_dir() else []
    if not paths:
        parser.error(f'No HR images in {args.hr_dir}.')
    if len({p.stem for p in paths}) != len(paths):
        parser.error('HR filenames must have unique stems.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'config.json').write_text(json.dumps(vars(args), default=str, indent=2), encoding='utf-8')
    pipe = SRInpaintingPipeline.from_pretrained(args.device, args.precision)
    metrics = QualityMetrics(args.metric_device, perceptual=not args.psnr_only)
    rows = []
    for path in paths:
        for kind in args.degradations:
            hr, lr = get_pair(path, kind, paired, args.size, args.degradation_seed)
            for seed in args.seeds:
                root = args.output / 'images' / path.stem / kind / str(seed)
                save_image(hr, root / 'hr.png')
                save_image(lr, root / 'lr.png')
                kwargs = settings(args, seed)
                baseline = pipe(lr, **kwargs)
                save_image(baseline, root / 'baseline.png')
                for name in args.masks:
                    mask = make_mask(lr.size, name)
                    validate_pair(lr, hr, mask)
                    directory = root / name
                    save_image(mask_image(mask), directory / 'mask.png')
                    post = composite(baseline, hr, mask)
                    restored = baseline if mask.all() else hr.copy() if not mask.any() else pipe(lr, hr, mask, **kwargs)
                    for method, image in (('baseline', baseline), ('composite', post), ('inpaint', restored)):
                        if method != 'baseline':
                            save_image(image, directory / f'{method}.png')
                        row = dict(image=path.stem, degradation=kind, seed=seed, mask=name,
                                   known_fraction=float(1-mask.mean()), method=method)
                        row.update(metrics(image, hr, mask))
                        rows.append(row)
                    save_comparison(directory / 'comparison.png', [
                        ('HR', hr), ('LR', lr.resize(hr.size, Image.Resampling.NEAREST)),
                        ('Mask', mask_image(mask)), ('Baseline', baseline),
                        ('Composite', post), ('Inpainting', restored)])
                    write_csv(args.output / 'metrics.csv', rows)
                    print(f'{path.stem} / {kind} / seed={seed} / {name}', flush=True)
    summarize(rows, args.output)


if __name__ == '__main__':
    main()
