"""Generate small, synthetic feature packs for a local CBER-KC smoke test."""

import argparse
from pathlib import Path

import torch

from cber_kc.io import save_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/demo'))
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(42)
    classes, dimensions = 6, 12
    text_prototypes = torch.randn(classes, dimensions, generator=generator)
    image_prototypes = torch.randn(classes, dimensions, generator=generator)
    for split, examples_per_class in (('train', 9), ('val_select', 4), ('val_calibration', 4), ('test', 4)):
        labels = torch.arange(classes).repeat_interleave(examples_per_class)
        n = len(labels)
        text = text_prototypes[labels] + .4 * torch.randn(n, dimensions, generator=generator)
        image = image_prototypes[labels] + .4 * torch.randn(n, dimensions, generator=generator)
        has_image = torch.rand(n, generator=generator) > .3
        pack = dict(dataset='synthetic', seed=42, encoder_id='synthetic-demo', split=split,
                    row_ids=[f'{split}-{row}' for row in range(n)], labels=labels,
                    text=text, image=image * has_image[:, None], has_image=has_image)
        if split != 'train':
            pack['scores'] = torch.randn(n, classes, generator=generator).scatter_add_(
                1, labels[:, None], torch.ones(n, 1)
            )
        save_tensor(args.output / f'{split}.pt', pack)


if __name__ == '__main__':
    main()
