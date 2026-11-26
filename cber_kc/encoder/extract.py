"""Convert a question manifest and a frozen checkpoint into a feature pack."""

import json
from pathlib import Path

import torch

from ..io import fingerprint, read_json, save_tensor, sha256

DEFAULTS = dict(text_tokens=384, max_pixels=100352, min_pixels=3136, max_images=3, lora_r=16, lora_alpha=32,
                lora_dropout=.05, padding_side='right')
SPLITS = ('train', 'val_select', 'val_calibration', 'test', 'predict')


def add_extract_command(commands):
    entry = commands.add_parser('extract', help='Extract features with the frozen Qwen encoder (CUDA)')
    entry.add_argument('--manifest', type=Path, required=True)
    entry.add_argument('--snapshot', type=Path, required=True)
    entry.add_argument('--checkpoint', type=Path, required=True)
    entry.add_argument('--config', type=Path)
    entry.add_argument('--dataset', required=True)
    entry.add_argument('--seed', required=True, type=int)
    entry.add_argument('--split', choices=SPLITS, required=True)
    entry.add_argument('--output', required=True, type=Path)
    entry.add_argument('--batch-size', type=int)
    entry.add_argument('--frozen', type=Path, help='CBER-KC freeze record required for test extraction')
    scores = entry.add_mutually_exclusive_group()
    scores.add_argument('--scores', type=Path, help='Saved frozen classifier scores aligned by row ID')
    scores.add_argument('--reference', type=Path, help='MacBERT reference logits aligned by row ID')
    entry.add_argument('--mix-weight', type=float)
    entry.add_argument('--evidence-scale', type=float)
    entry.set_defaults(run=extract)


def read_rows(manifest, split):
    with manifest.open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError('Expected a nonempty question manifest')
    row_ids = [row['row_id'] for row in rows]
    if any(not isinstance(row, str) for row in row_ids) or len(row_ids) != len(set(row_ids)):
        raise ValueError('Question IDs must be unique strings')
    for row in rows:
        if not isinstance(row['text'], str):
            raise ValueError('Question text must be a string')
        if split != 'predict' and (type(row['label']) is not int or row['label'] < 0):
            raise ValueError('Labels must be nonnegative integer class IDs')
        images = row.get('images', [])
        if not isinstance(images, list) or any(not isinstance(path, str) for path in images):
            raise ValueError('Images must be an array of paths')
        row['images'] = [manifest.parent / path for path in images]
    return rows, row_ids


def extract(args):
    from ..cli import METHOD, SCHEMA

    if args.split == 'train':
        if args.scores or args.reference:
            raise ValueError('The training exemplar bank only needs labels and embeddings')
    elif not (args.scores or args.reference):
        raise ValueError('Supply frozen classifier scores or aligned reference logits')
    if args.reference and (args.mix_weight is None or not 0 <= args.mix_weight <= 1
                           or args.evidence_scale is None or not 0 <= args.evidence_scale <= 1):
        raise ValueError('Reference logits require validation-selected weights between zero and one')
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError('Batch size must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('Feature extraction requires CUDA')

    overrides = read_json(args.config) if args.config else {}
    if overrides.keys() - DEFAULTS.keys():
        raise ValueError('Unknown encoder configuration setting')
    config = dict(DEFAULTS, **overrides)
    snapshot = args.snapshot.resolve(strict=True)
    tensors = sorted(snapshot.glob('*.safetensors'))
    if not tensors:
        raise FileNotFoundError(f'Expected a local Qwen safetensors snapshot at {snapshot}')
    files = sorted({*tensors, *snapshot.glob('*.json'), *snapshot.glob('*.txt'), *snapshot.glob('*.jinja')})
    checkpoint_sha = sha256(args.checkpoint)
    source = dict(dataset=args.dataset, seed=args.seed, encoder_id=fingerprint(dict(
        checkpoint=checkpoint_sha, snapshot={p.name: sha256(p) for p in files}, settings=config)))
    if args.split == 'test':
        if args.frozen is None:
            raise ValueError('Test extraction requires a matching CBER-KC freeze record')
        record = read_json(args.frozen)
        if (record.get('schema') != SCHEMA or record.get('method') != METHOD
                or record.get('selected_on') != 'val_select' or record.get('source') != source):
            raise ValueError('Test extraction must use the validation-frozen encoder')

    rows, row_ids = read_rows(args.manifest, args.split)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    classes = len(checkpoint['semantics'])
    features = dict(source, split=args.split, row_ids=row_ids)
    if args.split != 'predict':
        features['labels'] = torch.tensor([row['label'] for row in rows])
        if int(features['labels'].max()) >= classes:
            raise ValueError('A label is outside the frozen class vocabulary')
    reference = None
    if args.scores or args.reference:
        payload = torch.load(args.scores or args.reference, map_location='cpu', weights_only=True)
        logits = payload['logits']
        if (payload['row_ids'] != row_ids or logits.shape != (len(rows), classes)
                or not logits.is_floating_point() or not logits.isfinite().all()):
            raise ValueError('Input logits must be finite and aligned with question IDs and checkpoint classes')
        if args.reference:
            reference = logits
        else:
            features['scores'] = logits.float()

    from .model import FrozenEncoder

    model = FrozenEncoder(dict(config, model_path=str(snapshot), gradient_checkpointing=False), checkpoint)
    batch = args.batch_size or (16 if args.split == 'train' else 4)
    features.update(model.extract(rows, batch, config['max_images'], reference, args.mix_weight, args.evidence_scale))
    save_tensor(args.output, features)
    print(args.output)
