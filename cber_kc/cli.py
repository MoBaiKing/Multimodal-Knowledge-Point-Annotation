"""Select frozen reranking parameters and evaluate CBER-KC features."""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from .io import read_json, save_tensor, sha256, write_json
from .rerank import ExemplarBank, Parameters, measure

METHOD = 'CBER-KC'
SCHEMA = 'cber-kc.freeze.1'


def identity(features):
    return {key: features[key] for key in ('dataset', 'seed', 'encoder_id')}


def load_features(path, split):
    features = torch.load(path, map_location='cpu', weights_only=True)
    if features.get('split') != split:
        raise ValueError(f'Expected a {split} feature pack at {path}')
    source = identity(features)
    if (not isinstance(source['dataset'], str) or not source['dataset']
            or type(source['seed']) is not int
            or not isinstance(source['encoder_id'], str) or not source['encoder_id']):
        raise ValueError(f'Invalid feature provenance in {path}')
    text = features['text']
    image = features['image']
    has_image = features['has_image']
    rows = features['row_ids']
    if (text.ndim != 2 or not all(text.shape) or image.ndim != 2 or not image.shape[1]
            or len(image) != len(text) or not text.is_floating_point() or not image.is_floating_point()
            or not text.isfinite().all() or not image.isfinite().all()
            or has_image.dtype != torch.bool or has_image.shape != (len(text),)
            or len(rows) != len(text) or any(not isinstance(row, str) for row in rows)
            or len(set(rows)) != len(rows)):
        raise ValueError(f'Malformed features or row IDs in {path}')
    if split != 'predict':
        labels = features['labels']
        if labels.dtype != torch.int64 or labels.shape != (len(text),) or (labels < 0).any():
            raise ValueError(f'Invalid class IDs in {path}')
    if split != 'train':
        scores = features['scores']
        if (scores.ndim != 2 or len(scores) != len(text) or scores.shape[1] < 2
                or not scores.is_floating_point() or not scores.isfinite().all()):
            raise ValueError(f'Invalid frozen classifier scores in {path}')
    return features


def check_query(train, query):
    classes = query['scores'].shape[1]
    if identity(train) != identity(query) or any(
        train[key].shape[1] != query[key].shape[1] for key in ('text', 'image')
    ):
        raise ValueError('Feature provenance or dimensions disagree')
    if int(train['labels'].max()) >= classes or 'labels' in query and int(query['labels'].max()) >= classes:
        raise ValueError('Class IDs must index the columns of the frozen classifier scores')


def verify_freeze(record, train, train_sha):
    if (record.get('schema') != SCHEMA or record.get('method') != METHOD
            or record.get('selected_on') != 'val_select' or record.get('train_sha256') != train_sha
            or record.get('source') != identity(train)):
        raise ValueError('The freeze record does not match the training feature pack')
    return Parameters(**record['parameters'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True, dest='command')
    for command in ('select', 'evaluate', 'predict'):
        entry = commands.add_parser(command)
        entry.add_argument('--train', type=Path, required=True)
        entry.add_argument('--query', type=Path, required=True)
        entry.add_argument('--frozen', type=Path, required=True)
        entry.add_argument('--device', default='cpu')
        if command != 'select':
            entry.add_argument('--output', type=Path, required=True)
        if command == 'evaluate':
            entry.add_argument('--split', choices=('val_select', 'val_calibration', 'test'), required=True)
    from .encoder.extract import add_extract_command
    add_extract_command(commands)
    args = parser.parse_args(argv)
    if args.command == 'extract':
        args.run(args)
        return

    train_sha = sha256(args.train)
    train = load_features(args.train, 'train')
    if args.command == 'select':
        validation_sha = sha256(args.query)
        validation = load_features(args.query, 'val_select')
        check_query(train, validation)
        if args.frozen.exists():
            previous = read_json(args.frozen)
            verify_freeze(previous, train, train_sha)
            if (previous['validation_sha256'] != validation_sha
                    or previous['num_classes'] != validation['scores'].shape[1]):
                raise ValueError('A different validation pack is already frozen at this path')
            print(args.frozen)
            return
        bank = ExemplarBank(train, args.device)
        params = bank.select(validation)
        metrics = measure(validation['labels'], bank.predict(validation, params), validation['scores'].shape[1])
        write_json(args.frozen, dict(schema=SCHEMA, method=METHOD, source=identity(train),
                                    train_sha256=train_sha, validation_sha256=validation_sha, selected_on='val_select',
                                    num_classes=validation['scores'].shape[1], parameters=asdict(params), metrics=metrics))
        print(args.frozen)
        return

    record = read_json(args.frozen)
    params = verify_freeze(record, train, train_sha)
    split = args.split if args.command == 'evaluate' else 'predict'
    if split == 'val_select' and sha256(args.query) != record['validation_sha256']:
        raise ValueError('The query differs from the frozen validation-selection pack')
    query = load_features(args.query, split)
    check_query(train, query)
    if query['scores'].shape[1] != record['num_classes']:
        raise ValueError('Classifier columns disagree with the frozen validation pack')
    predictions = ExemplarBank(train, args.device).predict(query, params)
    if args.command == 'evaluate':
        metrics = measure(query['labels'], predictions, query['scores'].shape[1])
        write_json(args.output, dict(method=METHOD, split=split, metrics=metrics,
                                     source=identity(query), frozen_sha256=sha256(args.frozen)))
    else:
        save_tensor(args.output, dict(row_ids=query['row_ids'], predictions=predictions))
    print(args.output)
