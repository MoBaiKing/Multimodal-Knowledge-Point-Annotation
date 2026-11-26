"""Regression tests for the standalone release."""

import tempfile
import unittest
from pathlib import Path

import torch

from cber_kc.cli import main
from cber_kc.io import read_json, save_tensor, stable_topk
from cber_kc.rerank import ExemplarBank, combine


class RerankingTests(unittest.TestCase):
    def test_class_quota_and_centered_adjustment(self):
        train = dict(text=torch.eye(4), image=torch.zeros(4, 4), has_image=torch.zeros(4, dtype=torch.bool),
                     labels=torch.tensor([0, 0, 0, 1]))
        bank = ExemplarBank(train, 'cpu')
        similarities = torch.tensor([[1., .75, .25, .5]])
        candidates = torch.tensor([[0, 1, 2]])
        scores = torch.tensor([[.25, .2, .1, 0.]])
        support = bank.supports(similarities, candidates, 2)
        torch.testing.assert_close(support, torch.tensor([[.875, .5, .25]]))
        updated = combine(scores, candidates, support, 1.0)
        torch.testing.assert_close(updated.gather(1, candidates).sum(), scores.gather(1, candidates).sum())
        torch.testing.assert_close(updated[:, -1], scores[:, -1])
        torch.testing.assert_close(combine(scores, candidates, support, 0), scores)
        self.assertEqual(stable_topk(torch.tensor([[1., 0., 1.]]), 2).tolist(), [[0, 2]])

    def test_freeze_then_evaluate(self):
        train = dict(dataset='demo', seed=42, encoder_id='unit-test', split='train',
                     row_ids=['a', 'b'], text=torch.eye(2), image=torch.zeros(2, 1),
                     has_image=torch.zeros(2, dtype=torch.bool), labels=torch.tensor([0, 1]))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            training, query, frozen, output = [str(directory / filename)
                                               for filename in ('train.pt', 'query.pt', 'frozen.json', 'test.json')]
            save_tensor(training, train)
            data = dict(train, split='val_select', scores=torch.zeros(2, 2))
            save_tensor(query, data)
            inputs = ['--train', training, '--query', query, '--frozen', frozen]
            main(['select', *inputs])
            selected = read_json(frozen)
            self.assertEqual(selected['method'], 'CBER-KC')
            self.assertEqual(selected['selected_on'], 'val_select')
            self.assertEqual(selected['metrics']['accuracy'], 1.0)
            save_tensor(query, dict(data, split='test'))
            with self.assertRaisesRegex(ValueError, 'val_select'):
                main(['select', *inputs])
            main(['evaluate', *inputs, '--split', 'test', '--output', output])
            self.assertEqual(read_json(output)['metrics']['accuracy'], 1.0)
            self.assertEqual(read_json(output)['split'], 'test')
            (directory / 'query.pt').unlink()
            save_tensor(training, dict(train, encoder_id='different'))
            with self.assertRaisesRegex(ValueError, 'freeze record'):
                main(['evaluate', *inputs, '--split', 'test', '--output', output])


if __name__ == '__main__':
    unittest.main()
