"""Frozen question, concept and image encoders used by CBER-KC."""

from functools import lru_cache

import torch
from torch import nn

from ..io import stable_topk
from ..rerank import CANDIDATES
from .backbone import QwenBackbone
from .heads import ConceptHead, PairEvidenceHead


class FrozenEncoder(nn.Module):
    def __init__(self, config, checkpoint):
        super().__init__()
        self.backbone = QwenBackbone(config).cuda()
        self.register_buffer('semantics', checkpoint['semantics'].float().cuda())
        classes, hidden = self.semantics.shape
        self.concept = ConceptHead(self.semantics).cuda()
        self.evidence = PairEvidenceHead(hidden, classes, max_residual=3.0, gate_bias=0.0).cuda()
        self.evidence_image = PairEvidenceHead(hidden, classes, max_residual=3.0, gate_bias=0.0).cuda()
        trainables = {name: param for name, param in self.named_parameters() if param.requires_grad}
        if trainables.keys() != checkpoint['trainables'].keys():
            raise ValueError('Checkpoint parameter names disagree with the encoder architecture')
        with torch.no_grad():
            for name, param in trainables.items():
                param.copy_(checkpoint['trainables'][name])
        self.requires_grad_(False)
        self.eval()

    def text(self, texts):
        hidden, tokens, mask = self.backbone.encode_question_states(texts)
        with torch.autocast('cuda', enabled=False):
            scores = self.concept(hidden.float())
        return hidden, scores, tokens, mask

    def mixed_scores(self, text, images, reference, weight, scale):
        hidden, student, states, question_mask = text
        scores = ((1. - weight) * reference.float().softmax(-1) +
                  weight * student.float().softmax(-1)).clamp_min(1e-10).log()
        if scale == 0:
            return scores

        max_tokens = max(1, max(len(tokens) for tokens in images))
        tokens = torch.zeros(len(images), max_tokens, self.backbone.hidden_size, device='cuda', dtype=torch.bfloat16)
        mask = torch.zeros(len(images), max_tokens, device='cuda', dtype=torch.bool)
        for index, image in enumerate(images):
            if len(image):
                tokens[index, :len(image)] = image.to('cuda')
                mask[index, :len(image)] = True
        candidates = stable_topk(scores, min(CANDIDATES, scores.shape[1]))
        with torch.autocast('cuda', enabled=False):
            question, _ = self.evidence(hidden.float(), self.semantics, states, question_mask, candidates)
            visual = torch.zeros_like(question)
            if mask.any():
                visual, _ = self.evidence_image(hidden.float(), self.semantics, tokens, mask, candidates)
        return scores + scale * (question + visual)

    @torch.no_grad()
    def extract(self, rows, batch_size, max_images, reference=None, weight=None, scale=None):
        hidden_size = self.backbone.hidden_size

        @lru_cache(maxsize=96)
        def image(path):
            return self.backbone.extract_image(path)[0].to(torch.bfloat16)

        text, image_mean, has_image, scores = [], [], [], []
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                encoding = self.text([row['text'] for row in batch])
                tokens = []
                for row in batch:
                    paths = row['images'][:max_images]
                    tokens.append(torch.cat([image(path) for path in paths]) if paths
                                  else torch.empty((0, hidden_size), dtype=torch.bfloat16))
                if reference is not None:
                    scores.append(self.mixed_scores(encoding, tokens, reference[start:start + len(batch)].cuda(),
                                                    weight, scale).float().cpu())
            text.append(encoding[0].float().cpu())
            has_image.extend(bool(len(value)) for value in tokens)
            image_mean.extend(value.float().mean(0) if len(value) else torch.zeros(hidden_size) for value in tokens)
        features = dict(text=torch.cat(text), image=torch.stack(image_mean), has_image=torch.tensor(has_image))
        if scores:
            features['scores'] = torch.cat(scores)
        return features
