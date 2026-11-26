"""Class-balanced exemplar scores over frozen classifier candidates."""

from dataclasses import dataclass
from itertools import product

import torch
from torch.nn import functional as F

from .io import stable_topk

CANDIDATES = 5
WEIGHTS = (0, .5, 1, 2, 5, 10, 20, 40)
QUOTAS = (1, 3, 8)
IMAGE_WEIGHTS = (.25, .5, 1.)


@dataclass(frozen=True)
class Parameters:
    weight: float
    examples_per_class: int
    image_weight: float

    def __post_init__(self):
        if self.weight not in WEIGHTS or self.examples_per_class not in QUOTAS or self.image_weight not in IMAGE_WEIGHTS:
            raise ValueError('Parameters must match the published search grid')


def project(features, text_mean, image_mean, device):
    text = F.normalize(features['text'].float() - text_mean, dim=-1)
    has_image = features['has_image']
    image = F.normalize(features['image'].float() - image_mean, dim=-1) * has_image[:, None]
    return text.to(device), image.to(device), has_image.to(device)


class ExemplarBank:
    def __init__(self, train, device):
        images = train['image'].float()
        has_image = train['has_image']
        self.text_mean = train['text'].float().mean(0)
        self.image_mean = images[has_image].mean(0) if has_image.any() else torch.zeros_like(images[0])
        self.text, self.image, self.has_image = project(train, self.text_mean, self.image_mean, device)
        self.labels = train['labels'].to(device)
        self.device = device

    def similarities(self, query, image_weight):
        text, image, has_image = query
        text_similarity = text.float() @ self.text.float().T
        both_have_images = has_image[:, None].float() * self.has_image[None, :].float()
        return text_similarity + image_weight * (image.float() @ self.image.float().T) * both_have_images

    def supports(self, similarities, candidates, quota):
        support = torch.empty(candidates.shape, device=similarities.device)
        floor = similarities.min(-1).values
        for label in candidates.unique().tolist():
            columns = (self.labels == label).nonzero().flatten()
            rows, positions = (candidates == label).nonzero(as_tuple=True)
            if len(columns):
                nearest = similarities[rows[:, None], columns].topk(min(quota, len(columns)), dim=-1).values
                support[rows, positions] = nearest.mean(-1)
            else:
                support[rows, positions] = floor[rows]
        return support

    def inputs(self, features):
        scores = features['scores'].to(self.device)
        candidates = stable_topk(scores, min(CANDIDATES, scores.shape[1]))
        return scores, candidates, project(features, self.text_mean, self.image_mean, self.device)

    def predict(self, features, params):
        scores, candidates, query = self.inputs(features)
        if params.weight == 0:
            return scores.argmax(-1).cpu()
        support = self.supports(self.similarities(query, params.image_weight), candidates, params.examples_per_class)
        return combine(scores, candidates, support, params.weight).argmax(-1).cpu()

    def select(self, validation):
        scores, candidates, query = self.inputs(validation)
        support = {}
        for beta in IMAGE_WEIGHTS:
            similarities = self.similarities(query, beta)
            for quota in QUOTAS:
                support[quota, beta] = self.supports(similarities, candidates, quota)
            del similarities
        best_key = None
        best_params = None
        for weight, quota, beta in product(WEIGHTS, QUOTAS, IMAGE_WEIGHTS):
            predictions = combine(scores, candidates, support[quota, beta], weight).argmax(-1).cpu()
            metrics = measure(validation['labels'], predictions, scores.shape[1])
            key = (round(metrics['accuracy'], 6), round(metrics['weighted_f1'], 6), -weight)
            if best_key is None or key > best_key:
                best_key = key
                best_params = Parameters(weight, quota, beta)
        return best_params


def combine(scores, candidates, support, weight):
    if weight == 0:
        return scores.clone()
    return scores.clone().float().scatter_add_(1, candidates, weight * (support - support.mean(-1, keepdim=True)))


def measure(labels, predictions, classes):
    correct = labels == predictions
    true_positive = torch.bincount(labels[correct], minlength=classes)
    observed = torch.bincount(labels, minlength=classes)
    predicted = torch.bincount(predictions, minlength=classes)
    f1 = 2 * true_positive.double() / (observed + predicted).clamp_min(1)
    return {'n': len(labels), 'accuracy': int(correct.sum()) / len(labels),
            'weighted_f1': float((observed * f1).sum()) / len(labels)}
