"""Full-class concept prediction and null-subtracted pairwise visual evidence.

This module contains no backbone, dataset access, or experiment inference. Native
visual tokens are encoded independently of the question by the frozen backbone.
The pair head learns a question/concept-conditioned readout, not box supervision.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def stable_topk(logits: Tensor, k: int = 5) -> Tensor:
    """Descending FP32 score order; original class ID breaks exact ties."""
    if logits.ndim != 2 or not 1 <= k <= logits.shape[-1]:
        raise ValueError("Expected [batch, classes] logits and 1 <= k <= classes")
    return torch.argsort(logits.float(), dim=-1, descending=True, stable=True)[:, :k]


class ConceptHead(nn.Module):
    """Trainable full-class semantic prototypes plus an unrestricted direct head."""

    def __init__(self, semantics: Tensor, model_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        if semantics.ndim != 2 or min(semantics.shape) < 1:
            raise ValueError("semantics must have shape [classes, hidden_size]")
        self.num_classes, self.hidden_size = semantics.shape
        self.register_buffer("semantics", semantics.detach().float().clone())
        self.question_norm = nn.LayerNorm(self.hidden_size)
        self.question_projection = nn.Linear(self.hidden_size, model_dim, bias=False)
        self.semantic_projection = nn.Linear(self.hidden_size, model_dim, bias=False)
        self.prototype_offsets = nn.Parameter(torch.zeros(self.num_classes, model_dim))
        self.direct = nn.Linear(self.hidden_size, self.num_classes)
        self.dropout = nn.Dropout(dropout)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(5.0)))
        nn.init.zeros_(self.direct.bias)

    def forward(self, hq: Tensor) -> Tensor:
        if hq.ndim != 2 or hq.shape[-1] != self.hidden_size:
            raise ValueError("Question representation must have shape [batch, hidden_size]")
        # Heads operate in FP32 even when cached/backbone tokens are BF16.
        question = self.question_norm(hq.float())
        projected = F.normalize(self.question_projection(question), dim=-1)
        concepts = self.semantic_projection(self.semantics) + self.prototype_offsets
        concepts = F.normalize(concepts, dim=-1)
        semantic_logits = self.logit_scale.clamp(max=math.log(100.0)).exp() * (projected @ concepts.T)
        return semantic_logits + self.direct(self.dropout(question))


class PairEvidenceHead(nn.Module):
    """Bounded visual residual restricted to caller-provided candidates.

    The same scoring network evaluates actual and zero visual evidence. Ordered
    pair scores are antisymmetrized before subtraction. Consequently image-free
    rows and zero-token inputs contribute exactly zero, and candidate order has
    no semantic effect. Attention is an evidence proposal, not a verified box.
    """

    def __init__(self, hidden_size: int, num_classes: int, model_dim: int = 256,
                 num_heads: int = 4, max_residual: float = 2.0, dropout: float = 0.0,
                 gate_bias: float = -2.0):
        super().__init__()
        if model_dim % num_heads or min(hidden_size, num_classes, model_dim, num_heads) < 1:
            raise ValueError("Positive dimensions and model_dim divisible by num_heads required")
        if not 0 <= dropout < 1 or max_residual <= 0:
            raise ValueError("Invalid dropout/residual bound")
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.max_residual = float(max_residual)
        self.dropout = float(dropout)
        self.question_projection = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, model_dim), nn.GELU())
        self.semantic_projection = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, model_dim), nn.GELU())
        # Bias-free values make padded and zero visual tokens carry no evidence.
        self.key_projection = nn.Linear(hidden_size, model_dim, bias=False)
        self.value_projection = nn.Linear(hidden_size, model_dim, bias=False)
        self.query_projection = nn.Sequential(nn.Linear(4 * model_dim, model_dim), nn.GELU(), nn.LayerNorm(model_dim))
        self.readout = nn.Sequential(nn.Linear(3 * model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, 1))
        self.gate = nn.Linear(model_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_bias))
        nn.init.normal_(self.readout[-1].weight, std=0.02)
        nn.init.zeros_(self.readout[-1].bias)

    def _attend(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        batch, count, dim = query.shape
        length = key.shape[1]
        width = dim // self.num_heads
        query_heads = query.view(batch, count, self.num_heads, width).transpose(1, 2)
        key_heads = key.view(batch, length, self.num_heads, width).transpose(1, 2)
        value_heads = value.view(batch, length, self.num_heads, width).transpose(1, 2)
        scores = (query_heads @ key_heads.transpose(-1, -2)) / math.sqrt(width)
        expanded_mask = mask[:, None, None, :]
        scores = scores.masked_fill(~expanded_mask, torch.finfo(scores.dtype).min)
        attention = scores.softmax(-1) * expanded_mask
        # An entirely masked row returns zeros instead of NaNs or uniform mass.
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
        pooled = (attention @ value_heads).transpose(1, 2).reshape(batch, count, dim)
        return pooled, attention.mean(1)

    def _score(self, query: Tensor, attended: Tensor) -> Tensor:
        return self.readout(torch.cat((query, attended, query * attended), dim=-1)).squeeze(-1)

    def forward(self, hq: Tensor, semantics: Tensor, visual_tokens: Tensor,
                visual_mask: Tensor, candidates: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if hq.ndim != 2 or hq.shape[1] != self.hidden_size:
            raise ValueError("hq must have shape [batch, hidden_size]")
        batch = hq.shape[0]
        if semantics.shape != (self.num_classes, self.hidden_size):
            raise ValueError("Concept semantics dimensions differ from head")
        if visual_tokens.ndim != 3 or visual_tokens.shape[0] != batch or visual_tokens.shape[2] != self.hidden_size:
            raise ValueError("visual_tokens must have shape [batch, tokens, hidden_size]")
        if visual_mask.shape != visual_tokens.shape[:2] or visual_mask.dtype != torch.bool:
            raise ValueError("visual_mask must be a boolean [batch, tokens] tensor")
        if candidates.ndim != 2 or candidates.shape[0] != batch or candidates.dtype != torch.long:
            raise ValueError("candidates must be a long [batch, k] tensor")
        count = candidates.shape[1]
        if count < 1 or count > self.num_classes:
            raise ValueError("Candidate count is out of range")
        if bool(((candidates < 0) | (candidates >= self.num_classes)).any()):
            raise ValueError("Candidate class ID out of range")
        if count > 1 and bool((candidates.sort(-1).values[:, 1:] == candidates.sort(-1).values[:, :-1]).any()):
            raise ValueError("Candidates must be unique; gold injection is not performed")

        # A zero-length native sequence is represented by one fully masked token.
        if visual_tokens.shape[1] == 0:
            visual_tokens = visual_tokens.new_zeros((batch, 1, self.hidden_size))
            visual_mask = visual_mask.new_zeros((batch, 1))
        has_image = visual_mask.any(-1)
        question = self.question_projection(hq.float())
        concept = self.semantic_projection(semantics.float())[candidates]
        first = concept[:, :, None, :].expand(-1, -1, count, -1)
        second = concept[:, None, :, :].expand(-1, count, -1, -1)
        question_pairs = question[:, None, None, :].expand(-1, count, count, -1)
        query = self.query_projection(torch.cat((question_pairs, first, second, first - second), dim=-1))
        query = query.reshape(batch, count * count, self.model_dim)
        # Sanitize masked values before linear projection, including masked NaNs.
        tokens = torch.where(visual_mask[..., None], visual_tokens.float(), 0.0)
        # Optional token dropout is sampled once and reused by both score terms;
        # query/readout have no separate dropout that could break null subtraction.
        tokens = F.dropout(tokens, self.dropout, self.training)
        key, value = self.key_projection(tokens), self.value_projection(tokens)
        attended, attention = self._attend(query, key, value, visual_mask)
        raw = self._score(query, attended).reshape(batch, count, count)
        null = self._score(query, torch.zeros_like(attended)).reshape(batch, count, count)
        pair = 0.5 * (raw - raw.transpose(1, 2)) - 0.5 * (null - null.transpose(1, 2))
        pair = pair * has_image[:, None, None]
        gate = self.gate(question).sigmoid()
        candidate_residual = self.max_residual * torch.tanh(pair.sum(-1) / max(count - 1, 1)) * gate
        candidate_residual = (candidate_residual * has_image[:, None]).float()
        residual = hq.new_zeros((batch, self.num_classes), dtype=torch.float32).scatter(1, candidates, candidate_residual)
        return residual, dict(pair_scores=pair, raw_pair_scores=raw, null_pair_scores=null,
                              attention=attention.reshape(batch, count, count, -1),
                              gate=gate, has_image=has_image, candidates=candidates,
                              candidate_residual=candidate_residual)


V7_LOSS = dict(base_ce=0.5, nearest_rival=0.1, rival_margin=0.2, preserve_correct=0.2, mismatch_kl=0.05)
# v7's preserve-correct and mismatched-image terms drove the evidence gate to ~0.01
# (val_select diagnostics, 2026-10-31); this loss drops them and leans on the nearest rival.
CONCEPT_EVIDENCE_LOSS = dict(base_ce=0.5, nearest_rival=0.2, rival_margin=0.5, preserve_correct=0.0, mismatch_kl=0.0)


def classification_loss(base: Tensor, fused: Tensor, y: Tensor,
                        delta_mismatch: Tensor | None = None, *, weights: dict | None = None,
                        return_parts: bool = False) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Full-class supervised objective with separate confusion and harm costs.

    No rows are dropped when gold is outside the visual candidate set. The
    mismatch residual, if supplied, uses the same anchor logits; anchor targets
    are detached so the robustness penalty cannot be reduced by moving them.
    """
    if base.shape != fused.shape or base.ndim != 2 or y.shape != base.shape[:1] or base.shape[-1] < 2:
        raise ValueError("Expected matching full-class logits and [batch] targets")
    base32, fused32 = base.float(), fused.float()
    parts: dict[str, Tensor] = dict(fused_ce=F.cross_entropy(fused32, y), base_ce=F.cross_entropy(base32, y))
    gold_base = base32.gather(1, y[:, None]).squeeze(1)
    gold_fused = fused32.gather(1, y[:, None]).squeeze(1)
    competitors_base = base32.masked_fill(F.one_hot(y, base.shape[-1]).bool(), -torch.inf)
    rival = competitors_base.detach().argmax(-1)
    margin_base = gold_base - base32.gather(1, rival[:, None]).squeeze(1)
    margin_fused = gold_fused - fused32.gather(1, rival[:, None]).squeeze(1)
    w = V7_LOSS if weights is None else weights
    parts["nearest_rival"] = F.relu(w["rival_margin"] - margin_fused).mean()
    originally_correct = base32.detach().argmax(-1).eq(y)
    degradation = F.relu(margin_base.detach() - margin_fused)
    parts["preserve_correct"] = (degradation * originally_correct).sum() / originally_correct.sum().clamp_min(1)
    if delta_mismatch is None:
        parts["mismatch_kl"] = fused32.sum() * 0.0
    else:
        if delta_mismatch.shape != base.shape:
            raise ValueError("Mismatch residual must have full-class shape")
        anchor = base32.detach()
        parts["mismatch_kl"] = F.kl_div((anchor + delta_mismatch.float()).log_softmax(-1),
                                         anchor.softmax(-1), reduction="batchmean")
    loss = (parts["fused_ce"] + w["base_ce"] * parts["base_ce"] + w["nearest_rival"] * parts["nearest_rival"]
            + w["preserve_correct"] * parts["preserve_correct"] + w["mismatch_kl"] * parts["mismatch_kl"])
    return (loss, parts) if return_parts else loss
