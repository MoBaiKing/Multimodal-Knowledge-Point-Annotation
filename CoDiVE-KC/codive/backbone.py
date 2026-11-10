"""LoRA question encoder and frozen, alpha-correct native visual evidence.

The visual tower's merger projects into the same native Qwen language hidden
space used for questions and concept names. No generative LM head is loaded.
Model inference is performed by the registered GPU worker, never at import.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLModel

from .data import RENDERER_VERSION, render_image_white


DEFAULT_MODEL_PATH = (
    "/data/dyl/huggingface-cache/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/"
    "snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
)
BACKBONE_VERSION = "qwen25vl3b_native_merged_visual_lora16_headtail_v1"


def _get(cfg: Any, key: str, default: Any) -> Any:
    return cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)


def last_valid_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Find the last real token under either left or right padding."""
    if attention_mask.ndim != 2 or attention_mask.shape[1] == 0:
        raise ValueError("Expected a nonempty [batch, sequence] attention mask")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    positions = positions.unsqueeze(0).expand_as(attention_mask)
    indices = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
    if bool((indices < 0).any()):
        raise ValueError("Cannot pool a sequence containing only padding")
    return indices


class QwenBackbone(nn.Module):
    """Native Qwen2.5-VL backbone with trainable language-attention LoRA only.

    ``cfg`` may be a dict or attribute object. Supported keys: ``model_path``,
    ``text_tokens`` (384, including the full chat template), ``max_pixels``
    (128 * 28 * 28), ``min_pixels`` (4 * 28 * 28), ``padding_side`` (right),
    ``lora_r`` (16), ``lora_alpha`` (32), ``lora_dropout`` (0.05), and
    ``gradient_checkpointing`` (True). Construction loads on CPU; callers move
    the complete module onto their authorized GPU before real inference.
    """

    def __init__(self, cfg: Any):
        super().__init__()
        self.model_path = str(_get(cfg, "model_path", DEFAULT_MODEL_PATH))
        self.text_tokens = int(_get(cfg, "text_tokens", 384))
        self.max_pixels = int(_get(cfg, "max_pixels", 128 * 28 * 28))
        self.min_pixels = int(_get(cfg, "min_pixels", 4 * 28 * 28))
        self.padding_side = str(_get(cfg, "padding_side", "right"))
        if self.padding_side not in ("left", "right"):
            raise ValueError("padding_side must be left or right")
        if self.min_pixels <= 0 or self.max_pixels < self.min_pixels:
            raise ValueError("Expected 0 < min_pixels <= max_pixels")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, local_files_only=True, use_fast=False,
            min_pixels=self.min_pixels, max_pixels=self.max_pixels,
        )
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = self.padding_side
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = Qwen2_5_VLModel.from_pretrained(
            self.model_path, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        self.hidden_size = int(base.config.text_config.hidden_size)
        base.config.use_cache = False
        base.config.text_config.use_cache = False
        base.visual.requires_grad_(False)
        base.visual.eval()
        self.model = get_peft_model(base, LoraConfig(
            r=int(_get(cfg, "lora_r", 16)),
            lora_alpha=int(_get(cfg, "lora_alpha", 32)),
            lora_dropout=float(_get(cfg, "lora_dropout", 0.05)),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        ))
        if bool(_get(cfg, "gradient_checkpointing", True)):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            # Explicitly support checkpointing through frozen token embeddings.
            self.model.enable_input_require_grads()
        self.visual.requires_grad_(False)
        self.visual.eval()
        marker = "__CODIVE_V7_CONTENT_MARKER__"
        template = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": marker}], tokenize=False,
            add_generation_prompt=True,
        )
        if template.count(marker) != 1:
            raise ValueError("Unexpected Qwen chat template content placement")
        prefix, suffix = template.split(marker)
        self._prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        self._suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        self._content_budget = self.text_tokens - len(self._prefix_ids) - len(self._suffix_ids)
        if self._content_budget < 8:
            raise ValueError("text_tokens is too small to preserve the chat template")

    @property
    def visual(self) -> nn.Module:
        return self.model.get_base_model().visual

    @property
    def device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    def train(self, mode: bool = True) -> "QwenBackbone":
        super().train(mode)
        self.visual.eval()
        return self

    def metadata(self) -> dict[str, Any]:
        return dict(version=BACKBONE_VERSION, model_path=self.model_path,
                    hidden_size=self.hidden_size, text_tokens=self.text_tokens,
                    max_pixels=self.max_pixels, min_pixels=self.min_pixels,
                    padding_side=self.padding_side, renderer=RENDERER_VERSION,
                    pool="last_valid_chat_token", visual="native_frozen_merger")

    def tokenize_questions(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        """Build full chat sequences with deterministic head/tail content cuts.

        The returned keys are exactly ``input_ids`` and ``attention_mask``.
        Tokenization does not decode/re-encode truncated contents, so token
        boundaries and the assistant prefix remain intact.
        """
        if not texts:
            raise ValueError("Tokenization expects at least one text")
        content_ids = self.tokenizer(
            [str(text) for text in texts], add_special_tokens=False,
            padding=False, truncation=False, return_attention_mask=False,
        )["input_ids"]
        sequences = []
        for ids in content_ids:
            if len(ids) > self._content_budget:
                head = (self._content_budget + 1) // 2
                tail = self._content_budget - head
                ids = ids[:head] + (ids[-tail:] if tail else [])
            sequences.append(self._prefix_ids + ids + self._suffix_ids)
        result = self.tokenizer.pad(
            {"input_ids": sequences}, padding=True, return_attention_mask=True,
            return_tensors="pt",
        )
        return {key: result[key] for key in ("input_ids", "attention_mask")}

    def _encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.hidden_size), device=self.device, dtype=torch.float32)
        batch = {key: value.to(self.device) for key, value in self.tokenize_questions(texts).items()}
        output = self.model(**batch, use_cache=False, return_dict=True)
        hidden = output.last_hidden_state
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device),
                        last_valid_indices(batch["attention_mask"])]
        return pooled.float()

    def encode_questions(self, texts: Sequence[str]) -> torch.Tensor:
        """Return trainable [B, H] question states; gradients reach LoRA."""
        return self._encode_texts(texts)

    def encode_question_states(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pooled [B, H] state plus all [B, T, H] question token states and [B, T] mask.

        Same forward as ``encode_questions``; token states keep gradients to LoRA
        so the concept-pair evidence head can shape the question encoding.
        """
        batch = {key: value.to(self.device) for key, value in self.tokenize_questions(texts).items()}
        hidden = self.model(**batch, use_cache=False, return_dict=True).last_hidden_state
        mask = batch["attention_mask"].bool()
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_valid_indices(batch["attention_mask"])]
        return pooled.float(), hidden, mask

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        return self.encode_questions(texts)

    @torch.no_grad()
    def encode_concepts(self, labels: Sequence[str]) -> torch.Tensor:
        """Encode label names using untouched base weights, without dropout.

        Call and cache once before optimization. Explicit adapter disabling
        also preserves the pretrained semantics after a resumed LoRA load.
        No descriptions, labels of examples, or validation targets are used.
        """
        was_training = self.training
        self.eval()
        adapter_context = self.model.disable_adapter() if hasattr(self.model, "disable_adapter") else nullcontext()
        try:
            with adapter_context:
                return self._encode_texts([str(label) for label in labels]).detach()
        finally:
            self.train(was_training)

    @torch.no_grad()
    def extract_image(self, path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
        """Return CPU [M,H] frozen native tokens and [M,2] (x,y) centers.

        Transparent source pixels are composed over white before resizing.
        Tokens remain in the native visual-merger spatial order. Uniform
        rendered images return empty tensors, allowing exact no-image fallback.
        """
        image = render_image_white(path)
        if not any(low != high for low, high in image.getextrema()):
            return (torch.empty((0, self.hidden_size), dtype=torch.float32),
                    torch.empty((0, 2), dtype=torch.float32))
        inputs = self.processor.image_processor(images=[image], return_tensors="pt")
        pixels = inputs["pixel_values"].to(device=self.device, dtype=self.visual.dtype)
        grid = inputs["image_grid_thw"].to(self.device)
        self.visual.eval()
        features = self.visual(pixels, grid_thw=grid)
        merge = int(self.visual.spatial_merge_size)
        coordinate_groups = []
        for temporal, height, width in grid.tolist():
            if height % merge or width % merge:
                raise ValueError("Native visual grid is not spatially mergeable")
            rows, columns = height // merge, width // merge
            yy, xx = torch.meshgrid(
                (torch.arange(rows, dtype=torch.float32) + 0.5) / rows,
                (torch.arange(columns, dtype=torch.float32) + 0.5) / columns,
                indexing="ij",
            )
            centers = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
            coordinate_groups.append(centers.repeat(temporal, 1))
        positions = torch.cat(coordinate_groups, dim=0)
        if features.ndim != 2 or features.shape != (positions.shape[0], self.hidden_size):
            raise ValueError(f"Unexpected native visual alignment: {tuple(features.shape)} vs {tuple(positions.shape)}")
        return features.detach().float().cpu(), positions
