"""Small deterministic and atomic I/O helpers for CoDiVE-KC."""
from __future__ import annotations

import hashlib
import html
import json
import os
import random
import re
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


sha256 = sha256_file


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


write_json = atomic_write_json


def save_tensor(path: str | Path, value: Any) -> None:
    """Atomically save a CPU/GPU torch payload, without overwriting partial files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def canonical_text(text: str) -> str:
    """For overlap audits only; never substitute this for raw classifier input."""
    value = unicodedata.normalize("NFKC", html.unescape(text))
    value = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]", "", value)
    return re.sub(r"\s+", " ", value).strip()


normalize_text = canonical_text


def stable_topk(scores: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """Descending candidate IDs with ascending class-ID tie order.

    Returns only indices. No epsilon perturbation changes close scores. In
    particular a BF16 tie preserves the original full-class torch.argmax.
    """
    if scores.ndim == 0:
        raise ValueError("Candidate scores must have at least one dimension")
    if k < 0 or k > scores.shape[dim]:
        raise ValueError(f"Invalid k={k} for dimension of size {scores.shape[dim]}")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("Candidate scores contain non-finite values")
    return torch.argsort(scores, dim=dim, descending=True, stable=True).narrow(dim, 0, k)


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

