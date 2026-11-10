"""Preserved train/validation protocol, alpha-correct images, verified text OOF.

This module deliberately does not import old learners. Normal preparation never
opens test.csv, old test predictions, or test-only image files. The explicit test
entry point requires a frozen final checkpoint record first.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .io import canonical_text, fingerprint, read_json, sha256_file

ASSETS_ROOT = Path("/data/dyl/knowledge_point_assets")
REFERENCE_ROOT = Path("/data/dyl/DIVE-KC-v5-RL")
PILOT_ROOT = Path("/data/dyl/DIVE-KC-v6.1-Grounding")
RENDERER_VERSION = "rgba_alpha_composite_on_white_v1"
SUBJECTS = ("Math", "Biology")
SEED = 200408


def render_image_white(path: str | Path | Image.Image) -> Image.Image:
    """Render transparent educational line drawings on a white page."""
    if isinstance(path, Image.Image):
        rgba = path.convert("RGBA")
    else:
        with Image.open(path) as source:
            rgba = source.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rendered = Image.alpha_composite(background, rgba).convert("RGB")
    rendered.load()
    return rendered


render_rgb = render_image_white


def inspect_image(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    result: dict[str, Any] = {"path": str(path), "sha256": sha256_file(path), "valid": False,
                              "renderer": RENDERER_VERSION}
    try:
        with Image.open(path) as source:
            source.verify()
        rgb = render_image_white(path)
        result.update(width=rgb.width, height=rgb.height)
        result["valid"] = any(low != high for low, high in rgb.getextrema())
        result["reason"] = "verified_and_rendered" if result["valid"] else "uniform_rendered_image"
    except Exception as exc:
        result.update(reason="unreadable", error=f"{type(exc).__name__}: {exc}")
    return result


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {"index", "content", "label", "kn_id"}
        if not required.issubset(fields):
            raise ValueError(f"Missing fields in {path}: {required - set(fields)}")
        return fields, list(reader)


def _verify_hash(path: str | Path, expected: str, label: str = "artifact") -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Changed {label}: {path} ({actual} != {expected})")
    return actual


def _sources(subject: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if subject not in SUBJECTS:
        raise ValueError(f"Expected subject in {SUBJECTS}, got {subject!r}")
    protocol = read_json(REFERENCE_ROOT / "data" / subject / "protocol.json")
    roles_path = REFERENCE_ROOT / "data" / subject / "val_roles.json"
    roles = read_json(roles_path)
    if protocol["subject"] != subject or roles["subject"] != subject:
        raise ValueError("Reference subject mismatch")
    # The historical field is a canonical JSON fingerprint, not a file SHA.
    role_content = {key: value for key, value in roles.items() if key != "fingerprint"}
    if roles.get("fingerprint") != fingerprint(role_content) or roles["fingerprint"] != protocol["validation_roles_sha256"]:
        raise ValueError("Fixed validation role content fingerprint changed")
    _verify_hash(protocol["label_file"]["path"], protocol["label_file"]["sha256"], "class mapping source")
    mapping = protocol["label_mapping"]
    if [row["label"] for row in mapping] != list(range(len(mapping))):
        raise ValueError("Class mapping must retain contiguous original-ID order")
    if [row["original_id"] for row in mapping] != sorted(row["original_id"] for row in mapping):
        raise ValueError("Changed original class-ID order")
    if fingerprint([(row["original_id"], row["name"]) for row in mapping]) != protocol["label_mapping_sha256"]:
        raise ValueError("Class mapping fingerprint mismatch")
    return protocol, roles


def _materialize(subject: str, split: str, protocol: dict[str, Any],
                 image_groups: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    info = protocol["source_files"][split]
    path = ASSETS_ROOT / "DataSet" / subject / f"{split}.csv"
    actual_sha = _verify_hash(path, info["sha256"], f"original {split} CSV")
    fields, raw = _read_rows(path)
    mapping = {row["original_id"]: row for row in protocol["label_mapping"]}
    records = []
    for offset, item in enumerate(raw):
        original_id = int(item["kn_id"])
        entry = mapping.get(original_id)
        if entry is None or item["label"] != entry["name"]:
            raise ValueError(f"Original label mismatch: {subject}/{split} row {offset}")
        source_sha = fingerprint(item)
        row_id = f"{subject}:{split}:{offset:06d}:{source_sha[:16]}"
        if row_id != protocol["split_row_ids"][split][offset]:
            raise ValueError(f"Original row identity mismatch: {subject}/{split} row {offset}")
        images = image_groups.get(item["index"], [])
        records.append(dict(source_row=offset, row_id=row_id, index=item["index"], text=item["content"],
                            label=entry["label"], original_label_id=original_id, split=split,
                            source_row_sha256=source_sha, image_paths=[im["path"] for im in images],
                            image_sha256=[im["sha256"] for im in images], image_available=bool(images),
                            images=[dict(path=im["path"], sha256=im["sha256"]) for im in images]))
    if len(records) != info["row_count"] or len(records) != len(protocol["split_row_ids"][split]):
        raise ValueError("Original row count changed")
    if fingerprint([row["row_id"] for row in records]) != info["row_ids_sha256"]:
        raise ValueError("Row ID order fingerprint changed")
    if fingerprint([row["source_row_sha256"] for row in records]) != info["rows_sha256"]:
        raise ValueError("Raw row content fingerprint changed")
    return records, dict(path=str(path), sha256=actual_sha, row_count=len(records), fields=fields,
                         row_ids_sha256=info["row_ids_sha256"], rows_sha256=info["rows_sha256"])


def _image_groups_from_manifest(subject: str, verify_images: bool) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    path = PILOT_ROOT / "artifacts" / subject / "image_manifest_train_val.json"
    manifest = read_json(path)
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    seen = set()
    for record in manifest:
        image_path = Path(record["path"])
        if str(image_path) in seen:
            raise ValueError(f"Repeated image manifest path: {image_path}")
        seen.add(str(image_path))
        if record.get("renderer") != RENDERER_VERSION:
            raise ValueError("Wrong old image renderer; refusing visual cache reuse")
        if verify_images:
            current = inspect_image(image_path)
            if current["sha256"] != record["sha256"] or current["valid"] != record["valid"]:
                raise ValueError(f"Image bytes/rendered validity changed: {image_path}")
            if current.get("width") != record.get("width") or current.get("height") != record.get("height"):
                raise ValueError(f"Decoded image dimensions changed: {image_path}")
        if not record["valid"]:
            continue
        match = re.fullmatch(r"(.+)_(\d+)\.[^.]+", image_path.name)
        if not match:
            raise ValueError(f"Unexpected audited image name: {image_path}")
        groups[match[1]].append((int(match[2]), record))
    grouped = {index: [row for _, row in sorted(values, key=lambda item: (item[0], item[1]["path"]))]
               for index, values in groups.items()}
    return grouped, dict(path=str(path), sha256=sha256_file(path), renderer=RENDERER_VERSION,
                         inspected_images=len(manifest), valid_images=sum(row["valid"] for row in manifest),
                         source_bytes_verified=verify_images)


def _training_groups(rows: list[dict[str, Any]], include_images: bool = False) -> list[str]:
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, row in enumerate(rows):
        keys = [("id", row["index"]), ("text", canonical_text(row["text"]))]
        if include_images:
            keys += [("image", digest) for digest in row["image_sha256"]]
        for key in keys:
            if key in seen:
                left, right = find(index), find(seen[key])
                parents[max(left, right)] = min(left, right)
            else:
                seen[key] = index
    members = defaultdict(list)
    for index, row in enumerate(rows):
        members[find(index)].append(row["row_id"])
    names = {root: fingerprint(sorted(values)) for root, values in members.items()}
    return [names[find(index)] for index in range(len(rows))]


def _duplicate_audit(train: list[dict[str, Any]], val: list[dict[str, Any]]) -> dict[str, Any]:
    train_ids = {row["index"] for row in train}
    train_texts = {canonical_text(row["text"]) for row in train}
    train_images = {digest for row in train for digest in row["image_sha256"]}
    matched = []
    for row in val:
        reasons = []
        if row["index"] in train_ids:
            reasons.append("index")
        if canonical_text(row["text"]) in train_texts:
            reasons.append("normalized_text")
        if set(row["image_sha256"]) & train_images:
            reasons.append("image_sha256")
        row["overlap_train"] = bool(reasons)
        row["train_overlap_reasons"] = reasons
        if reasons:
            matched.append(row["source_row"])
    return dict(policy="Descriptive only: no original rows or labels removed/changed",
                train_rows=len(train), train_unique_ids=len(train_ids), train_unique_normalized_texts=len(train_texts),
                validation_overlap_train_source_rows=matched, validation_overlap_train_count=len(matched),
                normalization="HTML unescape; Unicode NFKC; remove zero-width; collapse whitespace; audit only")


def load_train_val(subject: str, *, verify_images: bool = True) -> dict[str, Any]:
    """Load train/val with fixed source indices; never open any test artifacts."""
    protocol, roles = _sources(subject)
    groups, image_manifest = _image_groups_from_manifest(subject, verify_images)
    splits, sources = {}, {}
    for split in ("train", "val"):
        splits[split], sources[split] = _materialize(subject, split, protocol, groups)
    assignments = protocol["oof"]["assignments"]
    if fingerprint([row["fold"] for row in assignments]) != protocol["oof"]["assignment_sha256"]:
        raise ValueError("Original OOF assignments fingerprint changed")
    if len(assignments) != len(splits["train"]):
        raise ValueError("Missing training OOF assignments")
    text_groups = _training_groups(splits["train"])
    if fingerprint(text_groups) != protocol["oof"]["groups_sha256"]:
        raise ValueError("Original text OOF grouping fingerprint changed")
    n_folds = protocol["oof"]["n_splits"]
    folds = [dict(fold=fold, train=[], heldout=[]) for fold in range(n_folds)]
    for offset, (row, assignment, group) in enumerate(zip(splits["train"], assignments, text_groups)):
        if assignment["source_row"] != offset or assignment["row_id"] != row["row_id"] or assignment["group"] != group:
            raise ValueError("Training group, row or source order changed")
        if not 0 <= assignment["fold"] < n_folds:
            raise ValueError("Invalid OOF fold")
        row["text_fold"] = assignment["fold"]
        row["text_group"] = group
        for fold in folds:
            fold["heldout" if assignment["fold"] == fold["fold"] else "train"].append(offset)
    train_groups = _training_groups(splits["train"], include_images=True)
    for row, group in zip(splits["train"], train_groups):
        row["group"] = group
    select = roles["val_select"]["source_rows"]
    calibration = roles["val_calibration"]["source_rows"]
    if (len(set(select)) != len(select) or len(set(calibration)) != len(calibration)
            or set(select) & set(calibration) or sorted(select + calibration) != list(range(len(splits["val"])))):
        raise ValueError("Invalid/overlapping validation roles")
    if roles["external_val_sha256"] != sources["val"]["sha256"]:
        raise ValueError("Validation roles target a different original CSV")
    for name in ("val_select", "val_calibration"):
        if [splits["val"][i]["row_id"] for i in roles[name]["source_rows"]] != roles[name]["row_ids"]:
            raise ValueError("Validation role row identities changed")
    duplicates = _duplicate_audit(splits["train"], splits["val"])
    labels = [row["name"] for row in protocol["label_mapping"]]
    manifest = dict(method="CoDiVE-KC-v7", subject=subject, seed=SEED, allowed_pre_freeze_splits=["train", "val_select"],
                    external_data_policy="Unmodified source CSV; duplicates and original Biology labels preserved",
                    class_mapping=protocol["label_mapping"], class_mapping_sha256=protocol["label_mapping_sha256"],
                    source_files=sources, source_protocol_path=str(REFERENCE_ROOT / "data" / subject / "protocol.json"),
                    source_protocol_sha256=sha256_file(REFERENCE_ROOT / "data" / subject / "protocol.json"),
                    val_roles_path=str(REFERENCE_ROOT / "data" / subject / "val_roles.json"),
                    val_roles_sha256=sha256_file(REFERENCE_ROOT / "data" / subject / "val_roles.json"),
                    image_manifest=image_manifest, oof_assignment_sha256=protocol["oof"]["assignment_sha256"],
                    text_input="Original content, no normalization or rewriting",
                    image_policy="Independent white alpha composite; original bytes unchanged; no old visual-cache reuse",
                    candidates="Natural text top-k only; no gold-label injection; stable ascending class-ID ties",
                    calibration_policy="Reserved for final frozen model only; never training, selection or tuning",
                    duplicate_audit=duplicates)
    manifest["fingerprint"] = fingerprint(manifest)
    return dict(subject=subject, seed=SEED, splits=splits, rows=splits, labels=labels, label_names=labels,
                label_ids=[row["original_id"] for row in protocol["label_mapping"]],
                label_mapping=protocol["label_mapping"], select=select, calibration=calibration,
                val_select_indices=select, val_calibration_indices=calibration,
                folds=folds, train_groups=train_groups, duplicates=duplicates, manifest=manifest,
                protocol=manifest, fingerprint=manifest["fingerprint"])


def load_macbert_oof(subject: str, prepared: dict[str, Any] | None = None, *,
                     verify_checkpoints: bool = True) -> dict[str, Any]:
    """Reuse only verified same-protocol MacBERT grouped-OOF train/val predictions.

    All prediction hashes, completion hashes, split indices, targets, stage
    signatures, source checkpoint links and OOF coverage are checked. By default
    checkpoint bytes are also SHA verified. No old test predictions are opened.
    """
    prepared = load_train_val(subject) if prepared is None else prepared
    if prepared["subject"] != subject or prepared["seed"] != SEED:
        raise ValueError("Prepared dataset/seed mismatch")
    audit_path = PILOT_ROOT / "artifacts" / subject / "audit_summary.json"
    audit = read_json(audit_path)
    if audit["protocol_sha256"] != prepared["manifest"]["source_protocol_sha256"]:
        raise ValueError("Old text audit targets a different raw protocol")
    n_train, n_val = (len(prepared["splits"][name]) for name in ("train", "val"))
    n_classes = len(prepared["labels"])
    logits = torch.empty((n_train, n_classes), dtype=torch.float32)
    val_logits = None
    covered = torch.zeros(n_train, dtype=torch.bool)
    expected_train_targets = torch.tensor([row["label"] for row in prepared["splits"]["train"]], dtype=torch.long)
    expected_val_targets = torch.tensor([row["label"] for row in prepared["splits"]["val"]], dtype=torch.long)
    provenance, stages = [], set()
    for record in audit["predictions_provenance"]:
        stage = record["stage"]
        if stage in stages:
            raise ValueError("Duplicate old text stage")
        stages.add(stage)
        if stage != "full" and not re.fullmatch(r"fold\d+", stage):
            raise ValueError("Unexpected old text stage")
        folder = REFERENCE_ROOT / "shared" / subject / "text" / "macbert" / stage
        name = "val.pt" if stage == "full" else "heldout.pt"
        prediction_path = folder / name
        if prediction_path != Path(record["prediction_path"]):
            raise ValueError("Old text prediction is not original MacBERT stage")
        _verify_hash(folder / "complete.json", record["complete_sha256"], "OOF completion record")
        complete = read_json(folder / "complete.json")
        expected_seed = SEED if stage == "full" else SEED + 10 + int(stage[4:])
        if complete.get("stage") != f"text/macbert/{stage}" or complete.get("seed") != expected_seed:
            raise ValueError("Original text initialization/stage seed mismatch")
        if complete["common_signature"] != record["common_signature"]:
            raise ValueError("Original text training protocol signature mismatch")
        _verify_hash(prediction_path, record["prediction_sha256"], "OOF predictions")
        if complete["artifacts"][name] != record["prediction_sha256"]:
            raise ValueError("Prediction completion checksum mismatch")
        checkpoint_path = folder / "best.pt"
        if (Path(record["checkpoint_path"]) != checkpoint_path
                or complete["artifacts"]["best.pt"] != record["checkpoint_sha256"]):
            raise ValueError("Frozen reference checkpoint link mismatch")
        if verify_checkpoints:
            _verify_hash(checkpoint_path, record["checkpoint_sha256"], "source text checkpoint")
        payload = torch.load(prediction_path, map_location="cpu", weights_only=False)
        if payload["checkpoint_sha256"] != record["checkpoint_sha256"]:
            raise ValueError("Predictions refer to a different checkpoint")
        legacy_signature_payload = dict(common_signature=complete["common_signature"], stage=f"text/macbert/{stage}")
        expected_signature = hashlib.sha256(json.dumps(legacy_signature_payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        # The frozen text trainer signs its stage as text/{model}/{stage}.
        if payload["signature"] != expected_signature:
            raise ValueError("OOF prediction stage signature mismatch")
        expected = list(range(n_val)) if stage == "full" else prepared["folds"][int(stage[4:])]["heldout"]
        indices = [int(value) for value in payload["indices"]]
        if indices != expected:
            raise ValueError(f"Original heldout indices/order changed: {stage}")
        prediction = payload["logits"].float()
        targets = torch.as_tensor(payload["targets"], dtype=torch.long)
        expected_targets = expected_val_targets if stage == "full" else expected_train_targets[indices]
        if prediction.shape != (len(expected), n_classes) or not torch.isfinite(prediction).all():
            raise ValueError("Invalid original text prediction dimensions/values")
        if not torch.equal(targets, expected_targets):
            raise ValueError("Original heldout targets differ from unchanged CSV labels")
        if stage == "full":
            val_logits = prediction
        else:
            if covered[indices].any():
                raise ValueError("Duplicate original OOF coverage")
            covered[indices] = True
            logits[indices] = prediction
        provenance.append(dict(stage=stage, prediction_path=str(prediction_path), prediction_sha256=record["prediction_sha256"],
                               checkpoint_path=str(checkpoint_path), checkpoint_sha256=record["checkpoint_sha256"],
                               checkpoint_bytes_verified=verify_checkpoints, complete_sha256=record["complete_sha256"],
                               common_signature=complete["common_signature"], stage_signature=payload["signature"]))
    if (stages != {"full", *(f"fold{fold}" for fold in range(len(prepared["folds"])))}
            or not covered.all() or val_logits is None):
        raise ValueError("Original MacBERT OOF/reference is incomplete")
    return dict(train_logits=logits, val_logits=val_logits, train_targets=expected_train_targets,
                val_targets=expected_val_targets, covered=covered, train=logits, val=val_logits,
                provenance=provenance, provenance_fingerprint=fingerprint(provenance),
                selection_role="val_select only; val_calibration reserved until final freeze")


def load_test_after_freeze(subject: str, freeze_path: str | Path, seed: int = SEED) -> dict[str, Any]:
    """Explicit post-freeze test loader; independently audit only test-row images.

    Freeze record must say frozen=true, identify this subject/seed, and bind the
    existing final checkpoint bytes. Only this entry point may open test.csv.
    """
    freeze_path = Path(freeze_path)
    if not freeze_path.is_file():
        raise ValueError("Final freeze record is required before loading test")
    freeze = read_json(freeze_path)
    if (freeze.get("frozen") is not True or freeze.get("dataset", freeze.get("subject")) != subject
            or freeze.get("seed") != seed):
        raise ValueError("Freeze record lacks frozen=true or matching dataset/seed")
    checkpoint = freeze.get("checkpoint_path", freeze.get("checkpoint"))
    if not checkpoint or not freeze.get("checkpoint_sha256"):
        raise ValueError("Freeze record must bind final checkpoint path and checksum")
    _verify_hash(checkpoint, freeze["checkpoint_sha256"], "final frozen checkpoint")
    protocol, roles = _sources(subject)
    if freeze.get("val_roles_sha256", sha256_file(REFERENCE_ROOT / "data" / subject / "val_roles.json")) != sha256_file(REFERENCE_ROOT / "data" / subject / "val_roles.json"):
        raise ValueError("Freeze record uses different validation roles")
    test_path = ASSETS_ROOT / "DataSet" / subject / "test.csv"
    _verify_hash(test_path, protocol["source_files"]["test"]["sha256"], "original frozen test CSV")
    _, raw = _read_rows(test_path)
    permitted_indices = {row["index"] for row in raw}
    image_dir = Path(protocol["image_directory"])
    audit, candidates = [], defaultdict(list)
    for path in sorted(image_dir.iterdir()):
        match = re.fullmatch(r"(.+)_(\d+)\.[^.]+", path.name)
        if not path.is_file() or match is None or match[1] not in permitted_indices:
            continue
        record = inspect_image(path)
        audit.append(record)
        if record["valid"]:
            candidates[match[1]].append((int(match[2]), record))
    images = {index: [record for _, record in sorted(values, key=lambda item: (item[0], item[1]["path"]))]
              for index, values in candidates.items()}
    rows, source = _materialize(subject, "test", protocol, images)
    return dict(subject=subject, seed=seed, split="test", rows=rows,
                labels=[row["name"] for row in protocol["label_mapping"]],
                image_audit=audit, source_file=source, freeze_path=str(freeze_path),
                freeze_sha256=sha256_file(freeze_path), renderer=RENDERER_VERSION)
