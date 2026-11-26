# CBER-KC protocol

CBER-KC builds its exemplar bank from training rows, with one class label per row. The same train-only means are used to center both bank and query features. The visual mean is computed over training rows with images. The frozen classifier defines the five candidates, with ascending class ID as the tiebreaker. The support for a class is the mean of the nearest `min(m, class_count)` exemplars. A missing candidate class receives the query's minimum bank similarity. Candidate scores receive a centered support adjustment, and the final prediction is the full-class argmax.

The validation search enumerates `λ ∈ {0, 0.5, 1, 2, 5, 10, 20, 40}`, `m ∈ {1, 3, 8}`, and `β ∈ {0.25, 0.5, 1}` in that order. It compares accuracy, then weighted F1, then smaller λ, then grid order. Metrics are rounded to six decimal places for selection. The command writes a freeze record that binds the training and validation feature packs. Calibration and test use the frozen parameters.

## Feature packs

Each pack is a `torch.save` dictionary of tensors and primitive values. Use the same dataset, seed and encoder ID on all splits of a run.

| Key | Definition |
| --- | --- |
| `dataset`, `seed`, `encoder_id` | Dataset name, integer seed and identifier for the frozen encoder |
| `split` | `train`, `val_select`, `val_calibration`, `test` or `predict` |
| `row_ids` | List of unique question IDs in tensor order |
| `text`, `image` | Floating-point matrices of uncentered text and image representations |
| `has_image` | Boolean vector identifying valid images |
| `labels` | Int64 vector of class IDs, supplied on labeled splits |
| `scores` | Floating-point matrix of frozen classifier scores in class-ID order, supplied on query splits |

The frozen encoder uses a Qwen2.5-VL-3B-Instruct safetensors snapshot and a checkpoint with `semantics` (the concept prototype tensor) and `trainables` (the named LoRA and head tensors). The question representation is the last valid token state of the Qwen chat template. Transparent images are composed on white and uniform images are skipped. The visual representation is the mean of the native visual tokens over up to three images. With CUDA, a manifest can be converted as follows.

```bash
cber-kc extract --manifest inputs/train.jsonl --snapshot inputs/qwen-snapshot \
  --checkpoint inputs/encoder.pt --dataset Math --seed 200408 \
  --split train --output inputs/train.pt

cber-kc extract --manifest inputs/val_select.jsonl --snapshot inputs/qwen-snapshot \
  --checkpoint inputs/encoder.pt --scores inputs/val_select_scores.pt \
  --dataset Math --seed 200408 --split val_select --output inputs/val_select.pt

cber-kc select --train inputs/train.pt --query inputs/val_select.pt \
  --device cuda --frozen runs/math-frozen.json

cber-kc extract --manifest inputs/test.jsonl --snapshot inputs/qwen-snapshot \
  --checkpoint inputs/encoder.pt --scores inputs/test_scores.pt \
  --dataset Math --seed 200408 --split test --frozen runs/math-frozen.json \
  --output inputs/test.pt

cber-kc evaluate --train inputs/train.pt --query inputs/test.pt --device cuda \
  --frozen runs/math-frozen.json --split test --output runs/math-test.json
```

Each manifest row is a JSON object, for example `{"row_id":"question-1","text":"Question text","images":["images/question-1.png"],"label":0}`. Image paths may be relative to the manifest directory. Supply `images: []` for a text-only row. Score files contain `row_ids` and a floating-point `logits` tensor in the same order. For the reported protocol, the selection and test roles use the saved frozen classifier scores. The calibration role may compute these scores from aligned MacBERT logits using `--reference`, `--mix-weight` and `--evidence-scale` with the frozen encoder settings.

## Metrics

`val_select` is used for parameter selection. `val_calibration` and `test` report separate results. The overlap-excluded test slice removes rows that share an original question index, normalized question text, or image SHA-256 with training. The public tables report ACC and support-weighted F1 on a 0–100 scale. The main seed group uses the sample standard deviation. Seed 200409 is listed separately.

The headline values are aggregated from the existing runs of the class-balanced configuration. That configuration was selected for reporting after the comparison study. The run-level weights, quotas and image weights had already been selected on validation and frozen for each test.
