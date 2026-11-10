# Multimodal-Knowledge-Point-Annotation

Code for multimodal knowledge-point (concept) classification of exam questions on two subjects, Math and Biology.

## CoDiVE-KC

Directory: [`CoDiVE-KC/`](CoDiVE-KC)

CoDiVE-KC predicts the knowledge point of a question with a Qwen2.5-VL-3B-Instruct backbone adapted by language-side LoRA, followed by class-balanced exemplar reranking:

- **Concept head** (`codive/heads.py`, `ConceptHead`): full-class semantic prototypes plus a direct classification head over the question token states.
- **Pair evidence head** (`PairEvidenceHead`): question/concept-conditioned, null-subtracted evidence over question token states (all rows) and native image tokens (rows with images).
- **Class-balanced exemplar reranking** (`egmr/`): for each frozen Top-5 candidate, retrieve up to the same number of similar training examples from that class and average their text/image similarities. Add the centered class support to the frozen candidate scores. The reranking stage requires no gradient training and does not use conflict-aware labels.

During training the student logits are mixed with a seed-matched MacBERT reference produced by grouped out-of-fold prediction. The mixing weight and the visual-evidence scale are selected on the validation split only, then frozen before the single test evaluation.

The reranking stage freezes the base classifier, builds its memory from training examples only, and selects its parameters on `val_select`. `val_calibration` is used only for independent checks. Test is read after the reranking settings are frozen. Similarity combines centered, L2-normalized text representations and global image-token means; the image term is used only when both examples have images.

```text
similarity(q,j) = cosine(text_q,text_j) + beta * cosine(image_q,image_j) * both_have_images
e_c = mean of the top min(m, class_size) similarities in training class c
score_c = frozen_score_c + lambda * (e_c - mean_candidate_support)
```

### Layout

| Path | Content |
|---|---|
| `codive/` | Backbone wrapper, data loading, heads, training/selection/test pipeline |
| `egmr/egmr.py` | Class-balanced exemplar scoring |
| `egmr/run_egmr.py` | Frozen feature extraction, training memory, parameter selection and test |
| `scripts/worker.py` | Entry point for one train → freeze → test run |
| `scripts/queue_runner.py` | Multi-GPU job scheduler used on our server |
| `scripts/export_results.py` | Collects verified runs into `results/` |
| `scripts/export_retrieval_results.py` | Exports the final method's per-seed scores and paired comparisons |
| `scripts/aggregate_retrieval.py` | Summarizes the retrieval controls and ablations |
| `config/` | Per-subject, per-seed run configurations |
| `stage0/`, `stage0b/`, `stage0c/` | Pre-study probes on whether image information helps on the validation split |
| `analysis_ablation_20261101/` | Component analysis scripts |
| `results/` | Per-seed and summary test metrics |
| `artifacts/experiment_plan.json` | Frozen experiment plan |
| `artifacts/retrieval_plan.json` | Retrieval grids, selection policy and code fingerprints |

After configuring the required external data, prepared references and frozen base-run paths, run reranking for one dataset/seed:

```bash
cd CoDiVE-KC
CUDA_VISIBLE_DEVICES=0 python egmr/run_egmr.py --dataset Math --seed 200408
```

The implementation retains its research controls; the final method is `egmr_mm_noconflict`, the class-balanced reranking variant. The `base` output is the unchanged frozen classifier. To train a base classifier with the already-prepared reference artifacts, use `scripts/worker.py --config config/math_concept_evidence_seed200408.json --device cuda:0`.

### Test results

Mean over seeds **200404, 200405, 200406, 200407 and 200408** (%). Std is the sample standard deviation. Seed200409 is listed separately in `results/final.md`; validation and test results are distinguished.

| Dataset | ACC | Weighted-F1 | Macro-F1 |
|---|---|---|---|
| Math | 77.48 ± 0.53 | 77.11 ± 0.48 | 73.88 ± 0.62 |
| Biology | 78.09 ± 0.48 | 77.74 ± 0.52 | 75.05 ± 0.72 |

These are the saved final-test results from the completed experiments. Overlap-excluded scores and the frozen-base comparisons are included in `results/final.{json,csv,md}`.

The reported final variant was included in the predefined comparison and selected for presentation after the comparison results were available. Its already-frozen validation-selected parameters and existing test outputs were reused.

### Environment

Python 3.10, torch 2.2.2 (CUDA 12.1), transformers 4.57.1, peft 0.17.1, Pillow 12.3.0, numpy 1.26.4.

### Not included

This repository contains code, configurations and result summaries only. The following are not included:

- the Math/Biology datasets, images and fixed splits;
- pretrained weights (Qwen2.5-VL-3B-Instruct, MacBERT) and trained checkpoints;
- feature caches and run logs;
- the fixed split protocol and validation roles, MacBERT grouped-OOF reference predictions, and prepared train/validation data required by the base classifier.

Paths in `codive/data.py` and `config/*.json` point to locations on our server and must be changed before running elsewhere.
