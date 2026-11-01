# Multimodal-Knowledge-Point-Annotation

Code for multimodal knowledge-point (concept) classification of exam questions on two subjects, Math and Biology.

## CoDiVE-KC v8

Directory: [`CoDiVE-KC-v8/`](CoDiVE-KC-v8)

v8 predicts the knowledge point of a question with a Qwen2.5-VL-3B-Instruct backbone adapted by language-side LoRA. It combines two heads:

- **Concept head** (`codive/heads.py`, `ConceptHead`): full-class semantic prototypes plus a direct classification head over the question token states.
- **Pair evidence head** (`PairEvidenceHead`): question/concept-conditioned, null-subtracted evidence over question token states (all rows) and native image tokens (rows with images).

During training the student logits are mixed with a seed-matched MacBERT reference produced by grouped out-of-fold prediction. The mixing weight and the visual-evidence scale are selected on the validation split only, then frozen before the single test evaluation.

### Layout

| Path | Content |
|---|---|
| `codive/` | Backbone wrapper, data loading, heads, training/selection/test pipeline |
| `scripts/worker.py` | Entry point for one train → freeze → test run |
| `scripts/queue_runner.py` | Multi-GPU job scheduler used on our server |
| `scripts/export_results.py` | Collects verified runs into `results/` |
| `config/` | Per-subject, per-seed run configurations |
| `stage0/`, `stage0b/`, `stage0c/` | Pre-study probes on whether image information helps on the validation split |
| `analysis_ablation_20261101/` | Component analysis scripts |
| `results/` | Per-seed and summary test metrics |
| `artifacts/experiment_plan.json` | Frozen experiment plan |

Run one experiment:

```bash
python CoDiVE-KC-v8/scripts/worker.py --config CoDiVE-KC-v8/config/math_concept_evidence_seed200408.json --device cuda:0
```

### Test results

Mean over seeds 200408 and 200409 (%). Std is the sample standard deviation.

| Dataset | ACC | Weighted-F1 | Macro-F1 |
|---|---|---|---|
| Math | 76.90 ± 0.46 | 76.84 ± 0.61 | 73.89 ± 0.60 |
| Biology | 78.18 ± 0.00 | 77.67 ± 0.13 | 75.04 ± 0.28 |

### Environment

Python 3.10, torch 2.2.2 (CUDA 12.1), transformers 4.57.1, peft 0.17.1, Pillow 12.3.0, numpy 1.26.4.

### Not included

This repository contains code, configurations and result summaries only. The following are not included:

- the Math/Biology datasets, images and fixed splits;
- pretrained weights (Qwen2.5-VL-3B-Instruct, MacBERT) and trained checkpoints;
- feature caches and run logs;
- artifacts from earlier versions that v8 reads: the split protocol and validation roles (v5), the MacBERT grouped-OOF reference predictions (v6.1), and the prepared train/validation data (v7).

Paths in `codive/data.py` and `config/*.json` point to locations on our server and must be changed before running elsewhere.
