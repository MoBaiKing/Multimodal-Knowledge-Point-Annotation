# CBER-KC

**Class-Balanced Exemplar Reranking for Multimodal Knowledge Concept Labeling**

CBER-KC assigns a knowledge concept to a question using its text and optional images. A frozen classifier proposes the five highest-scoring concepts. For each candidate, CBER-KC retrieves the most similar labeled training questions from that class, using the same quota per class whenever enough examples are available.

Text and image features are centered using the training bank and L2-normalized. The image term contributes only when both the query and the exemplar have an image. Let `f(c)` be the frozen classifier score and `e(c)` the mean similarity of the selected examples for class `c`. CBER-KC adds `λ × (e(c) − mean(e))` to each candidate score. The mean is taken over the five candidates. All other scores keep their original values.

The frozen encoder used in the reported experiments combines a LoRA-adapted Qwen2.5-VL-3B-Instruct backbone, a concept classification head, a question and image evidence readout, and a seed-matched MacBERT reference. Qwen question states and mean native visual tokens furnish the retrieval features. The search grid and scoring rules are in [the protocol](docs/protocol.md).

## Run CBER-KC

Install with Python 3.10+ and PyTorch 2.6+.

```bash
python -m pip install -e .
python examples/demo.py
cber-kc select --train runs/demo/train.pt --query runs/demo/val_select.pt \
  --frozen runs/demo/frozen.json
cber-kc evaluate --train runs/demo/train.pt --query runs/demo/test.pt \
  --frozen runs/demo/frozen.json --split test --output runs/demo/test.json
```

The demo generates synthetic feature packs and runs on CPU. With a CUDA-enabled PyTorch installation, use `--device cuda` to score real feature packs on a GPU. Use `--split val_calibration` to evaluate the held-out validation role. The `predict` command takes an unlabeled query pack with `split="predict"` and returns predictions keyed by row ID.

Use supplied feature packs directly or install `python -m pip install -e '.[encoder]'` and run `cber-kc extract` to generate them from question manifests and a trained checkpoint. [Input formats and extraction commands](docs/protocol.md) describe all external assets. This repository distributes the implementation, a runnable synthetic example, and aggregate experiment results. Reproducing the benchmark scores requires the benchmark splits, images, public pretrained snapshot, trained encoder checkpoint, and aligned frozen scores or reference logits.

## Reported test results

Mean ± sample standard deviation over seeds 200404–200408. Both metrics are percentages. Seed 200409 uses a separately trained reference and is reported separately in the [per-seed results](results/per_seed.csv).

| Dataset | ACC (%) | Weighted-F1 (%) |
| --- | ---: | ---: |
| Math | 77.48 ± 0.53 | 77.11 ± 0.48 |
| Biology | 78.09 ± 0.48 | 77.74 ± 0.52 |

The [result summary](results/summary.json) includes both validation roles and the test slice that excludes questions overlapping training. These are archived experimental scores. The demo uses synthetic data.
