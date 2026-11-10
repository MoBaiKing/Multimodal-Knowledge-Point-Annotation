#!/usr/bin/env python3
"""One registered GPU train -> freeze -> immediate final-test experiment."""
import argparse
import json
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codive import pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, pipeline.on_signal)
    cfg = json.loads(Path(args.config).read_text())
    if 'teacher_stage' in cfg:      # seed-matched MacBERT 参考阶段（kind=reference）
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'teacher'))
        import train_stage
        sys.exit(train_stage.run(cfg, args.device))
    sys.exit(pipeline.run(cfg, args.device))
