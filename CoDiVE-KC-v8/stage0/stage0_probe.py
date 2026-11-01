#!/usr/bin/env python3
"""CoDiVE-KC v8 stage 0: is there extractable visual evidence for top-5 reranking?

Data: v7 seed200408 prepared train/val (no test exists in prepared.pt). Text base
is the frozen MacBERT reference (grouped OOF on train, full model on val).
Probes rerank the base top-5 on image rows only; the question is whether a
visual probe beats an identically trained text-only probe on val_select.

  extract: base Qwen2.5-VL-3B question states (adapters off), concept-name
           semantics, frozen native visual tokens at v7 resolution and at 4x pixels
  probe:   text_only / visual_native / visual_high, 3 init seeds each; epoch
           chosen on a group-held-out 10% of train image rows; reported on
           val_select image rows with real, shuffled and no images
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

V7 = Path('/data/dyl/CoDiVE-KC-v7')
sys.path.insert(0, str(V7))
OUT = Path('/data/dyl/CoDiVE-KC-v8/stage0')
MODEL = '/data/dyl/huggingface-cache/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3'
RESOLUTIONS = {'native': 100352, 'high': 401408}
K, MAX_IMAGES = 5, 3


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def load(ds):
    p = torch.load(V7/f'data/{ds}/prepared.pt', map_location='cpu', weights_only=False)
    d, r = p['data'], p['reference']
    train, val = d['splits']['train'], d['splits']['val']
    sel = d['val_select_indices']
    rows = dict(train=[dict(x, base=r['train_logits'][i]) for i, x in enumerate(train) if x['image_paths']],
                val_select=[dict(val[i], base=r['val_logits'][i]) for i in sel if val[i]['image_paths']])
    stats = {}
    for name, logits, ys, part in (('train_oof', r['train_logits'], r['train_targets'], train),
                                   ('val_select', r['val_logits'][sel], r['val_targets'][sel], [val[i] for i in sel])):
        img = torch.tensor([bool(x['image_paths']) for x in part])
        top = logits.float().topk(K, -1).indices
        right = top[:, 0] == ys
        inside = (top == ys[:, None]).any(-1)
        stats[name] = dict(n=len(part), image_rows=int(img.sum()), acc_all=round(100*float(right.float().mean()), 2),
                           acc_image=round(100*float(right[img].float().mean()), 2),
                           gold_in_top5_image=round(100*float(inside[img].float().mean()), 2),
                           image_wrong_but_in_top5=int((img & ~right & inside).sum()),
                           oracle_rerank_gain_points_all=round(100*float((img & ~right & inside).sum())/len(part), 2))
    return d, rows, stats


@torch.no_grad()
def extract(ds, device):
    from codive.backbone import QwenBackbone
    from codive.data import render_image_white
    from transformers import AutoProcessor
    d, rows, stats = load(ds)
    backbone = QwenBackbone(dict(model_path=MODEL, max_pixels=RESOLUTIONS['native'], gradient_checkpointing=False)).to(device)
    backbone.eval()
    out = dict(dataset=ds, labels=d['labels'], stats=stats, splits={})
    with backbone.model.disable_adapter():
        sem = torch.cat([backbone._encode_texts(d['labels'][i:i+12]).cpu() for i in range(0, len(d['labels']), 12)])
        out['semantics'] = sem.half()
        for split, part in rows.items():
            hq = torch.cat([backbone._encode_texts([x['text'] for x in part[i:i+16]]).cpu()
                            for i in range(0, len(part), 16)])
            out['splits'][split] = dict(hq=hq.half(), base=torch.stack([x['base'].float() for x in part]),
                                        y=torch.tensor([x['label'] for x in part]),
                                        group=[x.get('group', x['index']) for x in part], index=[x['index'] for x in part],
                                        image_sha256=[x['image_sha256'][:MAX_IMAGES] for x in part])
            log(f'{ds} {split}: {len(part)} question states')
    for res, pixels in RESOLUTIONS.items():
        processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, use_fast=False,
                                                  min_pixels=4*28*28, max_pixels=pixels).image_processor
        cache = {}
        for split, part in rows.items():
            tokens = []
            for x in part:
                feats = []
                for path, digest in zip(x['image_paths'][:MAX_IMAGES], x['image_sha256'][:MAX_IMAGES]):
                    if digest not in cache:
                        image = render_image_white(path)
                        if not any(lo != hi for lo, hi in image.getextrema()):
                            cache[digest] = torch.empty(0, backbone.hidden_size, dtype=torch.float16)
                        else:
                            inputs = processor(images=[image], return_tensors='pt')
                            cache[digest] = backbone.visual(inputs['pixel_values'].to(device, backbone.visual.dtype),
                                                            grid_thw=inputs['image_grid_thw'].to(device)).half().cpu()
                    feats.append(cache[digest])
                tokens.append(torch.cat(feats) if feats else torch.empty(0, backbone.hidden_size, dtype=torch.float16))
            out['splits'][split][f'visual_{res}'] = tokens
            lengths = torch.tensor([len(t) for t in tokens]).float()
            log(f'{ds} {split} {res}: tokens/row median {lengths.median():.0f} max {lengths.max():.0f}')
    path = OUT/f'features_{ds}.pt'
    torch.save(out, path)
    log(f'saved {path}')


class Probe(nn.Module):
    def __init__(self, hidden, visual, d=256):
        super().__init__()
        self.visual = visual
        self.q = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, d))
        self.c = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, d))
        self.text = nn.Sequential(nn.Linear(3*d + K + 1, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 1))
        if visual:
            self.vn = nn.LayerNorm(hidden)
            self.k = nn.Linear(hidden, d, bias=False)
            self.v = nn.Linear(hidden, d, bias=False)
            self.vis = nn.Sequential(nn.Linear(3*d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 1))

    def forward(self, hq, sem, z, tokens=None, mask=None):
        B = hq.shape[0]
        q = self.q(hq)[:, None].expand(-1, K, -1)
        c = self.c(sem)
        rank = torch.eye(K, device=hq.device)[None].expand(B, -1, -1)
        score = z + self.text(torch.cat([q, c, q*c, z[..., None], rank], -1)).squeeze(-1)
        if self.visual and mask is not None and mask.any():
            vn = self.vn(tokens)
            key, value = self.k(vn), self.v(vn)
            att = ((q + c) @ key.transpose(1, 2)) / math.sqrt(key.shape[-1])
            att = att.masked_fill(~mask[:, None], torch.finfo(att.dtype).min).softmax(-1) * mask[:, None]
            r = att @ value
            contrast = c - c.mean(1, keepdim=True)
            score = score + self.vis(torch.cat([r, r*c, r*contrast], -1)).squeeze(-1) * mask.any(-1, keepdim=True)
        return score


def batches(split, idx, sem, res, device, donor=None, drop_images=False):
    hq = split['hq'][idx].float().to(device)
    base = split['base'][idx].log_softmax(-1)
    cand = base.topk(K, -1).indices
    z = base.gather(1, cand).to(device)
    y = split['y'][idx]
    target = (cand == y[:, None]).float().argmax(-1)
    inside = (cand == y[:, None]).any(-1)
    tokens = mask = None
    if res is not None:
        src = [donor.get(i, i) if donor else i for i in idx]
        seqs = [split[f'visual_{res}'][i] for i in src]
        if drop_images:
            seqs = [s[:0] for s in seqs]
        L = max(1, max(len(s) for s in seqs))
        tokens = torch.zeros(len(idx), L, sem.shape[1], device=device)
        mask = torch.zeros(len(idx), L, dtype=torch.bool, device=device)
        for i, s in enumerate(seqs):
            if len(s):
                tokens[i, :len(s)] = s.float().to(device)
                mask[i, :len(s)] = True
    return hq, sem[cand.to(device)], z, tokens, mask, target.to(device), inside.to(device), cand


def evaluate(model, split, sem, res, device, idx, **kw):
    model.eval()
    right, changed, fix, harm = 0, 0, 0, 0
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            b = idx[s:s+64]
            hq, c, z, tokens, mask, target, inside, _ = batches(split, b, sem, res, device, **kw)
            pred = model(hq, c, z, tokens, mask).argmax(-1)
            ok = (pred == target) & inside
            base_ok = (target == 0) & inside
            right += int(ok.sum()); changed += int((pred != 0).sum())
            fix += int((ok & ~base_ok).sum()); harm += int((~ok & base_ok).sum())
    return dict(acc=round(100*right/len(idx), 2), changed=changed, fix=fix, harm=harm)


def probe(ds, device):
    feats = torch.load(OUT/f'features_{ds}.pt', map_location='cpu', weights_only=False)
    sem = feats['semantics'].float().to(device)
    train, val = feats['splits']['train'], feats['splits']['val_select']
    dev_groups = {g for g in set(train['group']) if int(hashlib.sha256(str(g).encode()).hexdigest(), 16) % 10 == 0}
    fit_idx = [i for i, g in enumerate(train['group']) if g not in dev_groups]
    dev_idx = [i for i, g in enumerate(train['group']) if g in dev_groups]
    val_idx = list(range(len(val['y'])))
    donor = {}
    for k, i in enumerate(val_idx):
        for step in range(1, len(val_idx)):
            j = val_idx[(k + step) % len(val_idx)]
            if val['index'][j] != val['index'][i] and not set(val['image_sha256'][j]) & set(val['image_sha256'][i]):
                donor[i] = j
                break
    teacher_acc = round(100*float((val['base'].argmax(-1) == val['y']).float().mean()), 2)
    report = dict(dataset=ds, stats=feats['stats'], fit_rows=len(fit_idx), dev_rows=len(dev_idx),
                  val_select_image_rows=len(val_idx), teacher_top1_val_image_acc=teacher_acc, runs={})
    for variant, res in (('text_only', None), ('visual_native', 'native'), ('visual_high', 'high')):
        runs = []
        for seed in (0, 1, 2):
            torch.manual_seed(seed); random.seed(seed)
            model = Probe(sem.shape[1], res is not None).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
            best, best_state, bad = -1, None, 0
            for epoch in range(30):
                model.train()
                order = fit_idx[:]
                random.shuffle(order)
                for s in range(0, len(order), 64):
                    hq, c, z, tokens, mask, target, inside, _ = batches(train, order[s:s+64], sem, res, device)
                    if not inside.any():
                        continue
                    loss = F.cross_entropy(model(hq, c, z, tokens, mask)[inside], target[inside])
                    opt.zero_grad(); loss.backward(); opt.step()
                score = evaluate(model, train, sem, res, device, dev_idx)['acc']
                if score > best:
                    best, bad, best_state = score, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch + 1
                else:
                    bad += 1
                    if bad >= 5:
                        break
            model.load_state_dict(best_state)
            run = dict(seed=seed, best_epoch=best_epoch, dev_acc=best, val_real=evaluate(model, val, sem, res, device, val_idx))
            if res is not None:
                run['val_shuffled'] = evaluate(model, val, sem, res, device, val_idx, donor=donor)
                run['val_no_image'] = evaluate(model, val, sem, res, device, val_idx, drop_images=True)
            runs.append(run)
            log(f'{ds} {variant} seed{seed}: {json.dumps(run)}')
        mean = {k: round(sum(r[k]['acc'] for r in runs)/len(runs), 2) for k in runs[0] if k.startswith('val_')}
        report['runs'][variant] = dict(mean_acc=mean, runs=runs)
    (OUT/f'probe_{ds}.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    log(f'{ds} summary: teacher {report["teacher_top1_val_image_acc"]} | ' +
        ' | '.join(f'{v} {r["mean_acc"]}' for v, r in report['runs'].items()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['stats', 'extract', 'probe', 'all'], default='all')
    parser.add_argument('--datasets', default='Math,Biology')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets.split(','):
        if args.phase == 'stats':
            print(json.dumps(load(ds)[2], ensure_ascii=False))
            continue
        if not torch.cuda.is_available():
            sys.exit('stage 0 extract/probe needs a reservation-authorized GPU')
        if args.phase in ('extract', 'all'):
            extract(ds, 'cuda:0')
        if args.phase in ('probe', 'all'):
            probe(ds, 'cuda:0')
