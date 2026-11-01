#!/usr/bin/env python3
"""CoDiVE-KC v8 stage 0b: does joint image+question encoding expose usable evidence?

Base Qwen2.5-VL-3B (adapters off, no training of the backbone) encodes each
image row twice through the identical chat template: question only (hq_txt)
and images + question (hq_mm). For val_select it also encodes the question
with another question's images (hq_shuf). The stage 0 top-5 reranking probe
(text branch only) is trained on hq_txt and on hq_mm, 3 seeds each, epoch
chosen on the same group-held-out 10% of train image rows, and reported on
val_select image rows. Test data is never loaded.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, '/data/dyl/CoDiVE-KC-v8/stage0')
import stage0_probe as s0  # noqa: E402

OUT = Path('/data/dyl/CoDiVE-KC-v8/stage0b')


def donors(part):
    out = {}
    for k, x in enumerate(part):
        for step in range(1, len(part)):
            y = part[(k + step) % len(part)]
            if y['index'] != x['index'] and not set(y['image_sha256']) & set(x['image_sha256']):
                out[k] = (k + step) % len(part)
                break
    return out


@torch.no_grad()
def extract(ds, device):
    from codive.backbone import QwenBackbone
    from codive.data import render_image_white
    _, rows, _ = s0.load(ds)
    bb = QwenBackbone(dict(model_path=s0.MODEL, max_pixels=s0.RESOLUTIONS['native'],
                           gradient_checkpointing=False)).to(device).eval()
    proc, tok = bb.processor, bb.tokenizer

    def images_of(x):
        imgs = [render_image_white(p) for p in x['image_paths'][:s0.MAX_IMAGES]]
        return [im for im in imgs if any(lo != hi for lo, hi in im.getextrema())]

    def encode(text, images):
        ids = tok(str(text), add_special_tokens=False)['input_ids']
        if len(ids) > bb._content_budget:
            head = (bb._content_budget + 1) // 2
            ids = ids[:head] + ids[-(bb._content_budget - head):]
        content = [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': tok.decode(ids)}]
        prompt = proc.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                          add_generation_prompt=True)
        inputs = proc(text=[prompt], images=images or None, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(bb.visual.dtype)
        out = bb.model(**inputs, use_cache=False, return_dict=True)
        return out.last_hidden_state[0, -1].float().cpu()

    result = dict(dataset=ds, splits={})
    with bb.model.disable_adapter():
        for split, part in rows.items():
            imgs = [images_of(x) for x in part]
            hq_txt = torch.stack([encode(x['text'], []) for x in part])
            hq_mm = torch.stack([encode(x['text'], im) for x, im in zip(part, imgs)])
            entry = dict(hq_txt=hq_txt.half(), hq_mm=hq_mm.half(), n_images=[len(i) for i in imgs])
            if split == 'val_select':
                d = donors(part)
                entry['donor'] = d
                entry['hq_shuf'] = torch.stack([encode(x['text'], imgs[d.get(k, k)]) for k, x in enumerate(part)]).half()
            result['splits'][split] = entry
            s0.log(f'{ds} {split}: encoded {len(part)} rows (text-only, joint{", shuffled" if split == "val_select" else ""})')
    torch.save(result, OUT/f'joint_{ds}.pt')


def train_probe(train, fit_idx, dev_idx, sem, device, seed):
    torch.manual_seed(seed); random.seed(seed)
    model = s0.Probe(sem.shape[1], False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    best, best_state, best_epoch, bad = -1, None, 0, 0
    for epoch in range(30):
        model.train()
        order = fit_idx[:]
        random.shuffle(order)
        for s in range(0, len(order), 64):
            hq, c, z, _, _, target, inside, _ = s0.batches(train, order[s:s+64], sem, None, device)
            if inside.any():
                loss = F.cross_entropy(model(hq, c, z)[inside], target[inside])
                opt.zero_grad(); loss.backward(); opt.step()
        score = s0.evaluate(model, train, sem, None, device, dev_idx)['acc']
        if score > best:
            best, best_epoch, bad = score, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    model.load_state_dict(best_state)
    return model, best, best_epoch


def probe(ds, device):
    feats = torch.load(s0.OUT/f'features_{ds}.pt', map_location='cpu', weights_only=False)
    joint = torch.load(OUT/f'joint_{ds}.pt', map_location='cpu', weights_only=False)
    sem = feats['semantics'].float().to(device)
    tr, va = feats['splits']['train'], feats['splits']['val_select']
    jt, jv = joint['splits']['train'], joint['splits']['val_select']
    dev_groups = {g for g in set(tr['group']) if int(s0.hashlib.sha256(str(g).encode()).hexdigest(), 16) % 10 == 0}
    fit_idx = [i for i, g in enumerate(tr['group']) if g not in dev_groups]
    dev_idx = [i for i, g in enumerate(tr['group']) if g in dev_groups]
    val_idx = list(range(len(va['y'])))
    view = lambda base, hq: dict(base, hq=hq)
    report = dict(dataset=ds, val_select_image_rows=len(val_idx),
                  teacher_top1_val_image_acc=round(100*float((va['base'].argmax(-1) == va['y']).float().mean()), 2),
                  rows_without_valid_image=dict(train=sum(n == 0 for n in jt['n_images']),
                                                val_select=sum(n == 0 for n in jv['n_images'])),
                  hq_cosine_txt_vs_mm_val=round(float(F.cosine_similarity(jv['hq_txt'].float(), jv['hq_mm'].float()).mean()), 4),
                  runs={})
    for variant, key in (('probe_text', 'hq_txt'), ('probe_joint', 'hq_mm')):
        runs = []
        for seed in (0, 1, 2):
            model, dev_acc, epoch = train_probe(view(tr, jt[key]), fit_idx, dev_idx, sem, device, seed)
            run = dict(seed=seed, best_epoch=epoch, dev_acc=dev_acc,
                       val_real=s0.evaluate(model, view(va, jv[key]), sem, None, device, val_idx))
            if key == 'hq_mm':
                run['val_shuffled'] = s0.evaluate(model, view(va, jv['hq_shuf']), sem, None, device, val_idx)
                run['val_no_image'] = s0.evaluate(model, view(va, jv['hq_txt']), sem, None, device, val_idx)
            runs.append(run)
            s0.log(f'{ds} {variant} seed{seed}: {json.dumps(run)}')
        mean = {k: round(sum(r[k]['acc'] for r in runs)/len(runs), 2) for k in runs[0] if k.startswith('val_')}
        report['runs'][variant] = dict(mean_acc=mean, runs=runs)
    (OUT/f'probe_{ds}.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    s0.log(f'{ds} summary: teacher {report["teacher_top1_val_image_acc"]} | cos(txt,mm) {report["hq_cosine_txt_vs_mm_val"]} | ' +
           ' | '.join(f'{v} {r["mean_acc"]}' for v, r in report['runs'].items()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', default='Math,Biology')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        sys.exit('stage 0b needs a reservation-authorized GPU')
    for ds in args.datasets.split(','):
        extract(ds, 'cuda:0')
        probe(ds, 'cuda:0')
