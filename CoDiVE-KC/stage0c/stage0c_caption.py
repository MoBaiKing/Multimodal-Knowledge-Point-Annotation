#!/usr/bin/env python3
"""CoDiVE-KC stage 0c: image -> text (OCR + structure description), then the stage 0 probe.

caption: Qwen2.5-VL-3B-Instruct (generation head, no training) describes each
         unique train/val_select image from the image alone (the question is
         not shown, so a donor caption is a clean shuffled control). Greedy,
         <=128 new tokens, 4x v7 pixels for small labels. Sharded, resumable.
probe:   base Qwen question states for "question + image description" vs the
         stage 0 question-only states; same top-5 reranking probe, 3 seeds,
         same group-held-out epoch choice; val_select image rows with the real
         description, a donor image's description, and no description.
Test data is never loaded.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, '/data/dyl/CoDiVE-KC/stage0')
sys.path.insert(0, '/data/dyl/CoDiVE-KC/stage0b')
import stage0_probe as s0  # noqa: E402
import stage0b_joint as s0b  # noqa: E402

OUT = Path('/data/dyl/CoDiVE-KC/stage0c')
PIXELS = s0.RESOLUTIONS['high']
PROMPT = {
    'Math': '这是一道中学数学题的配图。请只根据图片：1）逐字写出图中所有文字、字母、数字和标注；'
            '2）说明图形类型和关键结构（点、线段、角、平行或垂直、圆、坐标轴、函数图像、表格或统计图等）。'
            '不要解题，不要推测题目问什么。120字以内。',
    'Biology': '这是一道中学生物题的配图。请只根据图片：1）逐字写出图中所有文字、字母、数字和标注；'
               '2）说明图的类型和关键内容（细胞或结构示意、实验装置、曲线或柱状图的变量与趋势、遗传系谱、食物网等）。'
               '不要解题，不要推测题目问什么。120字以内。',
}


def unique_images(rows):
    seen = {}
    for part in rows.values():
        for x in part:
            for path, digest in zip(x['image_paths'][:s0.MAX_IMAGES], x['image_sha256'][:s0.MAX_IMAGES]):
                seen.setdefault(digest, path)
    return seen


@torch.no_grad()
def caption(ds, shard, nshards, device):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from codive.data import render_image_white
    _, rows, _ = s0.load(ds)
    todo = {d: p for d, p in sorted(unique_images(rows).items()) if int(d, 16) % nshards == shard}
    path = OUT/f'captions_{ds}_shard{shard}of{nshards}.json'
    done = json.loads(path.read_text()) if path.exists() else {}
    proc = AutoProcessor.from_pretrained(s0.MODEL, local_files_only=True, use_fast=False,
                                         min_pixels=4*28*28, max_pixels=PIXELS)
    proc.tokenizer.padding_side = 'left'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        s0.MODEL, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    prompt = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT[ds]}]}],
                                      tokenize=False, add_generation_prompt=True)
    pending = [d for d in todo if d not in done]
    s0.log(f'{ds} shard {shard}/{nshards}: {len(todo)} images, {len(pending)} to caption')
    for start in range(0, len(pending), 8):
        batch, images = [], []
        for d in pending[start:start+8]:
            image = render_image_white(todo[d])
            if any(lo != hi for lo, hi in image.getextrema()):
                batch.append(d); images.append(image)
            else:
                done[d] = ''
        if batch:
            inputs = proc(text=[prompt]*len(batch), images=images, padding=True, return_tensors='pt').to(device)
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            texts = proc.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            done.update({d: t.strip() for d, t in zip(batch, texts)})
        if start % 160 == 0:
            path.write_text(json.dumps(done, ensure_ascii=False))
            s0.log(f'{ds} shard {shard}: {len(done)}/{len(todo)}')
    path.write_text(json.dumps(done, ensure_ascii=False))
    s0.log(f'{ds} shard {shard}: finished {len(done)}/{len(todo)}')


def with_caption(text, digests, captions):
    parts = [captions.get(d, '') for d in digests]
    parts = [f'图{i+1}：{c}' for i, c in enumerate(parts) if c]
    return f'{text}\n【配图描述】' + '；'.join(parts) if parts else text


@torch.no_grad()
def encode(ds, device):
    from codive.backbone import QwenBackbone
    _, rows, _ = s0.load(ds)
    captions = {}
    for f in sorted(OUT.glob(f'captions_{ds}_shard*.json')):
        captions.update(json.loads(f.read_text()))
    missing = set(unique_images(rows)) - set(captions)
    if missing:
        raise SystemExit(f'{ds}: {len(missing)} images have no caption yet')
    bb = QwenBackbone(dict(model_path=s0.MODEL, gradient_checkpointing=False)).to(device).eval()
    enc = lambda texts: torch.cat([bb._encode_texts(texts[i:i+16]).cpu() for i in range(0, len(texts), 16)]).half()
    result = dict(dataset=ds, splits={})
    with bb.model.disable_adapter():
        for split, part in rows.items():
            digests = [x['image_sha256'][:s0.MAX_IMAGES] for x in part]
            entry = dict(hq_cap=enc([with_caption(x['text'], d, captions) for x, d in zip(part, digests)]),
                         caption_chars=[sum(len(captions[h]) for h in d) for d in digests])
            if split == 'val_select':
                donor = s0b.donors(part)
                entry['hq_shuf'] = enc([with_caption(x['text'], digests[donor.get(k, k)], captions) for k, x in enumerate(part)])
                entry['examples'] = [dict(index=x['index'], text=x['text'][:200], caption=with_caption('', d, captions))
                                     for x, d in list(zip(part, digests))[:12]]
            result['splits'][split] = entry
            s0.log(f'{ds} {split}: encoded {len(part)} question+caption rows')
    torch.save(result, OUT/f'caption_states_{ds}.pt')


def probe(ds, device):
    feats = torch.load(s0.OUT/f'features_{ds}.pt', map_location='cpu', weights_only=False)
    cap = torch.load(OUT/f'caption_states_{ds}.pt', map_location='cpu', weights_only=False)
    sem = feats['semantics'].float().to(device)
    tr, va = feats['splits']['train'], feats['splits']['val_select']
    ct, cv = cap['splits']['train'], cap['splits']['val_select']
    dev_groups = {g for g in set(tr['group']) if int(s0.hashlib.sha256(str(g).encode()).hexdigest(), 16) % 10 == 0}
    fit_idx = [i for i, g in enumerate(tr['group']) if g not in dev_groups]
    dev_idx = [i for i, g in enumerate(tr['group']) if g in dev_groups]
    val_idx = list(range(len(va['y'])))
    view = lambda base, hq: dict(base, hq=hq)
    chars = torch.tensor(ct['caption_chars'] + cv['caption_chars']).float()
    report = dict(dataset=ds, val_select_image_rows=len(val_idx),
                  teacher_top1_val_image_acc=round(100*float((va['base'].argmax(-1) == va['y']).float().mean()), 2),
                  caption_chars=dict(median=float(chars.median()), empty_rows=int((chars == 0).sum())),
                  hq_cosine_txt_vs_cap_val=round(float(F.cosine_similarity(va['hq'].float(), cv['hq_cap'].float()).mean()), 4),
                  examples=cv['examples'], runs={})
    for variant, train_hq in (('probe_text', tr['hq']), ('probe_caption', ct['hq_cap'])):
        runs = []
        for seed in (0, 1, 2):
            model, dev_acc, epoch = s0b.train_probe(view(tr, train_hq), fit_idx, dev_idx, sem, device, seed)
            if variant == 'probe_text':
                run = dict(seed=seed, best_epoch=epoch, dev_acc=dev_acc,
                           val_real=s0.evaluate(model, va, sem, None, device, val_idx))
            else:
                run = dict(seed=seed, best_epoch=epoch, dev_acc=dev_acc,
                           val_real=s0.evaluate(model, view(va, cv['hq_cap']), sem, None, device, val_idx),
                           val_shuffled=s0.evaluate(model, view(va, cv['hq_shuf']), sem, None, device, val_idx),
                           val_no_image=s0.evaluate(model, va, sem, None, device, val_idx))
            runs.append(run)
            s0.log(f'{ds} {variant} seed{seed}: {json.dumps(run)}')
        mean = {k: round(sum(r[k]['acc'] for r in runs)/len(runs), 2) for k in runs[0] if k.startswith('val_')}
        report['runs'][variant] = dict(mean_acc=mean, runs=runs)
    (OUT/f'probe_{ds}.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    s0.log(f'{ds} summary: teacher {report["teacher_top1_val_image_acc"]} | cos(txt,cap) {report["hq_cosine_txt_vs_cap_val"]} | ' +
           ' | '.join(f'{v} {r["mean_acc"]}' for v, r in report['runs'].items()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['caption', 'probe'], required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--nshards', type=int, default=1)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        sys.exit('stage 0c needs a reservation-authorized GPU')
    if args.phase == 'caption':
        caption(args.dataset, args.shard, args.nshards, 'cuda:0')
    else:
        encode(args.dataset, 'cuda:0')
        probe(args.dataset, 'cuda:0')
