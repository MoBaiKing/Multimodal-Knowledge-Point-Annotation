#!/usr/bin/env python3
"""冻结 checkpoint 的证据分量导出（只推理，不训练、不选择）。

对一个已 TESTED 的 v7 pair_evidence 或 v8 concept_evidence 运行，在 val_select 与 test 上逐行导出：
文本基线 base = mix(student, teacher, 冻结 mix_weight)、top-k 候选，以及
  v8：题干证据残差 delta_q 与图片证据残差 delta_v（未乘视觉强度）
  v7：图片证据残差 delta_v（delta_q 恒 0）
无图行的 delta_v 置 0（与原推理的无图回退一致），并记录置零前的最大绝对值以供核查。
test 只用于冻结后的事后分析；所有开关组合离线计算，不回写任何选择。

用法：python dump_components.py --root <项目根> --config <运行配置> --out <输出.pt>
"""
import argparse
import json
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.root)
    sys.path.insert(0, str(root))
    from codive import pipeline as P
    from codive import data as D
    from codive.io import read_json, sha256
    topk = getattr(P, 'stable_topk', None) or __import__('codive.io', fromlist=['stable_topk']).stable_topk

    if not torch.cuda.is_available():
        raise RuntimeError('需要预约内的 GPU')
    cfg = json.loads(Path(args.config).read_text())
    run_dir = Path(cfg['run_dir'])
    if read_json(run_dir / 'status.json')['status'] != 'TESTED':
        raise ValueError('只分析已完成最终测试的运行')
    receipt = read_json(cfg['prepared_manifest'])
    if sha256(cfg['prepared']) != receipt['sha256']:
        raise ValueError('prepared 数据已变化')
    prepared = torch.load(cfg['prepared'], map_location='cpu', weights_only=False)
    data, reference = prepared['data'], prepared['reference']
    if data['subject'] != cfg['dataset'] or data['seed'] != cfg['seed']:
        raise ValueError('prepared 学科/种子不符')
    cfg = dict(cfg, labels=data['labels'])
    plan = read_json(root / 'artifacts/experiment_plan.json')
    frozen = read_json(run_dir / 'frozen.json')
    if sha256(run_dir / 'best.pt') != frozen['checkpoint_sha256']:
        raise ValueError('冻结 checkpoint 已变化')
    best = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=False)
    torch.set_num_threads(4)
    model = P.Model(cfg, semantics=best['semantics'])
    model.restore(best)
    model.eval()
    cache = P.VisionCache(cfg, model, plan['pretrained'])
    w = float(frozen['mix_weight'])
    k = cfg['candidate_k']
    v8 = cfg['variant'] == 'concept_evidence'

    # 种子匹配的教师 test 缓存（与 v8 teacher_test / v7 seed200409 补测相同的核验）
    full = next(x for x in reference['provenance'] if x['stage'] == 'full')
    folder = Path(full['checkpoint_path']).resolve().parent
    done = read_json(folder / 'complete.json')
    if done.get('seed') != cfg['seed'] or sha256(folder / 'test.pt') != done['artifacts']['test.pt'] \
            or full['checkpoint_sha256'] != done['artifacts']['best.pt']:
        raise ValueError('教师 test 缓存与种子/checkpoint 不符')
    teacher_test = torch.load(folder / 'test.pt', map_location='cpu', weights_only=False)

    try:
        test = D.load_test_after_freeze(cfg['dataset'], run_dir / 'frozen.json', seed=cfg['seed'])
    except TypeError:                                     # v7 的 data.py 无 seed 参数（种子为模块常量）
        test = D.load_test_after_freeze(cfg['dataset'], run_dir / 'frozen.json')
    if teacher_test['indices'] != list(range(len(test['rows']))) or \
            teacher_test['targets'].tolist() != [r['label'] for r in test['rows']]:
        raise ValueError('教师 test 行序/标签不符')
    select = data['val_select_indices']
    splits = {
        'val_select': ([data['splits']['val'][i] for i in select], reference['val_logits'][select].float()),
        'test': (test['rows'], teacher_test['logits'].float()),
    }
    out = dict(dataset=cfg['dataset'], seed=cfg['seed'], variant=cfg['variant'], root=str(root),
               mix_weight=w, frozen_visual_scale=float(frozen['visual_scale']), labels=data['labels'],
               checkpoint_sha256=frozen['checkpoint_sha256'], splits={})
    for name, (rows, teacher) in splits.items():
        rec = {key: [] for key in ('base', 'dq', 'dv', 'cands', 'has')}
        leak = 0.
        for start in range(0, len(rows), cfg['eval_batch']):
            batch = rows[start:start + cfg['eval_batch']]
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                if v8:
                    hq, student, states, qmask = model.text([r['text'] for r in batch], tokens=True)
                else:
                    hq, student = model.text([r['text'] for r in batch])
                base = P.mix(student, teacher[start:start + len(batch)].cuda(), w)
                cands = topk(base, min(k, base.shape[-1]))
                tokens, mask = cache.batch(batch)
                has = mask.any(-1) if mask is not None else torch.zeros(len(batch), dtype=torch.bool, device=base.device)
                if v8:
                    dq, dv = model.concept_evidence(hq, states, qmask, tokens, mask, cands)
                else:
                    dq = torch.zeros_like(base)
                    if bool(has.any()):
                        with torch.autocast('cuda', enabled=False):
                            dv, _ = model.evidence(hq.float(), model.semantics, tokens, mask, cands)
                    else:
                        dv = torch.zeros_like(base)
                dv = dv.float()
                if bool((~has).any()):
                    leak = max(leak, float(dv[~has].abs().max()))
                dv = torch.where(has[:, None], dv, torch.zeros_like(dv))
            for key, val in (('base', base), ('dq', dq), ('dv', dv), ('cands', cands), ('has', has)):
                rec[key].append(val.float().cpu() if key not in ('cands', 'has') else val.cpu())
        res = {key: torch.cat(v) for key, v in rec.items()}
        res['targets'] = torch.tensor([r['label'] for r in rows])
        res['row_ids'] = [r['row_id'] for r in rows]
        res['no_image_dv_max_abs_before_zeroing'] = leak
        out['splits'][name] = res
        print(json.dumps(dict(split=name, n=len(rows), image_rows=int(res['has'].sum()), no_image_leak=leak)), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print('DONE', args.out, flush=True)


if __name__ == '__main__':
    main()
