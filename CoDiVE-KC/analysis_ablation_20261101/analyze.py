#!/usr/bin/env python3
"""离线分析 dump_components.py 的导出（不加载模型，不做任何选择）。

(1) 固定 checkpoint 开关消融：冻结 mix_weight 下，比较
    文本基线 / 只开题干证据 / 只开图片证据 / 题干+图片，强度 0.5 与 1.0；
    报告全部行与有图行的 ACC、Weighted-F1，以及相对文本基线的改判、纠正(fix)、伤害(harm)。
(3) 易混知识点对：证据改判按 {基线预测, 新预测} 知识点对汇总 fix/harm；
    基线最易混的 (真值, 基线预测) 对上，开启证据前后的该对准确率。
val_select 与 test 分开报告；test 仅为冻结后的事后分析。
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def f1_weighted(y, p, C):
    tp = torch.zeros(C); fp = torch.zeros(C); fn = torch.zeros(C)
    for a, b in zip(y.tolist(), p.tolist()):
        if a == b:
            tp[a] += 1
        else:
            fp[b] += 1; fn[a] += 1
    support = tp + fn
    f1 = torch.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn).clamp_min(1), torch.zeros(C))
    return float((f1 * support).sum() / support.sum())


def stats(y, p, base_p, mask, C):
    m = mask
    out = dict(n=int(m.sum()), acc=round(100 * float((p[m] == y[m]).float().mean()), 2) if m.any() else None,
               f1w=round(100 * f1_weighted(y[m], p[m], C), 2) if m.any() else None)
    ch = (p != base_p) & m
    out.update(changed=int(ch.sum()), fix=int((ch & (p == y) & (base_p != y)).sum()),
               harm=int((ch & (p != y) & (base_p == y)).sum()))
    return out


def configs(variant):
    cfgs = [('text_only', 0., 0., 0.)]
    for s in (0.5, 1.0):
        if variant == 'concept_evidence':
            cfgs += [(f'question@{s}', s, 0., s), (f'image@{s}', 0., s, s), (f'question+image@{s}', s, s, s)]
        else:
            cfgs += [(f'image@{s}', 0., s, s)]
    return cfgs


def analyze(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    labels = d['labels']; C = len(labels)
    rep = dict(dataset=d['dataset'], seed=d['seed'], variant=d['variant'], mix_weight=d['mix_weight'],
               frozen_visual_scale=d['frozen_visual_scale'], splits={})
    for split, r in d['splits'].items():
        y, base, dq, dv, has = r['targets'], r['base'], r['dq'], r['dv'], r['has'].bool()
        bp = base.argmax(-1)
        allm = torch.ones_like(has)
        tab, preds = {}, {}
        for name, a, b, _ in configs(d['variant']):
            p = (base + a * dq + b * dv).argmax(-1)
            preds[name] = p
            tab[name] = dict(all=stats(y, p, bp, allm, C), image_rows=stats(y, p, bp, has, C),
                             no_image_rows=stats(y, p, bp, ~has, C))
        # (3) 易混对：用"全部证据、强度 1.0"作为开启状态
        on_name = 'question+image@1.0' if d['variant'] == 'concept_evidence' else 'image@1.0'
        p_on = preds[on_name]
        pair_fx = defaultdict(lambda: Counter())
        for i in (p_on != bp).nonzero().flatten().tolist():
            key = tuple(sorted((int(bp[i]), int(p_on[i]))))
            kind = 'fix' if p_on[i] == y[i] else ('harm' if bp[i] == y[i] else 'wrong_to_wrong')
            pair_fx[key][kind] += 1
        changed_pairs = sorted(({'pair': [labels[a], labels[b]], **dict(c)} for (a, b), c in pair_fx.items()),
                               key=lambda x: -(x.get('fix', 0) + x.get('harm', 0) + x.get('wrong_to_wrong', 0)))
        confused = Counter((int(a), int(b)) for a, b in zip(y.tolist(), bp.tolist()) if a != b)
        top = []
        for (a, b), n in confused.most_common(10):
            m = ((y == a) | (y == b)) & ((bp == a) | (bp == b))
            top.append(dict(gold=labels[a], base_pred=labels[b], base_errors=n, rows_in_pair=int(m.sum()),
                            base_pair_acc=round(100 * float((bp[m] == y[m]).float().mean()), 2),
                            on_pair_acc=round(100 * float((p_on[m] == y[m]).float().mean()), 2),
                            image_rows_in_pair=int((m & has).sum())))
        rep['splits'][split] = dict(toggle=tab, evidence_on=on_name, changed_pairs=changed_pairs[:15],
                                    top_confused_pairs_base=top,
                                    no_image_dv_max_abs_before_zeroing=r['no_image_dv_max_abs_before_zeroing'],
                                    gold_in_top5=round(100 * float((r['cands'] == y[:, None]).any(-1).float().mean()), 2))
    return rep


def main():
    reports = [analyze(p) for p in sorted((HERE / 'dumps').glob('*.pt'))]
    (HERE / 'ablation_results.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2))
    lines = ['# v7 / CoDiVE-KC 冻结模型证据开关消融与易混对分析', '',
             'val_select 为验证集；test 为冻结后的事后分析（未用于任何选择）。fix/harm 相对同一冻结模型的文本基线（证据关）。', '']
    for r in reports:
        lines.append(f"## {r['variant']} {r['dataset']} seed{r['seed']}（冻结 mix={r['mix_weight']}，冻结强度={r['frozen_visual_scale']}）")
        for split, s in r['splits'].items():
            lines.append(f'### {split}（候选 top-5 覆盖真值 {s["gold_in_top5"]}%）')
            lines.append('| 配置 | ACC | F1w | 有图 ACC | 改判 | fix | harm |')
            lines.append('|---|---|---|---|---|---|---|')
            for name, t in s['toggle'].items():
                a, im = t['all'], t['image_rows']
                lines.append(f"| {name} | {a['acc']} | {a['f1w']} | {im['acc']} | {a['changed']} | {a['fix']} | {a['harm']} |")
            lines.append('')
            lines.append(f"证据开启（{s['evidence_on']}）改判的知识点对：")
            for c in s['changed_pairs'][:8]:
                lines.append(f"- {c['pair'][0]} ↔ {c['pair'][1]}：fix {c.get('fix', 0)}，harm {c.get('harm', 0)}，错→错 {c.get('wrong_to_wrong', 0)}")
            lines.append('')
    (HERE / 'ablation_report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    sys.exit(main())
