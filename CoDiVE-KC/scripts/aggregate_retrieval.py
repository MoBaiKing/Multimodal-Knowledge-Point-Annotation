#!/usr/bin/env python3
"""汇总 method（base + EGMR）逐种子结果：主结果 = 种子 200404–200408 mean±std（样本标准差）与逐种子配对差；200409 单列。
只读各运行的 frozen_egmr.json / test_results.json；写 results/controls_summary.{json,csv,md}。验证与测试分开列出。"""
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN, EXTRA = [200404, 200405, 200406, 200407, 200408], [200409]
VARIANTS = ['base', 'knn_global', 'egmr_text', 'egmr_mm', 'egmr_mm_noconflict', 'egmr_mm_permuted']
METRICS = ['accuracy', 'weighted_f1', 'macro_f1']


def pct(x):
    return None if x is None else round(100 * x, 2)


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return dict(mean=round(st.mean(xs), 2), std=round(st.stdev(xs), 2) if len(xs) > 1 else None, n=len(xs))


def main():
    out, rows, md = dict(datasets={}), [], ['# CoDiVE-KC（base + EGMR）结果汇总', '',
                                           '主结果为种子 200404–200408 的 mean±std（样本标准差）；200409（独立参考教师）单列。验证集与测试集分开。', '']
    for ds in ('Math', 'Biology'):
        runs = {}
        for s in MAIN + EXTRA:
            d = ROOT / 'runs' / ds / f'seed{s}'
            if (d / 'test_results.json').exists():
                runs[s] = dict(test=json.loads((d / 'test_results.json').read_text()),
                               frozen=json.loads((d / 'frozen_egmr.json').read_text()))
        dsum = dict(completed_seeds=sorted(runs), missing_seeds=[s for s in MAIN + EXTRA if s not in runs], variants={})
        for v in VARIANTS:
            per = {}
            for s, r in runs.items():
                t, fz = r['test']['variants'][v], r['frozen']['variants'][v]
                per[s] = dict(params=t['params'],
                              test={m: pct(t['all'][m]) for m in METRICS},
                              test_excluding_train_overlap={m: pct((t['excluding_train_overlap'] or {}).get(m)) for m in METRICS},
                              val_select={m: pct(fz['val_select']['all'][m]) for m in METRICS},
                              val_calibration={m: pct(fz['val_calibration']['all'][m]) for m in METRICS},
                              fixes_vs_base=t['fixes_vs_base'], harms_vs_base=t['harms_vs_base'])
                for split in ('val_select', 'val_calibration', 'test', 'test_excluding_train_overlap'):
                    rows.append(dict(dataset=ds, seed=s, variant=v, split=split, **per[s][split],
                                     params=json.dumps(t['params'], ensure_ascii=False)))
            main_s = [s for s in MAIN if s in per]
            agg = {}
            for split in ('val_select', 'val_calibration', 'test', 'test_excluding_train_overlap'):
                agg[split] = {m: ms([per[s][split][m] for s in main_s]) for m in METRICS}
                agg[split + '_delta_vs_base'] = {m: ms([None if per[s][split][m] is None else per[s][split][m] - _base(runs[s], split, m)
                                                      for s in main_s]) for m in METRICS}
            agg['seeds_improved_test_acc'] = sum(per[s]['test']['accuracy'] > _base(runs[s], 'test', 'accuracy') for s in main_s)
            agg['seeds_worse_test_acc'] = sum(per[s]['test']['accuracy'] < _base(runs[s], 'test', 'accuracy') for s in main_s)
            dsum['variants'][v] = dict(per_seed=per, main_seeds=main_s, aggregate=agg)
        out['datasets'][ds] = dsum
        md += [f'## {ds}', '', f"已完成种子：{dsum['completed_seeds']}；缺：{dsum['missing_seeds']}", '',
               '| 变体 | val_select ACC | val_cal ACC | test ACC | test W-F1 | test Macro-F1 | Δtest ACC vs base（配对） | 改善/变差种子 | test ACC（排除与训练重合） | 200409 test ACC |',
               '|---|---|---|---|---|---|---|---|---|---|']
        for v in VARIANTS:
            a = dsum['variants'][v]['aggregate']
            f = lambda x: '—' if x is None else (f"{x['mean']:.2f}" + (f"±{x['std']:.2f}" if x['std'] is not None else ''))
            extra = dsum['variants'][v]['per_seed'].get(200409, {}).get('test', {}).get('accuracy')
            md.append(f"| {v} | {f(a['val_select']['accuracy'])} | {f(a['val_calibration']['accuracy'])} | {f(a['test']['accuracy'])} | "
                      f"{f(a['test']['weighted_f1'])} | {f(a['test']['macro_f1'])} | {f(a['test_delta_vs_base']['accuracy'])} | "
                      f"{a['seeds_improved_test_acc']}/{a['seeds_worse_test_acc']} | {f(a['test_excluding_train_overlap']['accuracy'])} | "
                      f"{'—' if extra is None else f'{extra:.2f}'} |")
        md.append('')
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results/controls_summary.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
    with open(ROOT / 'results/controls_summary.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=['dataset', 'seed', 'variant', 'split', *METRICS, 'params'])
        wr.writeheader()
        wr.writerows(rows)
    (ROOT / 'results/controls_summary.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


def _base(run, split, m):
    """同一种子、同一划分上 base 的指标（配对差的基线）。"""
    if split.startswith('test'):
        t = run['test']['variants']['base']
        return pct((t['all'] if split == 'test' else t['excluding_train_overlap'])[m])
    return pct(run['frozen']['variants']['base'][split]['all'][m])


if __name__ == '__main__':
    main()
