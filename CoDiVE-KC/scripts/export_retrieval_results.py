#!/usr/bin/env python3
"""method 最终方法 = 冻结 base + 类别均衡例证重排（只保留类别均衡；不用冲突感知标签）。
即预注册变体 egmr_mm_noconflict：只在 base 的 Top-K 候选内比较，每个候选取 m 道最相似的同类训练例题求平均相似度，
S_c = f_c + λ(e_c − 候选均值)；sim = 文本余弦 + β·图片余弦。每个 (数据集, 种子) 的 λ/m/β 仍是运行时只在 val_select 上
选择并冻结的结果；本脚本不重新选择、不重新推理，只从已核验的 frozen_egmr.json / test_results.json 导出。
输出 results/final.{json,csv,md}：逐种子 + 200404–200408 mean±std（样本标准差）+ 相对 base 的配对差；200409 单列。"""
import csv
import hashlib
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANT = 'egmr_mm_noconflict'
MAIN, EXTRA = [200404, 200405, 200406, 200407, 200408], [200409]
METRICS = ['accuracy', 'weighted_f1', 'macro_f1']
SPLITS = ['val_select', 'val_calibration', 'test', 'test_excluding_train_overlap']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pct(x):
    return None if x is None else round(100 * x, 2)


def ms(xs):
    return dict(mean=round(st.mean(xs), 2), std=round(st.stdev(xs), 2), n=len(xs))


def block(fz, te, v):
    return dict(val_select={m: pct(fz['variants'][v]['val_select']['all'][m]) for m in METRICS},
                val_calibration={m: pct(fz['variants'][v]['val_calibration']['all'][m]) for m in METRICS},
                test={m: pct(te['variants'][v]['all'][m]) for m in METRICS},
                test_excluding_train_overlap={m: pct(te['variants'][v]['excluding_train_overlap'][m]) for m in METRICS})


def main():
    out = dict(method='method = 冻结 base + 类别均衡例证重排（候选内、每类 m 道例题；无冲突感知标签）', source_variant=VARIANT,
               selection='每个 (数据集, 种子) 的 λ/m/β 只在 val_select 上选择并冻结（运行时完成），test 只评估一次',
               note='F1 主指标为 Weighted-F1 (%)；Macro-F1 仅作补充', datasets={})
    rows, md = [], ['# CoDiVE-KC 最终结果（类别均衡例证重排）', '',
                    '主结果 = 种子 200404–200408 mean±std（样本标准差）；200409（独立 MacBERT 教师）单列。验证集与测试集分开；F1 为 Weighted-F1 (%)。', '']
    for ds in ('Math', 'Biology'):
        per = {}
        for s in MAIN + EXTRA:
            d = ROOT / 'runs' / ds / f'seed{s}'
            fz, te = json.loads((d / 'frozen_egmr.json').read_text()), json.loads((d / 'test_results.json').read_text())
            if te['frozen_egmr_sha256'] != sha(d / 'frozen_egmr.json') or te['reproduce_base_test']['argmax_agree'] != te['reproduce_base_test']['n']:
                raise ValueError(f'{ds} seed{s} 核验失败')
            method, base = block(fz, te, VARIANT), block(fz, te, 'base')
            per[s] = dict(params=fz['variants'][VARIANT]['params'], method=method, base=base,
                          fixes_vs_base=te['variants'][VARIANT]['fixes_vs_base'], harms_vs_base=te['variants'][VARIANT]['harms_vs_base'])
            for split in SPLITS:
                rows.append(dict(dataset=ds, seed=s, split=split, **{f'method_{m}': method[split][m] for m in METRICS},
                                 **{f'base_{m}': base[split][m] for m in METRICS}, params=json.dumps(per[s]['params'])))
        agg = {}
        for split in SPLITS:
            agg[split] = {k: {m: ms([per[s][k][split][m] for s in MAIN]) for m in METRICS} for k in ('method', 'base')}
            agg[split]['delta'] = {m: ms([round(per[s]['method'][split][m] - per[s]['base'][split][m], 2) for s in MAIN]) for m in METRICS}
            agg[split]['seeds_better_worse_acc'] = [sum(per[s]['method'][split]['accuracy'] > per[s]['base'][split]['accuracy'] for s in MAIN),
                                                    sum(per[s]['method'][split]['accuracy'] < per[s]['base'][split]['accuracy'] for s in MAIN)]
        out['datasets'][ds] = dict(per_seed=per, main_seeds=MAIN, aggregate=agg)
        f = lambda x: f"{x['mean']:.2f}±{x['std']:.2f}"
        md += [f'## {ds}', '', '| 划分 | base ACC | method ACC | base W-F1 | method W-F1 | ΔACC（配对） | ΔW-F1（配对） | method 更好/更差种子 |',
               '|---|---|---|---|---|---|---|---|']
        for split in SPLITS:
            a = agg[split]
            md.append(f"| {split} | {f(a['base']['accuracy'])} | {f(a['method']['accuracy'])} | {f(a['base']['weighted_f1'])} | {f(a['method']['weighted_f1'])} | "
                      f"{a['delta']['accuracy']['mean']:+.2f}±{a['delta']['accuracy']['std']:.2f} | "
                      f"{a['delta']['weighted_f1']['mean']:+.2f}±{a['delta']['weighted_f1']['std']:.2f} | "
                      f"{a['seeds_better_worse_acc'][0]}/{a['seeds_better_worse_acc'][1]} |")
        md += ['', '| 种子 | λ / m / β | test ACC base→method | test W-F1 base→method | 修好/改坏 |', '|---|---|---|---|---|']
        for s in MAIN + EXTRA:
            p, r = per[s]['params'], per[s]
            md.append(f"| {s}{'（单列）' if s in EXTRA else ''} | {p['lam']} / {p['m']} / {p['beta']} | "
                      f"{r['base']['test']['accuracy']:.2f}→{r['method']['test']['accuracy']:.2f} | "
                      f"{r['base']['test']['weighted_f1']:.2f}→{r['method']['test']['weighted_f1']:.2f} | {r['fixes_vs_base']}/{r['harms_vs_base']} |")
        md.append('')
    (ROOT / 'results/final.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
    with open(ROOT / 'results/final.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (ROOT / 'results/final.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
