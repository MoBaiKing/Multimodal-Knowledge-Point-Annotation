#!/usr/bin/env python3
"""CoDiVE-KC = 冻结 base + EGMR（样例检索判别；免训练插件）。一个 (数据集, 种子) 一次运行，GPU 推理。

流程（不训练、不修改 base 任何文件）：
 1. 加载 base 冻结 checkpoint（按 complete.json 核验 SHA256）与冻结的 mix 权重/证据强度。
    base 最终分数 f 取 base 自己保存的冻结输出（val_select: selected_val.pt；test: test_predictions.pt），
    同时按 base evaluate 的同一算子重算一遍，记录逐行复现情况；val_calibration 无保存输出，用重算值。
 2. 记忆库 = 全部训练题：文本向量 h（base 题目表示）、图片向量 v（原生视觉 token 均值）；两者减训练集均值后 L2 归一化；
    标签集合（冲突感知：规范化文本+图片完全相同而标签不同的组，组内每条样本对组内全部标签计数）。
 3. 只在 val_select 上为每个变体选超参（网格预先固定，λ=0 即回退 base；排序键 acc → weighted-F1 → 较小 λ），
    冻结后写 frozen_egmr.json；val_calibration 只做独立复核。已有冻结记录时直接复用，不重新选择。
 4. 冻结之后才经 load_test_after_freeze 读取 test，每个冻结变体只评估一次。
变体：base / knn_global / egmr_text / egmr_mm / egmr_mm_noconflict / egmr_mm_permuted（零对照：训练标签随机置换，
保留类别边缘分布、破坏样例-标签对应；用同一网格同样在 val_select 上选择，度量网格选择本身带来的乐观偏差）。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_ROOT = ROOT / 'runs'
BASE = ROOT
SEEDS_CFG = ROOT / 'config'
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE))
from codive import pipeline as P                                        # noqa: E402
from codive import data as D                                            # noqa: E402
from codive.io import read_json, sha256, canonical_text, write_json, save_tensor   # noqa: E402
import egmr as E                                                         # noqa: E402

DEV = 'cuda'
LAMBDAS = (0, .5, 1, 2, 5, 10, 20, 40)
BETAS = (.25, .5, 1.)
GRID = dict(
    knn_global=[dict(lam=l, tau=t, k=k, beta=b) for l in (0, .1, .2, .3, .5) for t in (.02, .05, .1) for k in (8, 32)
                for b in (0., .25, .5, 1.)],
    egmr_text=[dict(lam=l, m=m, beta=0., conflict=True) for l in LAMBDAS for m in (1, 3, 8)],
    egmr_mm=[dict(lam=l, m=m, beta=b, conflict=True) for l in LAMBDAS for m in (1, 3, 8) for b in BETAS],
    egmr_mm_noconflict=[dict(lam=l, m=m, beta=b, conflict=False) for l in LAMBDAS for m in (1, 3, 8) for b in BETAS],
    egmr_mm_permuted=[dict(lam=l, m=m, beta=b, conflict=False, permuted=True) for l in LAMBDAS for m in (1, 3, 8) for b in BETAS],
)
VARIANTS = ['base'] + list(GRID)
CODE = ('egmr.py', 'run_egmr.py')


def config_path(ds, seed):
    if seed in (200408, 200409):
        return BASE / f'config/{ds.lower()}_concept_evidence_seed{seed}.json'
    return SEEDS_CFG / f'{ds.lower()}_concept_evidence_seed{seed}.json'


@torch.no_grad()
def encode(model, cache, cfg, rows, teacher, w, s, batch):
    """文本向量 h、图片向量 v（原生视觉 token 均值）、有图标记；给 teacher 时按 base evaluate 的同一算子重算最终分数。"""
    H = model.backbone.hidden_size
    hs, vs, has, finals = [], [], [], []
    for st in range(0, len(rows), batch):
        b = rows[st:st + batch]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            hq, student, states, qmask = model.text([r['text'] for r in b], tokens=True)
            if teacher is not None:
                base = P.mix(student, teacher[st:st + len(b)].to(DEV), w)
                if s:
                    tokens, mask = cache.batch(b)
                    dq, dv = model.concept_evidence(hq, states, qmask, tokens, mask,
                                                    P.stable_topk(base, min(cfg['candidate_k'], base.shape[-1])))
                    final = base + s * (dq + dv)
                else:
                    final = base
                finals.append(final.float().cpu())
        hs.append(hq.float().cpu())
        for r in b:
            t = cache.row(r)
            has.append(bool(len(t)))
            vs.append(t.float().mean(0) if len(t) else torch.zeros(H))
    out = dict(h=torch.cat(hs), v=torch.stack(vs), has=torch.tensor(has))
    if finals:
        out['f'] = torch.cat(finals)
    return out


def project(enc, mu_h, mu_v):
    """减训练集均值（去各向异性）后 L2 归一化；无图行的图片向量置 0。"""
    return dict(h=F.normalize(enc['h'] - mu_h, dim=-1).to(DEV),
                v=(F.normalize(enc['v'] - mu_v, dim=-1) * enc['has'][:, None]).to(DEV), has=enc['has'].to(DEV))


def apply(variant, p, q, mem, f, cands, num_classes):
    if variant == 'base' or p['lam'] == 0:
        return f.clone()
    sim = E.similarity(q['h'], mem['h'], q['v'], mem['v'], q['has'], mem['has'], p['beta'])
    if variant == 'knn_global':
        return E.knn_logits(f, sim, mem['y'], num_classes, p['k'], p['tau'], p['lam'])
    sets = mem['sets_permuted'] if p.get('permuted') else mem['sets_conflict' if p['conflict'] else 'sets_own']
    e = E.exemplar_scores(sim, sets, cands, p['m'])
    return E.egmr_logits(f, cands, e, p['lam'])


def overlaps(rows, train):
    """与 base 去重审计同一口径：原始 index / 规范化文本 / 图片 SHA256 与训练集重合。"""
    ids = {r['index'] for r in train}
    texts = {canonical_text(r['text']) for r in train}
    imgs = {d for r in train for d in r['image_sha256']}
    text = torch.tensor([r['index'] in ids or canonical_text(r['text']) in texts for r in rows])
    img = torch.tensor([bool(set(r['image_sha256']) & imgs) for r in rows])
    return text, text | img


def report(y, pred, base_pred, enc):
    has = enc['has'].cpu()
    out = dict(all=P.metric(y, pred))
    for name, m in (('image_rows', has), ('no_image_rows', ~has),
                    ('excluding_text_overlap', ~enc['ov_text']), ('excluding_train_overlap', ~enc['ov_any'])):
        out[name] = P.metric(y[m], pred[m]) if bool(m.any()) else None
    out['fixes_vs_base'] = int(((base_pred != y) & (pred == y)).sum())
    out['harms_vs_base'] = int(((base_pred == y) & (pred != y)).sum())
    return out


def predict(variant, p, q, mem, num_classes, k):
    f = q['f'].to(DEV)
    return apply(variant, p, q, mem, f, P.stable_topk(f, k), num_classes).argmax(-1).cpu()


def select(variant, grid, q, mem, num_classes, k):
    f = q['f'].to(DEV)
    cands = P.stable_topk(f, k)
    best_key, best_p, log = None, None, []
    for p in grid:
        m = P.metric(q['y'], apply(variant, p, q, mem, f, cands, num_classes).argmax(-1).cpu())
        log.append(dict(p, accuracy=m['accuracy'], weighted_f1=m['weighted_f1']))
        key = (round(m['accuracy'], 6), round(m['weighted_f1'], 6), -p['lam'])
        if best_key is None or key > best_key:
            best_key, best_p = key, p
    return best_p, log


def load_base_test(run_dir, done, rows):
    if sha256(run_dir / 'test_predictions.pt') != done['artifacts']['test_predictions.pt']:
        raise ValueError('base test 预测文件与完成记录不符')
    out = torch.load(run_dir / 'test_predictions.pt', map_location='cpu', weights_only=False)
    if out['row_ids'] != [r['row_id'] for r in rows] or out['targets'].tolist() != [r['label'] for r in rows]:
        raise ValueError('base test 预测行序/标签与 test 不符')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('需要本人预约内的 GPU')
    ds, seed = a.dataset, a.seed
    out = OUT_ROOT / ds / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'test_results.json').exists():
        print('已完成，跳过', out)
        return
    t0 = time.time()
    torch.set_num_threads(4)
    plan = read_json(ROOT / 'artifacts/retrieval_plan.json')
    code = {n: sha256(HERE / n) for n in CODE}
    if plan['code_sha256'] != code:
        raise ValueError('代码在实验计划冻结后被修改')
    cfg = json.loads(config_path(ds, seed).read_text())
    run_dir = Path(cfg['run_dir'])
    frozen, done = read_json(run_dir / 'frozen.json'), read_json(run_dir / 'complete.json')
    if done['status'] != 'TESTED' or frozen['seed'] != seed or frozen['dataset'] != ds \
            or done['artifacts']['best.pt'] != frozen['checkpoint_sha256'] \
            or sha256(run_dir / 'best.pt') != frozen['checkpoint_sha256'] \
            or sha256(run_dir / 'frozen.json') != done['artifacts']['frozen.json']:
        raise ValueError('base 运行未完成最终测试或冻结文件已变化')
    receipt = read_json(cfg['prepared_manifest'])
    if sha256(cfg['prepared']) != receipt['sha256']:
        raise ValueError('prepared 已变化')
    prepared = torch.load(cfg['prepared'], map_location='cpu', weights_only=False)
    data, ref = prepared['data'], prepared['reference']
    if data['seed'] != seed or data['subject'] != ds:
        raise ValueError('prepared 种子/学科不符')
    cfg = dict(cfg, labels=data['labels'])
    C, K = len(data['labels']), cfg['candidate_k']
    best = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=False)
    model = P.Model(cfg, semantics=best['semantics'])
    model.restore(best)
    model.eval()
    del best
    cache = P.VisionCache(cfg, model, read_json(BASE / 'artifacts/experiment_plan.json')['pretrained'])
    w, s = float(frozen['mix_weight']), float(frozen['visual_scale'])
    train = data['splits']['train']

    # ---- 记忆库（只用训练集）
    raw = encode(model, cache, cfg, train, None, w, s, 16)
    mu_h = raw['h'].mean(0)
    mu_v = raw['v'][raw['has']].mean(0) if bool(raw['has'].any()) else torch.zeros_like(raw['v'][0])
    mem = project(raw, mu_h, mu_v)
    mem['y'] = torch.tensor([r['label'] for r in train]).to(DEV)
    own = F.one_hot(mem['y'], C).bool()
    conf = own.clone()
    groups = {}
    for i, r in enumerate(train):
        groups.setdefault((canonical_text(r['text']), tuple(r['image_sha256'])), []).append(i)
    conflict_groups = [(idx, sorted({train[i]['label'] for i in idx})) for idx in groups.values()]
    conflict_groups = [(idx, labs) for idx, labs in conflict_groups if len(labs) > 1]
    for idx, labs in conflict_groups:
        conf[torch.tensor(idx, device=DEV)[:, None], torch.tensor(labs, device=DEV)[None, :]] = True
    mem['sets_own'], mem['sets_conflict'] = own, conf
    perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
    mem['sets_permuted'] = own[perm.to(own.device)]              # 零对照：第 i 条样例拿第 perm[i] 条的标签
    memory_info = dict(train_rows=len(train), image_rows=int(raw['has'].sum()), conflict_groups=len(conflict_groups),
                       conflict_rows=sum(len(i) for i, _ in conflict_groups))
    del raw

    # ---- 验证集：只在 val_select 上选择并冻结；val_calibration 只做独立复核
    freeze_path = out / 'frozen_egmr.json'
    preds = dict(labels=data['labels'])
    if freeze_path.exists():
        fz = read_json(freeze_path)
        if fz['base_checkpoint_sha256'] != frozen['checkpoint_sha256'] or fz['code_sha256'] != code or fz['seed'] != seed:
            raise ValueError('已有冻结记录与当前 checkpoint/代码不符')
        print('复用已有冻结记录', freeze_path, flush=True)
    else:
        sel, cal = data['val_select_indices'], data['val_calibration_indices']
        val = {}
        for name, idx in (('val_select', sel), ('val_calibration', cal)):
            rows = [data['splits']['val'][i] for i in idx]
            enc = encode(model, cache, cfg, rows, ref['val_logits'][idx], w, s, cfg['eval_batch'])
            enc['y'] = torch.tensor([r['label'] for r in rows])
            enc['ov_text'], enc['ov_any'] = overlaps(rows, train)
            enc['row_ids'] = [r['row_id'] for r in rows]
            enc['source_rows'] = [r['source_row'] for r in rows]
            val[name] = enc
        q = val['val_select']
        base_sel = torch.load(run_dir / 'selected_val.pt', map_location='cpu', weights_only=False)
        if base_sel['source_rows'] != q['source_rows'] or base_sel['targets'].tolist() != q['y'].tolist():
            raise ValueError('val_select 行序/标签与 base 保存的冻结输出不符')
        reproduction = dict(val_select=dict(n=len(q['y']), argmax_agree=int((q['f'].argmax(-1) == base_sel['predictions']).sum()),
                                            max_abs_logit_diff=float((q['f'] - base_sel['logits'].float()).abs().max())))
        q['f'] = base_sel['logits'].float()                         # 以 base 保存的冻结输出为准
        if not torch.equal(q['f'].argmax(-1), base_sel['predictions']):
            raise ValueError('base 保存的 val_select 输出与其预测不一致')
        for enc in val.values():
            enc.update(project(enc, mu_h, mu_v))
        variants, grid_log = {}, {}
        for variant in VARIANTS:
            p, log = select(variant, GRID.get(variant, [dict(lam=0)]), q, mem, C, K)
            grid_log[variant] = log
            variants[variant] = dict(params=p)
        base_pred = {name: enc['f'].argmax(-1) for name, enc in val.items()}
        for variant, v in variants.items():
            for name, enc in val.items():
                pred = predict(variant, v['params'], enc, mem, C, K)
                v[name] = report(enc['y'], pred, base_pred[name], enc)
                preds.setdefault(name, dict(targets=enc['y'], row_ids=enc['row_ids']))[variant] = pred
        fz = dict(frozen=True, dataset=ds, seed=seed, base_run=str(run_dir), base_checkpoint_sha256=frozen['checkpoint_sha256'],
                  mix_weight=w, visual_scale=s, candidate_k=K, code_sha256=code, memory=memory_info,
                  reproduce_base=reproduction, selection='val_select only: accuracy → weighted_f1 → smaller λ → grid order',
                  representation='h = base 题目表示，v = 原生视觉 token 均值；均减训练集均值后 L2 归一化',
                  variants=variants, test_policy='frozen before test is loaded; each variant evaluated once', frozen_at=P.now())
        write_json(freeze_path, fz)
        write_json(out / 'grid_val_select.json', grid_log)
        del val

    # ---- 冻结后一次性 test
    test = D.load_test_after_freeze(ds, run_dir / 'frozen.json', seed=seed)
    trows = test['rows']
    base_test = load_base_test(run_dir, done, trows)
    te = encode(model, cache, cfg, trows, P.teacher_test(cfg, trows, prepared), w, s, cfg['eval_batch'])
    y = torch.tensor([r['label'] for r in trows])
    te['ov_text'], te['ov_any'] = overlaps(trows, train)
    reproduction = dict(n=len(trows), argmax_agree=int((te['f'].argmax(-1) == base_test['predictions']).sum()),
                        max_abs_logit_diff=float((te['f'] - base_test['logits'].float()).abs().max()))
    te['f'] = base_test['logits'].float()
    te.update(project(te, mu_h, mu_v))
    base_pred = te['f'].argmax(-1)
    if not torch.equal(base_pred, base_test['predictions']):
        raise ValueError('base 保存的 test 输出与其预测不一致')
    base_official = read_json(run_dir / 'test_metrics.json')
    res = dict(dataset=ds, seed=seed, split='test', tested_at=P.now(), frozen_egmr_sha256=sha256(freeze_path),
               base_checkpoint_sha256=frozen['checkpoint_sha256'], reproduce_base_test=reproduction,
               train_overlap_rows=dict(text=int(te['ov_text'].sum()), any=int(te['ov_any'].sum()), n=len(trows)),
               base_official=dict(accuracy=base_official['accuracy'], weighted_f1=base_official['weighted_f1'],
                                macro_f1=base_official['macro_f1']), variants={})
    preds['test'] = dict(targets=y, row_ids=[r['row_id'] for r in trows])
    for variant, v in fz['variants'].items():
        pred = predict(variant, v['params'], te, mem, C, K)
        res['variants'][variant] = dict(params=v['params'], **report(y, pred, base_pred, te))
        preds['test'][variant] = pred
    if res['variants']['base']['all']['accuracy'] != base_official['accuracy']:
        raise ValueError('base 基线与官方 test 指标不一致')
    res['seconds'] = round(time.time() - t0, 1)
    save_tensor(out / 'predictions.pt', preds)
    write_json(out / 'test_results.json', res)
    print('DONE', out, json.dumps({k: round(100 * v['all']['accuracy'], 2) for k, v in res['variants'].items()}), flush=True)


if __name__ == '__main__':
    main()
