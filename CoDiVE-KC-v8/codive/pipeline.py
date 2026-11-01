"""Reservation-interruptible LoRA training, frozen selection and GPU final test."""
from __future__ import annotations

import fcntl
import gc
import json
import math
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .backbone import QwenBackbone, BACKBONE_VERSION
from .data import load_test_after_freeze, RENDERER_VERSION
from .heads import ConceptHead, PairEvidenceHead, classification_loss, V8_LOSS
from .io import now, read_json, write_json, save_tensor, sha256, fingerprint, seed_all, stable_topk, canonical_text

ROOT = Path(__file__).resolve().parents[1]
STOP = False


class Paused(Exception):
    pass


def on_signal(*_):
    global STOP
    STOP = True


def check_stop():
    if STOP:
        raise Paused('Checkpoint requested by latest live reservation controller')


def metric(y, p):
    y = torch.as_tensor(y).cpu().long(); p = torch.as_tensor(p).cpu().long()
    n = len(y)
    if not n:
        return dict(n=0, accuracy=None, weighted_f1=None, macro_f1=None)
    classes = torch.unique(torch.cat((y, p))).tolist()
    scores, weighted = [], 0.
    for c in classes:
        tp = int(((p == c) & (y == c)).sum()); fp = int(((p == c) & (y != c)).sum()); fn = int(((p != c) & (y == c)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
        scores.append(f1); weighted += int((y == c).sum()) * f1
    return dict(n=n, accuracy=int((p == y).sum()) / n, weighted_f1=weighted / n, macro_f1=sum(scores) / len(scores))


def mix(student, reference, weight):
    # The teacher was produced by grouped OOF for training, frozen full model for val/test.
    return ((1. - weight) * reference.float().softmax(-1) + weight * student.float().softmax(-1)).clamp_min(1e-10).log()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


class Model(nn.Module):
    def __init__(self, cfg, semantics=None):
        super().__init__()
        self.backbone = QwenBackbone(cfg).cuda()
        if semantics is None:
            chunks = []
            for start in range(0, len(cfg['labels']), 12):
                check_stop()
                chunks.append(self.backbone.encode_concepts(cfg['labels'][start:start+12]).detach().cpu())
            semantics = torch.cat(chunks)
        self.register_buffer('semantics', semantics.float().cuda())
        h, c = self.semantics.shape[1], len(self.semantics)
        self.concept = ConceptHead(self.semantics).cuda()
        if cfg['variant'] == 'concept_evidence':
            # v8: concept-pair evidence read from the question's own token states on every
            # row, plus the same readout over image tokens on image rows. Open gate (0.5)
            # and a wider residual bound than v7, whose gate collapsed to ~0.01.
            self.evidence = PairEvidenceHead(h, c, max_residual=3.0, gate_bias=0.0).cuda()
            self.evidence_image = PairEvidenceHead(h, c, max_residual=3.0, gate_bias=0.0).cuda()
        else:
            self.evidence = PairEvidenceHead(h, c).cuda()
            if cfg['variant'] == 'text_control':
                self.evidence.requires_grad_(False)

    def trainables(self):
        return {name: parameter.detach().cpu() for name, parameter in self.named_parameters() if parameter.requires_grad}

    def restore(self, checkpoint):
        current = dict(self.named_parameters())
        expected = {name for name, value in current.items() if value.requires_grad}
        if set(checkpoint['trainables']) != expected:
            raise ValueError('Trainable parameter identity changed')
        with torch.no_grad():
            for name, value in checkpoint['trainables'].items():
                current[name].copy_(value.to(current[name].device, current[name].dtype))

    def text(self, texts, tokens=False):
        if tokens:
            hq, states, mask = self.backbone.encode_question_states(texts)
        else:
            hq = self.backbone.encode_questions(texts)
        with torch.autocast('cuda', enabled=False):
            base = self.concept(hq.float())
        return (hq, base, states, mask) if tokens else (hq, base)

    def concept_evidence(self, hq, states, mask, image_tokens, image_mask, candidates):
        """v8 residuals over the candidates: (question-token evidence, image-token evidence)."""
        with torch.autocast('cuda', enabled=False):
            delta_q, _ = self.evidence(hq.float(), self.semantics, states, mask, candidates)
            if image_mask is not None and bool(image_mask.any()):
                delta_v, _ = self.evidence_image(hq.float(), self.semantics, image_tokens, image_mask, candidates)
            else:
                delta_v = torch.zeros_like(delta_q)
        return delta_q, delta_v


class VisionCache:
    def __init__(self, cfg, model, base_identity):
        self.cfg, self.model = cfg, model
        self.identity = fingerprint(dict(base=base_identity, backbone=BACKBONE_VERSION,
            renderer=RENDERER_VERSION, max_pixels=cfg['max_pixels'], min_pixels=cfg.get('min_pixels', 4*28*28),
            implementation=sha256(ROOT/'codive/backbone.py')))
        self.folder = ROOT/'cache/native_vision'/self.identity
        self.memory = OrderedDict()
        self.hits = 0; self.built = 0

    def image(self, path, digest):
        if digest in self.memory:
            self.hits += 1; self.memory.move_to_end(digest)
            return self.memory[digest]
        self.folder.mkdir(parents=True, exist_ok=True)
        target = self.folder/(digest+'.pt')
        with (self.folder/(digest+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if target.exists():
                d = torch.load(target, map_location='cpu', weights_only=False)
                if d['identity'] != self.identity or d['image_sha256'] != digest:
                    raise ValueError('Frozen visual feature provenance changed')
                value = d['tokens']
                self.hits += 1
            else:
                if sha256(path) != digest:
                    raise ValueError('Raw image changed after audit: '+path)
                value, xy = self.model.backbone.extract_image(path)
                if value.ndim != 2 or value.shape[1] != self.model.backbone.hidden_size or not torch.isfinite(value).all():
                    raise ValueError('Invalid native image features')
                save_tensor(target, dict(identity=self.identity, image_sha256=digest, tokens=value.to(torch.bfloat16), xy=xy))
                value = value.to(torch.bfloat16); self.built += 1
        self.memory[digest] = value
        while len(self.memory) > 96:
            self.memory.popitem(last=False)
        return value

    def row(self, record, epoch=0, training=False):
        paths, hashes = record['image_paths'], record['image_sha256']
        if not paths:
            return torch.empty(0, self.model.backbone.hidden_size, dtype=torch.bfloat16)
        count = min(self.cfg['max_images'], len(paths))
        # Deterministic image-slot coverage during training; no labels select images.
        offset = (epoch * count) % len(paths) if training else 0
        ids = [(offset+i) % len(paths) for i in range(count)]
        return torch.cat([self.image(paths[i], hashes[i]) for i in ids])

    def batch(self, records, epoch=0, training=False):
        values = [self.row(row, epoch, training) for row in records]
        length = max(1, max(len(value) for value in values))
        tokens = torch.zeros(len(values), length, self.model.backbone.hidden_size, device='cuda', dtype=torch.bfloat16)
        mask = torch.zeros(len(values), length, device='cuda', dtype=torch.bool)
        for i, value in enumerate(values):
            if len(value):
                tokens[i, :len(value)] = value.cuda(); mask[i, :len(value)] = True
        return tokens, mask


def evaluate(model, cfg, rows, reference, cache, settings, status=None):
    model.eval()
    predictions = {pair: [] for pair in settings}; anchors = {pair: [] for pair in settings}; logits = {pair: [] for pair in settings}
    question_only = {pair: [] for pair in settings}
    for start in range(0, len(rows), cfg['eval_batch']):
        check_stop(); batch = rows[start:start+cfg['eval_batch']]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            if cfg['variant'] == 'concept_evidence':
                hq, student, states, qmask = model.text([r['text'] for r in batch], tokens=True)
            else:
                hq, student = model.text([r['text'] for r in batch])
            teacher = reference[start:start+len(batch)].cuda()
            if cfg['variant'] in ('pair_evidence', 'concept_evidence'):
                tokens, mask = cache.batch(batch)
            else:
                tokens = mask = None
            for weight, scale in settings:
                base = mix(student, teacher, weight)
                if cfg['variant'] == 'concept_evidence':
                    # Question evidence applies to every row; image evidence is exactly 0 without images.
                    if scale:
                        delta_q, delta_v = model.concept_evidence(hq, states, qmask, tokens, mask,
                            stable_topk(base, min(cfg['candidate_k'], base.shape[-1])))
                        fused = base + scale * (delta_q + delta_v)
                        no_image = (base + scale * delta_q).argmax(-1)
                    else:
                        fused = base; no_image = base.argmax(-1)
                    pred = fused.argmax(-1)
                    question_only[(weight, scale)].append(no_image.cpu())
                elif scale and mask is not None and bool(mask.any()):
                    with torch.autocast('cuda', enabled=False):
                        delta, _ = model.evidence(hq.float(), model.semantics, tokens, mask, stable_topk(base, min(cfg['candidate_k'], base.shape[-1])))
                    fused = base + scale * delta
                    # No-image rows return original full-class argmax, including exact ties.
                    pred = torch.where(mask.any(-1), fused.argmax(-1), base.argmax(-1))
                    if not torch.equal(pred[~mask.any(-1)], base.argmax(-1)[~mask.any(-1)]):
                        raise ValueError('No-image fallback invariant failed')
                else:
                    fused, pred = base, base.argmax(-1)
                pair = (weight, scale)
                logits[pair].append(fused.float().cpu()); predictions[pair].append(pred.cpu()); anchors[pair].append(base.argmax(-1).cpu())
        if status and start % 256 == 0:
            status('VALIDATING', 'validation_select', rows_done=start+len(batch), total_rows=len(rows))
    y = torch.tensor([r['label'] for r in rows])
    result = {}
    for key in predictions:
        p, a = torch.cat(predictions[key]), torch.cat(anchors[key])
        result[key] = dict(**metric(y, p), predictions=p, anchor=a, logits=torch.cat(logits[key]), targets=y)
        if question_only[key]:
            result[key]['question_only_predictions'] = torch.cat(question_only[key])
    return result


def teacher_test(cfg, rows, prepared):
    # v8: the seed-matched full reference recorded in prepared provenance (v7 hard-coded
    # the seed200408 folder, which failed every other seed's final-test guard).
    full = next(x for x in prepared['reference']['provenance'] if x['stage'] == 'full')
    folder = Path(full['checkpoint_path']).resolve().parent
    done = read_json(folder/'complete.json')
    if done.get('seed', cfg['seed']) != cfg['seed'] or done.get('stage') != 'text/macbert/full':
        raise ValueError('Teacher folder is not the seed-matched full MacBERT stage')
    if sha256(folder/'test.pt') != done['artifacts']['test.pt']:
        raise ValueError('Frozen original text test cache hash changed')
    if full['checkpoint_sha256'] != done['artifacts']['best.pt']:
        raise ValueError('Teacher final test uses different checkpoint')
    payload = torch.load(folder/'test.pt', map_location='cpu', weights_only=False)
    if payload['checkpoint_sha256'] != full['checkpoint_sha256'] or payload['indices'] != list(range(len(rows))):
        raise ValueError('Teacher final test checkpoint/row order mismatch')
    if payload['targets'].tolist() != [r['label'] for r in rows]:
        raise ValueError('Teacher final test targets changed')
    return payload['logits'].float()


def run(cfg, device='cuda:0'):
    if device != 'cuda:0' or not torch.cuda.is_available():
        raise RuntimeError('Reservation-authorized GPU is required for training/validation/final test')
    torch.set_num_threads(4); seed_all(cfg['seed'])
    run_dir = Path(cfg['run_dir']); run_dir.mkdir(parents=True, exist_ok=True)
    stage = 'verify_protocol'
    def status(state, new_stage=None, **extra):
        nonlocal stage
        if new_stage: stage = new_stage
        resumable = extra.pop('resumable', True)
        d = dict(status=state, stage=stage, dataset=cfg['dataset'], variant=cfg['variant'], seed=cfg['seed'],
                 updated_at=now(), resumable=resumable, **extra)
        write_json(run_dir/'status.json', d); print(json.dumps(d, ensure_ascii=False), flush=True)
    try:
        status('RUNNING', 'verify_protocol')
        receipt = read_json(cfg['prepared_manifest'])
        if sha256(cfg['prepared']) != receipt['sha256']:
            raise ValueError('Verified prepared data changed')
        for path, guard in receipt['manifest']['verified_source_file_guards'].items():
            st = Path(path).stat()
            if st.st_size != guard['size'] or st.st_mtime_ns != guard['mtime_ns']:
                raise ValueError('Previously verified original source changed: '+path)
        prepared = torch.load(cfg['prepared'], map_location='cpu', weights_only=False)
        data, reference = prepared['data'], prepared['reference']
        if data['subject'] != cfg['dataset'] or data['seed'] != cfg['seed']:
            raise ValueError('Prepared subject/seed mismatch')
        cfg = dict(cfg, labels=data['labels'])
        plan = read_json(ROOT/'artifacts/experiment_plan.json')
        sources = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT/'codive').glob('*.py'))}
        if sources != plan['learner_sources']:
            raise ValueError('Learner source changed after experiment freeze')
        signature = fingerprint(dict(config=cfg, prepared_sha256=receipt['sha256'], learner=sources, pretrained=plan['pretrained']))
        if (run_dir/'provenance.json').exists():
            if read_json(run_dir/'provenance.json')['signature'] != signature:
                raise ValueError('Immutable training configuration changed; use a new run directory')
        else:
            write_json(run_dir/'provenance.json', dict(signature=signature, config=cfg, prepared_sha256=receipt['sha256'],
                       learner_sources=sources, pretrained=plan['pretrained'], created_at=now()))
        if (run_dir/'complete.json').exists():
            done = read_json(run_dir/'complete.json')
            if done['signature'] != signature or done['status'] != 'TESTED':
                raise ValueError('Final completion signature/state changed')
            for name, digest in done['artifacts'].items():
                if sha256(run_dir/name) != digest:
                    raise ValueError('Final completion artifact changed: '+name)
            status('TESTED', 'complete', reason='Verified completed GPU final test reused', resumable=False)
            return 0
        check_stop()
        previous = torch.load(run_dir/'last.pt', map_location='cpu', weights_only=False) if (run_dir/'last.pt').exists() else None
        frozen = read_json(run_dir/'frozen.json') if (run_dir/'frozen.json').exists() else None
        initial = (torch.load(run_dir/'best.pt', map_location='cpu', weights_only=False) if frozen else previous)
        if initial and initial['signature'] != signature:
            raise ValueError('Resume checkpoint signature changed')
        status('TRAINED' if frozen else 'RUNNING', 'load_qwen25vl3b_lora')
        model = Model(cfg, semantics=initial['semantics'] if initial else None)
        cache = VisionCache(cfg, model, plan['pretrained'])
        if initial:
            model.restore(initial)
        train = data['splits']['train']
        select = data['val_select_indices']; calibration = data['val_calibration_indices']
        val_rows = [data['splits']['val'][i] for i in select]
        val_reference = reference['val_logits'][select]
        settings = [(float(w), float(s)) for w in cfg['mix_grid'] for s in cfg['visual_scale_grid']]
        if not frozen:
            groups = []
            for is_backbone, lr in [(True, cfg['backbone_lr']), (False, cfg['head_lr'])]:
                parameters = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith('backbone.') == is_backbone]
                groups.append(dict(params=parameters, lr=lr, initial_lr=lr))
            optimizer = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
            state = dict(epoch=0, next_batch=0, optimizer_steps=0, best_key=[-1., -1., 0., 0.], bad_epochs=0)
            if previous:
                optimizer.load_state_dict(previous['optimizer']); state = previous['state']
                random.setstate(previous['python_rng']); np.random.set_state(previous['numpy_rng'])
                torch.set_rng_state(previous['torch_rng']); torch.cuda.set_rng_state(previous['cuda_rng'])
                del previous
            parameter_counts = dict(trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                backbone_lora=sum(p.numel() for p in model.backbone.parameters() if p.requires_grad),
                evidence=sum(p.numel() for p in model.evidence.parameters() if p.requires_grad),
                frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad))
            write_json(run_dir/'parameter_counts.json', parameter_counts)
            n_batches = math.ceil(len(train)/cfg['microbatch'])
            total_steps = cfg['epochs'] * math.ceil(n_batches/cfg['accumulation'])
            warmup = max(1, round(total_steps*cfg['warmup_fraction']))
            image_ids = [i for i, row in enumerate(train) if row['image_available']]
            def checkpoint():
                save_tensor(run_dir/'last.pt', dict(signature=signature, semantics=model.semantics.detach().cpu(),
                    trainables=model.trainables(), optimizer=cpu_tree(optimizer.state_dict()), state=state.copy(),
                    python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state()))
            for epoch in range(state['epoch'], cfg['epochs']):
                order = torch.randperm(len(train), generator=torch.Generator().manual_seed(cfg['seed']+epoch)).tolist()
                model.train(); optimizer.zero_grad(set_to_none=True)
                accumulated = 0; loss_sum = 0.; since = time.time()
                for batch_number in range(state['next_batch'], n_batches):
                    ids = order[batch_number*cfg['microbatch']:(batch_number+1)*cfg['microbatch']]
                    batch = [train[i] for i in ids]
                    y = reference['train_targets'][ids].cuda()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        if cfg['variant'] == 'concept_evidence':
                            hq, student, states, qmask = model.text([r['text'] for r in batch], tokens=True)
                        else:
                            hq, student = model.text([r['text'] for r in batch])
                        base = mix(student, reference['train_logits'][ids].cuda(), cfg['train_mix_eta'])
                        mismatch = None
                        loss_weights = None
                        if cfg['variant'] == 'concept_evidence':
                            tokens, mask = cache.batch(batch, epoch=epoch, training=True)
                            candidates = stable_topk(base.detach(), min(cfg['candidate_k'], base.shape[-1]))
                            delta_q, delta_v = model.concept_evidence(hq, states, qmask, tokens, mask, candidates)
                            fused = base + delta_q + delta_v
                            loss_weights = V8_LOSS
                        elif cfg['variant'] == 'pair_evidence':
                            tokens, mask = cache.batch(batch, epoch=epoch, training=True)
                            candidates = stable_topk(base.detach(), min(cfg['candidate_k'], base.shape[-1]))
                            with torch.autocast('cuda', enabled=False):
                                delta, aux = model.evidence(hq.float(), model.semantics, tokens, mask, candidates)
                            # Independent mismatched image regularization, half of optimizer windows.
                            if batch_number % 2 == 0 and bool(mask.any()):
                                donors = []
                                for record in batch:
                                    for _ in range(100):
                                        donor = train[random.choice(image_ids)]
                                        if donor['group'] != record['group'] and donor['index'] != record['index'] and not set(donor['image_sha256']) & set(record['image_sha256']):
                                            break
                                    else:
                                        raise ValueError('Cannot find an independent mismatch image donor')
                                    donors.append(donor)
                                mt, mm = cache.batch(donors, epoch=epoch, training=True)
                                with torch.autocast('cuda', enabled=False):
                                    mismatch, _ = model.evidence(hq.detach().float(), model.semantics, mt, mm, candidates)
                            fused = base + delta
                        else:
                            fused = base
                        loss, parts = classification_loss(base, fused, y, mismatch, weights=loss_weights, return_parts=True)
                        loss = loss + .5 * F.cross_entropy(student.float(), y)
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite training loss')
                    (loss/cfg['accumulation']).backward(); accumulated += 1; loss_sum += float(loss.detach())
                    boundary = accumulated == cfg['accumulation'] or batch_number+1 == n_batches
                    if boundary:
                        # Correct the final short accumulation to preserve microbatch weight.
                        if accumulated != cfg['accumulation']:
                            for p in model.parameters():
                                if p.grad is not None: p.grad.mul_(cfg['accumulation']/accumulated)
                        grad_norm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                        if not torch.isfinite(grad_norm):
                            raise ValueError('Non-finite trainable gradient')
                        step = state['optimizer_steps']
                        factor = (step+1)/warmup if step < warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(total_steps-warmup, 1)))
                        for group in optimizer.param_groups: group['lr'] = group['initial_lr']*factor
                        optimizer.step(); optimizer.zero_grad(set_to_none=True); accumulated = 0
                        state.update(epoch=epoch, next_batch=batch_number+1, optimizer_steps=step+1)
                        if step == 0 or (step+1) % 10 == 0:
                            status('RUNNING', 'lora_and_concept_evidence_training', epoch=epoch+1,
                                batch=batch_number+1, batches=n_batches, optimizer_steps=step+1, total_optimizer_steps=total_steps,
                                loss=float(loss.detach()), gradient_norm=float(grad_norm), visual_cache_built=cache.built,
                                visual_cache_hits=cache.hits, elapsed_epoch_seconds=round(time.time()-since, 1), **parameter_counts)
                        if STOP or (step+1) % 50 == 0: checkpoint()
                        if STOP: check_stop()
                # Keep this epoch pending until its selection evaluation is durable.
                # A reservation interruption during validation must resume it.
                state.update(epoch=epoch, next_batch=n_batches)
                checkpoint()
                values = evaluate(model, cfg, val_rows, val_reference, cache, settings, status)
                ranked = []
                for (weight, scale), result in values.items():
                    key = (result['accuracy'], result['weighted_f1'], -scale, -abs(weight-.5))
                    ranked.append((key, weight, scale, result))
                best = max(ranked, key=lambda item: item[0]); key, weight, scale, result = best
                write_json(run_dir/f'val_epoch{epoch+1}.json', dict(epoch=epoch+1, split='val_select',
                    grid=[dict(mix_weight=w, visual_scale=s, **{k:v for k,v in r.items() if k in ('n','accuracy','weighted_f1','macro_f1')}) for (_,w,s,r) in ranked],
                    selected=dict(mix_weight=weight, visual_scale=scale, **{k:result[k] for k in ('n','accuracy','weighted_f1','macro_f1')})))
                if tuple(key) > tuple(state['best_key']):
                    state.update(best_key=list(key), bad_epochs=0)
                    save_tensor(run_dir/'best.pt', dict(signature=signature, semantics=model.semantics.detach().cpu(),
                        trainables=model.trainables(), epoch=epoch+1, mix_weight=weight, visual_scale=scale,
                        validation_select={k:result[k] for k in ('n','accuracy','weighted_f1','macro_f1')}))
                    save_tensor(run_dir/'selected_val.pt', dict(signature=signature, source_rows=select,
                        logits=result['logits'], predictions=result['predictions'], targets=result['targets'], baseline=result['anchor']))
                else:
                    state['bad_epochs'] += 1
                state.update(epoch=epoch+1, next_batch=0)
                checkpoint()
                if state['bad_epochs'] >= cfg['patience']: break
            frozen_best = torch.load(run_dir/'best.pt', map_location='cpu', weights_only=False)
            frozen = dict(frozen=True, signature=signature, dataset=cfg['dataset'], variant=cfg['variant'], seed=cfg['seed'],
                checkpoint_path=str(run_dir/'best.pt'), checkpoint_sha256=sha256(run_dir/'best.pt'),
                mix_weight=frozen_best['mix_weight'], visual_scale=frozen_best['visual_scale'], selected_epoch=frozen_best['epoch'],
                validation_select=frozen_best['validation_select'], selection_split='val_select',
                selection_metric='accuracy_then_weighted_f1_then_lower_visual_scale', val_roles_sha256=data['manifest']['val_roles_sha256'],
                test_policy='No test used before this immutable final freeze', frozen_at=now())
            write_json(run_dir/'frozen.json', frozen)
            model.restore(frozen_best); del optimizer, frozen_best
            gc.collect(); torch.cuda.empty_cache()
        else:
            if frozen['signature'] != signature or sha256(run_dir/'best.pt') != frozen['checkpoint_sha256']:
                raise ValueError('Frozen checkpoint changed')
        status('TRAINED', 'frozen_checkpoint_gpu_calibration_and_final_test', reason='Training finished; final GPU test not yet complete')
        selected = [(float(frozen['mix_weight']), float(frozen['visual_scale']))]
        cal_path = run_dir/'val_calibration.json'
        if not cal_path.exists():
            cal_rows = [data['splits']['val'][i] for i in calibration]
            result = evaluate(model, cfg, cal_rows, reference['val_logits'][calibration], cache, selected)[selected[0]]
            scores = 1-result['logits'].softmax(-1).gather(1, result['targets'][:, None]).squeeze(1)
            rank = min(len(scores), math.ceil((len(scores)+1)*(1-cfg['alpha'])))
            q = float(scores.sort().values[rank-1])
            write_json(cal_path, dict(signature=signature, split='val_calibration', alpha=cfg['alpha'], q=q,
                n=len(scores), policy='LAC diagnostic sets only; does not alter frozen argmax or checkpoint',
                frozen_checkpoint_sha256=frozen['checkpoint_sha256'], frozen_at=now()))
        check_stop()
        test = load_test_after_freeze(cfg['dataset'], run_dir/'frozen.json', seed=cfg['seed'])
        test_rows = test['rows']; test_reference = teacher_test(cfg, test_rows, prepared)
        status('TRAINED', 'frozen_gpu_final_test', total_rows=len(test_rows))
        result = evaluate(model, cfg, test_rows, test_reference, cache, selected)[selected[0]]
        y, pred, anchor = result['targets'], result['predictions'], result['anchor']
        has = torch.tensor([r['image_available'] for r in test_rows])
        question_only = result.get('question_only_predictions')
        if question_only is None and not torch.equal(pred[~has], anchor[~has]):
            raise ValueError('Final no-image fallback not exact')
        if question_only is not None and not torch.equal(pred[~has], question_only[~has]):
            raise ValueError('Image evidence changed a row without images')
        calibration_record = read_json(cal_path)
        if calibration_record['signature'] != signature or calibration_record['frozen_checkpoint_sha256'] != frozen['checkpoint_sha256']:
            raise ValueError('Saved calibration belongs to different frozen model')
        candidates = result['logits'].softmax(-1) >= 1-calibration_record['q']
        candidates[torch.arange(len(pred)), pred] = True
        test_metrics = dict(dataset=cfg['dataset'], variant=cfg['variant'], seed=cfg['seed'], signature=signature,
            split='test', **metric(y, pred), f1_average='weighted', text_only=metric(y, anchor),
            fix_count=int(((anchor != y) & (pred == y)).sum()), harm_count=int(((anchor == y) & (pred != y)).sum()),
            image_rows=int(has.sum()), no_image_fallback_exact=question_only is None, device='cuda:0', tested_at=now(),
            frozen_checkpoint_sha256=frozen['checkpoint_sha256'], mix_weight=frozen['mix_weight'], visual_scale=frozen['visual_scale'],
            conformal_candidate_recall=float(candidates.gather(1, y[:,None]).float().mean()),
            conformal_mean_candidate_count=float(candidates.sum(-1).float().mean()),
            interpretation='Final frozen test; attention is a proposal, not a verified evidence box')
        train_texts = {canonical_text(row['text']) for row in train}
        train_ids = {row['index'] for row in train}
        train_images = {digest for row in train for digest in row['image_sha256']}
        clean = torch.tensor([row['index'] not in train_ids and canonical_text(row['text']) not in train_texts
                              and not set(row['image_sha256']) & train_images for row in test_rows])
        if question_only is not None:
            # Frozen-model ablation for reporting only: same checkpoint and settings without the image head.
            test_metrics['question_evidence_only'] = metric(y, question_only)
            test_metrics['image_head_fix'] = int(((question_only != y) & (pred == y)).sum())
            test_metrics['image_head_harm'] = int(((question_only == y) & (pred != y)).sum())
        test_metrics['train_overlap_rows'] = int((~clean).sum())
        test_metrics['excluding_train_overlap'] = metric(y[clean], pred[clean])
        save_tensor(run_dir/'test_predictions.pt', dict(signature=signature, targets=y, predictions=pred,
            logits=result['logits'], baseline_predictions=anchor, source_rows=[r['source_row'] for r in test_rows], row_ids=[r['row_id'] for r in test_rows]))
        write_json(run_dir/'test_metrics.json', test_metrics)
        artifacts = ['best.pt', 'frozen.json', 'test_predictions.pt', 'test_metrics.json', 'val_calibration.json']
        write_json(run_dir/'complete.json', dict(signature=signature, status='TESTED', completed_at=now(),
            artifacts={name:sha256(run_dir/name) for name in artifacts}))
        for name, digest in read_json(run_dir/'complete.json')['artifacts'].items():
            if sha256(run_dir/name) != digest: raise ValueError('Final artifact verification failed: '+name)
        # External results export is CPU bookkeeping over these saved predictions.
        import subprocess, sys
        subprocess.run([sys.executable, str(ROOT/'scripts/export_results.py')], check=True)
        status('TESTED', 'complete', reason='Frozen GPU final test, checkpoint, predictions and weighted-F1 results verified', resumable=False)
        return 0
    except Paused as exc:
        status('TRAINED' if (run_dir/'frozen.json').exists() else 'PAUSED', reason=str(exc))
        return 75
    except Exception as exc:
        import traceback
        status('FAILED', reason=str(exc)); traceback.print_exc()
        return 1
