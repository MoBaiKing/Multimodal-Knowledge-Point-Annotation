"""CPU only: does the image description lexically favour the gold concept over the text top-1?

Train image rows with OOF teacher logits (no val tuning, no test). For rows the
teacher gets wrong with gold in its top-5, compare the fraction of each concept
name's character bigrams that appear in the description, gold vs teacher top-1;
a donor row's description is the shuffled control.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/data/dyl/CoDiVE-KC/stage0')
sys.path.insert(0, '/data/dyl/CoDiVE-KC/stage0b')
import stage0_probe as s0  # noqa: E402
import stage0b_joint as s0b  # noqa: E402

HERE = Path('/data/dyl/CoDiVE-KC/stage0c')


def grams(text):
    text = ''.join(ch for ch in str(text) if not ch.isspace())
    return {text[i:i+2] for i in range(len(text) - 1)}


def cover(desc, name):
    return len(desc & name) / max(1, len(name))


out = {}
for ds in ('Math', 'Biology'):
    d, rows, _ = s0.load(ds)
    names = [grams(n) for n in d['label_names']]
    caps = {}
    for f in sorted(HERE.glob(f'captions_{ds}_shard*.json')):
        caps.update(json.loads(f.read_text()))
    part = rows['train']
    donor = s0b.donors(part)
    desc = [grams(''.join(caps.get(h, '') for h in x['image_sha256'][:s0.MAX_IMAGES])) for x in part]
    stats = {k: dict(gold_higher=0, pred_higher=0, tie=0) for k in ('real', 'donor')}
    for k, x in enumerate(part):
        top = x['base'].float().topk(s0.K).indices.tolist()
        y = x['label']
        if top[0] == y or y not in top:
            continue
        for key, g in (('real', desc[k]), ('donor', desc[donor.get(k, k)])):
            a, b = cover(g, names[y]), cover(g, names[top[0]])
            stats[key]['gold_higher' if a > b else 'pred_higher' if a < b else 'tie'] += 1
    out[ds] = dict(rows=sum(stats['real'].values()), **stats)
    print(ds, json.dumps(out[ds], ensure_ascii=False))
(HERE/'caption_lexical_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
