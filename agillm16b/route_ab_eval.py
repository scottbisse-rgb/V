#!/usr/bin/env python3
"""Offline A/B for the upcycled AGILLM-16B: standard top-2 routing vs group-limited top-2.

The 16B was seeded from the 8B (32 experts/layer) by copying every expert (copy 0 exact, copy 1 for all, copy 2 for the
4 busiest by router norm) -> 68 experts/layer. Copies of one source expert start with near-identical router rows, so
plain top-2 tends to put BOTH slots on two copies of the token's first-choice expert: the seeded model computes the
8B's TOP-1 output, not its top-2 (seed_8b_to_16b.py says so in its own docstring).

Group-limited routing: pick the k best DISTINCT source groups (a group's score is its best copy's probability), then
the best copy inside each chosen group. At the seed this is exactly the 8B's top-2; copies still compete inside their
group, so they keep specialising.

This script never touches the live run. It:
  * hardlinks RUN/ckpt.good into WORK/snap (so a checkpoint rotation cannot change files mid-read), loads it on CPU,
    then removes the links at once;
  * imports a private copy of the trainer with one line changed (the router's topk goes through a ROUTE hook);
  * scores the same validation rows the trainer uses with both routings, paired, and reports how often standard top-2
    co-selects two copies of one source expert.

Run (on the 16B box, CPU only, niced):
  nice -n 19 env THREADS=16 python3 route_ab_eval.py --nets 2 --rows 16
"""
import argparse
import json
import os
import re
import shutil
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU only: never take memory or time on the trainer's GPU
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np  # noqa: E402
import torch  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--src', default='/workspace/agillm16b/code/bintf2_16b.py', help='the live trainer (only read)')
p.add_argument('--run', default='/workspace/agillm16b_run', help='the live run directory (only read)')
p.add_argument('--ckpt', default='ckpt.good', help='checkpoint directory inside --run')
p.add_argument('--ckpt-dir', default='', help='score this checkpoint directory instead (e.g. the 8B seed source)')
p.add_argument('--compare', default='group', choices=['group', 'top1'],
               help='group: standard top-2 vs group-limited top-2 (needs the seed copy map); top1: top-2 vs top-1 '
                    '(gate 1 on the first choice), what plain top-2 over exact copies computes at the seed')
p.add_argument('--seed-log', default='/workspace/agillm16b_seed/seed.log', help='seed_8b_to_16b.py log (copy map)')
p.add_argument('--work', default='/workspace/claude_16b_eval')
p.add_argument('--nets', default='0,1,2')
p.add_argument('--rows', type=int, default=16, help='validation rows (of the trainer\'s 64) to score')
p.add_argument('--bs', type=int, default=2)
p.add_argument('--src-experts', type=int, default=32)
a = p.parse_args()
torch.set_num_threads(int(os.environ.get('THREADS', '16')))
os.makedirs(a.work, exist_ok=True)

# ---------------------------------------------------------------- a private copy of the trainer with a route hook
src = open(a.src).read()
NEEDLE = '            top_p, top_i = probs.topk(cfg.topk, dim=-1)\n'
assert src.count(NEEDLE) == 1, 'router topk line not found exactly once: trainer changed, re-check the hook'
src = src.replace(NEEDLE, '            top_p, top_i = ROUTE(self, probs, cfg.topk)\n')
HOOK = '''
def ROUTE(layer, probs, k):  # route_ab_eval.py: replaced at run time
    return probs.topk(k, dim=-1)


class Layer(nn.Module):'''
assert src.count('\nclass Layer(nn.Module):') == 1
src = src.replace('\nclass Layer(nn.Module):', HOOK, 1)
mod = os.path.join(a.work, 'bintf2_16b_routehook.py')
with open(mod, 'w') as f:
    f.write(src)
sys.path.insert(0, a.work)
import bintf2_16b_routehook as T  # noqa: E402

assert not T.CUDA, 'must run on CPU'

# ---------------------------------------------------------------- the copy map: new expert j -> its source expert
log = open(a.seed_log).read()
busy = {int(m.group(1)): json.loads(m.group(2)) for m in
        re.finditer(r'net(\d): split by \w+; experts with an extra copy per layer (\[\[.*?\]\])', log)}


def group_map(b, l, E):
    Es = a.src_experts
    g = list(range(Es)) + list(range(Es))  # copy 0 in place, then copy 1 of every expert in expert order
    g += sorted(busy[b][l])  # then copy 2 of the busiest, in expert order
    assert len(g) == E, (len(g), E)
    return torch.tensor(g, dtype=torch.long)


# ---------------------------------------------------------------- the two routings
STATS = {}


def route_standard(layer, probs, k):
    top_p, top_i = probs.topk(k, dim=-1)
    g = layer._grp
    s = STATS.setdefault((layer._net, layer._lid), [0, 0, 0.0])
    s[0] += int((g[top_i[:, 0]] == g[top_i[:, 1]]).sum())  # both slots on copies of one source expert
    s[1] += top_i.shape[0]
    s[2] += float(top_p.sum())  # probability mass the two chosen experts hold (before renormalising)
    return top_p, top_i


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from group_route import _group_topk  # noqa: E402  (the same text the trainer patch inserts)


def route_group(layer, probs, k):
    return _group_topk(probs, k, layer._grp)


def route_top1(layer, probs, k):
    top_p, top_i = probs.topk(k, dim=-1)
    return top_p * (torch.arange(k) == 0).to(top_p.dtype), top_i  # renormalised by the caller: gate 1, then 0


# ---------------------------------------------------------------- load the checkpoint from a hardlinked snapshot
ck_live = a.ckpt_dir or os.path.join(a.run, a.ckpt)
snap = os.path.join(a.work, 'snap')
shutil.rmtree(snap, ignore_errors=True)
os.makedirs(snap)
for n in os.listdir(ck_live):
    os.link(os.path.join(ck_live, n), os.path.join(snap, n))
try:
    st = json.load(open(os.path.join(snap, 'state.json')))
    cfg = T.Config(**st['cfg'])
    T.set_vocab(cfg.vocab)
    t = time.time()
    model = T.Model(cfg)
    T.load_ckpt(model, None, snap, None)
    print(f'loaded {ck_live} (steps {st["steps"]}, seen {st.get("seen")}) in {time.time() - t:.0f}s', flush=True)
finally:
    shutil.rmtree(snap, ignore_errors=True)  # never hold the trainer's checkpoint inodes longer than the read

for b, net in enumerate(model.nets if a.compare == 'group' else []):
    for l, layer in enumerate(net.layers):
        layer._net, layer._lid = b, l
        layer._grp = group_map(b, l, cfg.n_exp)
        # sanity: every copy's router row is nearest to its mapped source among the source experts
        W = (layer.router.weight.detach().double() * layer.n2.detach().double())
        near = torch.cdist(W[a.src_experts:], W[:a.src_experts]).argmin(1)
        bad = int((near != layer._grp[a.src_experts:]).sum())
        assert bad == 0, f'net {b} layer {l}: {bad} copies not nearest their mapped source'

# ---------------------------------------------------------------- the trainer's validation rows
vraw = np.fromfile(os.path.join(a.run, 'val.bin'), dtype=T.id_dtype())
vlen = min(int(st['args'].get('val_len') or cfg.ctx), st['args'].get('window', 4096))
vrows = min(int(st['args'].get('val_rows', 64)), len(vraw) // vlen)
vstep = max(1, len(vraw) // vlen // vrows)
val = torch.from_numpy(np.stack([vraw[i * vstep * vlen:(i * vstep + 1) * vlen] for i in range(vrows)])).long()
val = val[:a.rows]
print(f'{len(val)} validation rows of {vlen} ids (trainer: {vrows} rows)', flush=True)

out = {'ckpt': ck_live, 'steps': st['steps'], 'seen': st.get('seen'), 'rows': len(val), 'vlen': vlen, 'nets': {}}
with torch.no_grad():
    for b in [int(x) for x in a.nets.split(',')]:
        model.activate(b, True, -1)
        res = {}
        arms = (('standard', route_standard), ('group', route_group)) if a.compare == 'group' else \
            (('standard', lambda layer, probs, k: probs.topk(k, dim=-1)), ('top1', route_top1))
        for name, fn in arms:
            T.ROUTE = fn
            t = time.time()
            ev = T.evaluate(model, b, val, a.bs)
            res[name] = {k: v for k, v in ev.items() if isinstance(v, float)}
            res[name]['seconds'] = round(time.time() - t, 1)
            print(f'net {b} {name:8s}: {json.dumps(res[name])}', flush=True)
        other = arms[1][0]
        res['delta_ar'] = res[other]['ar'] - res['standard']['ar']
        res['coselect'] = {str(l): round(v[0] / max(1, v[1]), 4) for (bb, l), v in sorted(STATS.items()) if bb == b}
        res['top2_mass'] = {str(l): round(v[2] / max(1, v[1]), 4) for (bb, l), v in sorted(STATS.items()) if bb == b}
        print(f'net {b}: {other} - standard = {res["delta_ar"]:+.4f} nats; co-selected copies per layer '
              f'{res["coselect"]}', flush=True)
        out['nets'][str(b)] = res
        with open(os.path.join(a.work, f'route_ab_{a.compare}_result.json'), 'w') as f:
            json.dump(out, f, indent=1)
print(json.dumps(out))
