#!/usr/bin/env python3
"""Write the --route-groups file of the 16B: every layer's source expert for each of its 68 experts.

  python3 make_route_groups.py --seed-log /workspace/agillm16b_seed/seed.log --ckpt /workspace/agillm16b_run/ckpt.good \
      --out /workspace/agillm16b/route_groups_16b.json

The map is the seed's own expert order (seed_8b_to_16b.py: copy 0 of every source in place, copy 1 of every source in
expert order, copy 2 of the busiest per layer, from seed.log). It is checked against the checkpoint: every copy's
router row (times the norm gain in front of the router) must be nearest its mapped source among the source experts,
else nothing is written. Only reads the checkpoint's small fp files (net*.pt).
"""
import argparse
import hashlib
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from group_route import group_map_from_seed  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--seed-log', required=True)
p.add_argument('--ckpt', required=True, help='a checkpoint directory of the run (only its net*.pt are read)')
p.add_argument('--out', required=True)
p.add_argument('--src-experts', type=int, default=32)
a = p.parse_args()

log = open(a.seed_log).read()
busy = {int(m.group(1)): json.loads(m.group(2)) for m in
        re.finditer(r'net(\d): split by \w+; experts with an extra copy per layer (\[\[.*?\]\])', log)}
if not busy:
    raise SystemExit(f'no copy map in {a.seed_log}')
nets, report = [], []
for b in sorted(busy):
    fp = torch.load(os.path.join(a.ckpt, f'net{b}.pt'), map_location='cpu', weights_only=False)['fp']
    L = len(busy[b])
    E = fp['layers.0.router.weight'].shape[0]
    maps = group_map_from_seed(a.src_experts, busy[b], E)
    for l in range(L):
        W = fp[f'layers.{l}.router.weight'].double() * fp[f'layers.{l}.n2'].double()
        g = torch.tensor(maps[l])
        near = torch.cdist(W[a.src_experts:], W[:a.src_experts]).argmin(1)
        bad = int((near != g[a.src_experts:]).sum())
        if bad:
            raise SystemExit(f'net {b} layer {l}: {bad} copies are not nearest their mapped source; not written')
        d_copy = (W[a.src_experts:] - W[g[a.src_experts:]]).norm(dim=1).mean().item()
        D = torch.cdist(W[:a.src_experts], W[:a.src_experts])
        d_src = D[~torch.eye(a.src_experts, dtype=torch.bool)].mean().item()
        report.append(f'net {b} layer {l}: copy-to-source router distance {d_copy / d_src:.3f} of source-to-source')
    nets.append(maps)
st = json.load(open(os.path.join(a.ckpt, 'state.json')))
out = {'nets': nets, 'src_experts': a.src_experts, 'seed_log': a.seed_log,
       'seed_log_sha256': hashlib.sha256(log.encode()).hexdigest(), 'checked_against': a.ckpt,
       'checked_steps': st.get('steps')}
with open(a.out, 'w') as f:
    json.dump(out, f)
print('\n'.join(report))
print(f'wrote {a.out} sha256 {hashlib.sha256(open(a.out, "rb").read()).hexdigest()[:12]}: {len(nets)} nets x '
      f'{len(nets[0])} layers x {len(nets[0][0])} experts')
