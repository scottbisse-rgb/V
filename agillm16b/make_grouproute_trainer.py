#!/usr/bin/env python3
"""Make bintf2_16b_grouproute.py from the live 16B trainer: adds --route-groups (group-limited top-k routing).

  python3 make_grouproute_trainer.py --src /workspace/agillm16b/code/bintf2_16b.py --out bintf2_16b_grouproute.py

Every edit is an exact string replacement that must match once, so a changed trainer fails loudly instead of being
patched in the wrong place. Without --route-groups the output trainer computes exactly what the input does (the only
change on that path is a getattr on the router line).

--route-groups FILE (train, eval, generate): JSON {"nets": [[[source id per expert] per layer] per net], ...} as
written by make_route_groups.py. Each layer then routes its tokens to the k best DISTINCT source experts (the best
copy inside each): see group_route.py. Nothing new goes into the checkpoint; state.json's args record the file, and a
resume without the flag after a run with it prints a warning (the routing, and so the function, would change).
"""
import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from group_route import ROUTE_SRC  # noqa: E402

APPLY_SRC = '''


def apply_route_groups(model, path):
    """--route-groups FILE: every layer's source-expert map (see _group_topk). Returns the file's sha256[:12]."""
    if not path:
        return None
    raw = open(path, 'rb').read()
    nets = json.loads(raw)['nets']
    if len(nets) != len(model.nets):
        raise SystemExit(f'--route-groups {path}: {len(nets)} nets, the model has {len(model.nets)}')
    for b, (net, layers) in enumerate(zip(model.nets, nets)):
        if len(layers) != len(net.layers):
            raise SystemExit(f'--route-groups {path}: net {b} has {len(layers)} layers, the model {len(net.layers)}')
        for l, (layer, g) in enumerate(zip(net.layers, layers)):
            E = layer.router.weight.shape[0]
            if len(g) != E or min(g) < 0 or len(set(g)) < model.cfg.topk:
                raise SystemExit(f'--route-groups {path}: net {b} layer {l}: {len(g)} ids for {E} experts, '
                                 f'{len(set(g))} groups for top-{model.cfg.topk}')
            layer.grp = torch.tensor(g, dtype=torch.long, device=layer.router.weight.device)
    return hashlib.sha256(raw).hexdigest()[:12]
'''

EDITS = [
    # 1. the routing rule and its loader, in front of the layer that uses them
    ('\nclass Layer(nn.Module):',
     ROUTE_SRC.rstrip('\n') + APPLY_SRC + '\n\nclass Layer(nn.Module):'),
    # 2. the router: group-limited top-k when the layer has a source-expert map, else exactly as before
    ('            top_p, top_i = probs.topk(cfg.topk, dim=-1)\n',
     "            grp = getattr(self, 'grp', None)  # --route-groups\n"
     '            if grp is None:\n'
     '                top_p, top_i = probs.topk(cfg.topk, dim=-1)\n'
     '            else:\n'
     '                top_p, top_i = _group_topk(probs, cfg.topk, grp if grp.device == probs.device else grp.to(probs.device))\n'),
    # 3. train: the map after the checkpoint is read
    ('    else:\n        model.init_signs()\n    # ParScale, built in and on',
     '    else:\n        model.init_signs()\n'
     '    rg = apply_route_groups(model, a.route_groups)\n'
     '    if rg:\n'
     "        print(f'routing: top-{cfg.topk} over distinct source experts (--route-groups {a.route_groups}, sha256 {rg})',\n"
     '              flush=True)\n'
     "    elif ck and (st.get('args') or {}).get('route_groups'):\n"
     "        print(f'WARNING: this checkpoint was trained with --route-groups {st[\"args\"][\"route_groups\"]}; resuming '\n"
     "              'without it routes by plain top-k, which changes what the model computes', flush=True)\n"
     '    # ParScale, built in and on'),
    # 4. eval and generate read the same map
    ("    if load_ckpt(model, None, ck, None if a.latest else 'read'):\n"
     "        print('read from the running means of the latents (--latest: the latest latents)')\n",
     "    if load_ckpt(model, None, ck, None if a.latest else 'read'):\n"
     "        print('read from the running means of the latents (--latest: the latest latents)')\n"
     '    if apply_route_groups(model, a.route_groups):\n'
     "        print(f'routing: top-{cfg.topk} over distinct source experts (--route-groups {a.route_groups})')\n"),
    ("    means = load_ckpt(model, None, ck, None if a.latest else 'read')\n",
     "    means = load_ckpt(model, None, ck, None if a.latest else 'read')\n"
     '    if apply_route_groups(model, a.route_groups):\n'
     "        print(f'routing: top-{cfg.topk} over distinct source experts (--route-groups {a.route_groups})')\n"),
    # 5. the flag on the three commands
    ("    sp.add_argument('--run', required=True, help='directory for checkpoints, logs and the data cache')\n",
     "    sp.add_argument('--run', required=True, help='directory for checkpoints, logs and the data cache')\n"
     "    sp.add_argument('--route-groups', default='',\n"
     "                    help='JSON source-expert map of an upcycled MoE (make_route_groups.py): route each token to '\n"
     "                         'the top-k DISTINCT source experts, the best copy inside each. Give it on every resume')\n"),
    ("    sp = sub.add_parser('generate', help='sample all decoding modes from the checkpoint in --run')\n"
     "    sp.add_argument('--run', required=True)\n",
     "    sp = sub.add_parser('generate', help='sample all decoding modes from the checkpoint in --run')\n"
     "    sp.add_argument('--run', required=True)\n"
     "    sp.add_argument('--route-groups', default='', help='as train --route-groups (the same file)')\n"),
    ("    sp = sub.add_parser('eval', help='score the checkpoint in --run on its validation windows at any row length')\n"
     "    sp.add_argument('--run', required=True)\n",
     "    sp = sub.add_parser('eval', help='score the checkpoint in --run on its validation windows at any row length')\n"
     "    sp.add_argument('--run', required=True)\n"
     "    sp.add_argument('--route-groups', default='', help='as train --route-groups (the same file)')\n"),
]


def make(src):
    for old, new in EDITS:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f'edit anchor found {n} times (want 1): {old[:80]!r}')
        src = src.replace(old, new)
    return src


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    raw = open(a.src).read()
    out = make(raw)
    compile(out, a.out, 'exec')
    with open(a.out, 'w') as f:
        f.write(out)
    sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
    print(f'{a.src} sha256 {sha(raw)}\n{a.out} sha256 {sha(out)} ({len(EDITS)} edits)')
