"""Group-limited top-k routing for an upcycled MoE (AGILLM-16B seeded from the 8B by copying experts).

The source of truth for the routing rule: make_grouproute_trainer.py inserts ROUTE_SRC (below) into the trainer
verbatim, and the tests exercise this same text, so the trainer and the tests cannot drift apart.
"""

ROUTE_SRC = '''
def _group_topk(probs, k, grp):
    """Top-k over DISTINCT source experts of an upcycled MoE (--route-groups).

    probs (N, E) router probabilities; grp (E,) long, the source expert each expert was copied from. A group's score
    is its best copy's probability; the k best groups are taken, then the best copy inside each. Returns
    (top_p, top_i) like probs.topk(k): top_i[:, 0] is the overall argmax and the rows are in descending probability.
    With exact copies this is the source model's top-k (same experts up to copy, same renormalised gates), where a
    plain topk puts every slot on copies of the token's first expert (they tie): the source's TOP-1 output."""
    N, E = probs.shape
    best = probs.new_full((N, int(grp.max()) + 1), -1.0).scatter_reduce(1, grp.expand(N, E), probs, 'amax')
    tg = best.topk(k, dim=-1).indices
    top_i = torch.stack([probs.masked_fill(grp[None, :] != tg[:, j:j + 1], -1.0).argmax(-1) for j in range(k)], -1)
    return probs.gather(1, top_i), top_i
'''

exec(compile('import torch\n' + ROUTE_SRC, __file__ + ':ROUTE_SRC', 'exec'))  # defines _group_topk here


def group_map_from_seed(src_experts, busy_per_layer, n_exp):
    """The 16B seed's expert order (seed_8b_to_16b.py): new expert j < src_experts is source j (copy 0), then copy 1
    of every source in expert order, then copy 2 of the busiest (busy_per_layer[l], in expert order). Returns a list
    per layer of n_exp source ids."""
    out = []
    for busy in busy_per_layer:
        g = list(range(src_experts)) * 2 + sorted(busy)
        if len(g) != n_exp:
            raise ValueError(f'copy map has {len(g)} experts, the model {n_exp}')
        out.append(g)
    return out
