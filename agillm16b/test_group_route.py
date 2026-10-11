"""Tests for group-limited routing (group_route.py). Run: python3 -m pytest -q agillm16b/test_group_route.py"""
import torch

from group_route import _group_topk, group_map_from_seed

E_SRC, K = 32, 2
BUSY = [17, 21, 28, 31]  # net0 layer 0 of the real seed (seed.log)


def seeded(n=4096, noise=0.0, seed=0):
    """Router probabilities of a 16B layer seeded from a 32-expert source: 68 copies, copy logits = source logits
    (+ optional noise, the seed tool's --router-noise)."""
    g = torch.tensor(group_map_from_seed(E_SRC, [BUSY], 68)[0])
    gen = torch.Generator().manual_seed(seed)
    src_logits = torch.randn(n, E_SRC, generator=gen) * 2.0
    logits = src_logits[:, g] + noise * torch.randn(n, 68, generator=gen)
    return src_logits, logits.softmax(-1), g


def test_copy_map_matches_seed_order():
    g = group_map_from_seed(E_SRC, [BUSY], 68)[0]
    assert g[:32] == list(range(32)) and g[32:64] == list(range(32)) and g[64:] == sorted(BUSY)


def test_plain_topk_co_selects_copies_at_the_seed():
    """The defect: with exact copies, plain top-2 puts both slots on two copies of one source (source's TOP-1)."""
    _, probs, g = seeded()
    _, top_i = probs.topk(K, dim=-1)
    assert bool((g[top_i[:, 0]] == g[top_i[:, 1]]).all())


def test_group_topk_is_the_source_top2_at_the_seed():
    src_logits, probs, g = seeded()
    top_p, top_i = _group_topk(probs, K, g)
    src_p, src_i = src_logits.softmax(-1).topk(K, dim=-1)
    assert torch.equal(g[top_i], src_i)  # the same source experts, in the same order
    gate = top_p / top_p.sum(-1, keepdim=True)  # the trainer renormalises over the k chosen
    src_gate = src_p / src_p.sum(-1, keepdim=True)
    assert torch.allclose(gate, src_gate, atol=1e-5)


def test_group_topk_slots_are_distinct_groups_and_slot0_is_argmax():
    _, probs, g = seeded(noise=0.5, seed=1)
    top_p, top_i = _group_topk(probs, K, g)
    assert bool((g[top_i[:, 0]] != g[top_i[:, 1]]).all())
    assert torch.equal(top_i[:, 0], probs.argmax(-1))  # the aux loss reads slot 0 as the first choice
    assert bool((top_p[:, 0] >= top_p[:, 1]).all())  # descending, like topk
    assert torch.equal(top_p, probs.gather(1, top_i))


def test_group_topk_picks_the_best_copy_inside_a_group():
    _, probs, g = seeded(noise=0.5, seed=2)
    top_p, top_i = _group_topk(probs, K, g)
    for j in range(K):
        inside = probs.masked_fill(g[None, :] != g[top_i[:, j]][:, None], -1.0)
        assert torch.equal(top_p[:, j], inside.max(-1).values)


def test_identity_groups_equal_plain_topk():
    """Every expert its own group (an un-upcycled model): exactly plain top-k."""
    gen = torch.Generator().manual_seed(3)
    probs = torch.randn(1024, 32, generator=gen).softmax(-1)
    a = _group_topk(probs, K, torch.arange(32))
    b = probs.topk(K, dim=-1)
    assert torch.equal(a[1], b.indices) and torch.equal(a[0], b.values)


def test_gradient_reaches_the_chosen_probabilities():
    _, probs, g = seeded(n=64, noise=0.5, seed=4)
    logits = probs.log().clone().requires_grad_(True)
    top_p, _ = _group_topk(logits.softmax(-1), K, g)
    (top_p / top_p.sum(-1, keepdim=True))[:, 0].sum().backward()
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all()) and logits.grad.abs().sum() > 0
