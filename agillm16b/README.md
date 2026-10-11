# AGILLM-16B routing audit: expert copies are used as pairs

The live 16B (RTX 3090, Vast 53075549, `bintf2_16b.py` sha `ef0ca33a`) was seeded from the 8B by copying every
expert: 32 source experts per layer became 68 (copy 0 of each, copy 1 of each, copy 2 of the four busiest). Its router
picks the top 2 of the 68 by softmax probability, and copies of one expert have nearly the same router row. So on
almost every token **both slots go to two copies of the same 8B expert**. The 16B mostly runs the 8B's top-1 over its
32 experts. The extra experts are used as near-duplicate pairs, not as extra capacity.

**Bottom line**

1. This is real and pervasive: 89–98% of tokens, in every layer of every net.
2. It cost little at the start. Paired on the same 8B weights, top-1 instead of top-2 costs **+0.010 to +0.012 nats**.
3. Switching routing on the live checkpoint makes it **worse** (+0.020 to +0.023 nats), because the copies have
   co-adapted as pairs. **Do not change the live run** on the strength of this.
4. Whether *training* with distinct-expert routing does better is **not measured**. That needs a GPU training A/B.
   An opt-in, tested patch (`--route-groups`) is ready for it. Without the flag it is bit-identical to the live trainer.
5. Single-checkpoint evals swing by about ±0.05 nats, so comparisons across checkpoints (16B vs 8B, before vs after a
   hot change) need paired evaluation. I retracted an early "16B is 0.06 behind the 8B" claim for this reason.

All measurements were offline: CPU at nice 19 on the 3090 box, no GPU, nothing under `/workspace/agillm16b*` changed.
Two 16-thread evals at once slowed the trainer by about 15%, so they were run one at a time after that.

## Measurements

Every number below uses the same 16 validation rows × 2,048 tokens, the first 16 of the trainers' 64, from
`val.bin` (md5 `d9f4fbd9…`, identical on the 16B and 8B boxes), scored by the trainer's own `evaluate()` (next-token
nats).

### 1. Co-selection on the current 16B (`ckpt.good`, step ~31.7k)

The share of tokens whose two chosen experts are copies of one source expert:

| net | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.976 | 0.976 | 0.966 | 0.947 | 0.968 | 0.961 | 0.928 | 0.891 |
| 1 | 0.974 | 0.959 | 0.981 | 0.961 | 0.961 | 0.949 | 0.921 | 0.888 |
| 2 | 0.969 | 0.947 | 0.977 | 0.968 | 0.957 | 0.945 | 0.925 | 0.905 |

The copies are drifting apart slowly. Their router distance from their source, as a fraction of the distance between
two different source experts, went from 0.09 (checkpoint at 18:39 UTC) to 0.12–0.15 (19:10) to 0.14–0.16 (19:41),
against about 0.01 logit of seed noise.

### 2. Routing A/B (paired: same weights, same rows)

| net | 8B seed source, top-2 | same, top-1 | **top-1 cost** | 16B now, standard | 16B now, group-limited | **switch cost** |
|---|---|---|---|---|---|---|
| 0 | 4.2704 | 4.2815 | **+0.011** | 4.1820 | 4.2019 | **+0.020** |
| 1 | 4.0939 | 4.1038 | **+0.010** | 4.1373 | 4.1598 | **+0.023** |
| 2 | 4.1507 | 4.1624 | **+0.012** | 4.1339 | 4.1568 | **+0.023** |

* **Top-1 cost** is what the seed gave up by co-selecting copies, since standard routing over exact copies computes
  the source's top-1. It is about 0.01 nats: the 8B's second expert carries little of the output.
* **Switch cost** is what turning on group-limited routing at inference costs the current 16B. After 2,400 steps of
  training as pairs, a different second expert is worse than the co-adapted copy.
* The 16B-vs-8B columns are *different checkpoints* and are not a fair comparison (net 0: −0.088, net 1: +0.043,
  net 2: −0.017). See section 3.

### 3. Evaluation noise: compare checkpoints only with paired evals

The 8B's own net-0 evals (64 rows) went 4.264 (step 26,250) → 4.177 (30,000) → 4.225 (33,750). Swings of about 0.05
between consecutive evals of one net are larger than most effects worth deciding on. Both runs now train with
`--avg-turns 0`, so evals read the raw latents, which carry the last steps' sign flips. Before trusting a before/after
from the trainer's evals (for example after a hot.json change), score both checkpoints on fixed rows with
`route_ab_eval.py`, or re-enable running means for evaluation (`--avg-turns`: "Training never reads it").

## The patch: group-limited routing (`--route-groups`)

Pick the k best **distinct source experts**, scoring each by its best copy, then use the best copy inside each
(`group_route.py`). With exact copies this is exactly the source model's top-k: same experts, same renormalised gates
(tested). Copies still compete inside their group for each token, so they get different tokens and can specialise.

`make_grouproute_trainer.py` builds `bintf2_16b_grouproute.py` from the live trainer with 8 exact-anchor edits
(`grouproute.diff`, 115 lines). It adds `--route-groups FILE` to `train`, `eval` and `generate`:

* **Off (no flag): bit-identical to the live trainer.** CPU tiny-moe, 30 steps: original run twice and patched run
  gave identical losses at every step.
* **On:** it trains, saves, evaluates, and prints the routing line. Nothing new goes into the checkpoint; `state.json`
  args record the file, and a resume without the flag prints a warning (all verified on CPU).
* `route_groups_16b.json` (sha `393be3fa`) is the 16B's map: 3 nets × 8 layers × 68 experts. `make_route_groups.py`
  builds it from `seed.log` and refuses to write unless every copy's router row is nearest its mapped source in the
  checkpoint (checked against `ckpt.good` at step 32,607).
* A model trained with the flag must be evaluated and served with the same map. The trainer's `eval` and `generate`
  take it; any separate inference engine would need the same rule.

## Recommendation

* **Live 16B: no change.** Nothing here justifies a restart. Switching routing mid-run costs about 0.02 nats up front,
  and its long-run benefit is unmeasured.
* **If you want to know whether distinct-expert routing trains better:** run a training A/B off the live GPU. Fork the
  current checkpoint into two runs, standard vs `--route-groups route_groups_16b.json`, with the same data and steps.
  Run each for a few thousand steps and score both on the same fixed rows with `route_ab_eval.py`. Adopt only if the
  group arm overtakes the 0.02 starting penalty and keeps a margin.
* **Next time a model is grown by copying experts:** seed with `--route-groups` from step 0. There it costs nothing:
  it reproduces the source's top-2 exactly instead of its top-1, and lets copies specialise from the first step.
  The seed tool's own "ROUTER OFFSETS" note says no per-copy constant can do this. A routing rule can.

## Disk space and the auto-pruner (3090 box, done 2026-10-11 ~21:00–21:08 BST, on Scott's order)

**Freed 5.3 GB (18.1 → 23.4 GB free)**, with nothing removed that isn't backed up elsewhere:

* 3.6 GiB: 8,882 RPV16-era files under `/workspace`, deleted by `free_space.py` only where the file's sha256 equals
  its entry in `ARCHIVE_MANIFEST_20261011.sha256` (the verified GDrive/KnowHow/laptop archive of this box). The 3
  files that did not match were kept. 16B paths, dot-folders, the manifest and onstart files are never candidates.
* 0.9 GB: `/var/tmp/hfpub/stage`, the staging copy of the scrubbed RPV16 logs. The same tar.gz (92,696,548 bytes) is
  on the public HF repo `MarxistLeninist/AGILLM-GB10-1PF`. Its small receipts were kept.
* 0.8 GB: `/root/.cache/huggingface/xet`, an upload chunk cache left from the RPV16 era.
* **Kept:** `/workspace/seed_src_8b_29877` (7.9 GB), the exact 8B checkpoint the 16B was seeded from. No off-host
  copy of that exact step was found (the nearest 8B HF export is a different step), so it is kept until Scott decides.

**Auto-pruner: `prune_guard.py` v2 is on (`--apply`)**, as loop pid started 20:07 UTC and in `/workspace/onstart.sh`
so it restarts with the box (backups: `prune_guard.py.v1`, `onstart.sh.pre-pruner-v2-20261011T200658Z`). Every 10 min
it removes only:

1. stale backup snapshots (over 90 min, backup lock free) that are not the last copy of a checkpoint, or are older
   than both valid live generations;
2. the uploader's HF xet cache above 2 GiB, while no backup runs;
3. as a last resort, `ckpt.old`, but only if free disk is below what the next save needs (ckpt size + 2 GiB), no save
   is in progress, `ckpt` is valid and newer, and the off-host backup is under 12 h old. The trainer deletes
   `ckpt.old` at every save anyway; this only does it early so the save fits.

It never touches `ckpt`, `ckpt.new`, `ckpt.good`, data or logs. The trainer already rotates its own checkpoints
and deletes consumed data pools. `prune_guard.py self-test` covers every rule on fixtures and passes on the box.
Status: `/workspace/agillm16b/prune_guard_status.json`; log: `prune_guard.out`.

## Other observations (not acted on)

* **GPU power cap.** The 3090 runs at a 200 W limit (default 350 W, max 400 W), and "SW Power Cap" is active.
  `nvidia-smi -pl` from inside the container is refused, so only the host can lift it. It probably costs 15–25% of
  throughput; prefer an uncapped instance at the next move.
* **ParScale add-on.** "Gain over stream 0 alone" has been ±0.000/0.001 on every 16B log line since launch (merge
  weights 0.95/0.05). It runs at a duty of 1/32, and its sub-steps cause the dips in tokens/s.
* The live step size is now 20,480 tokens (the chatgpt-16b-opt lane, +5.5–5.9% tokens/s); the 8B uses 32,768.

## Files

| file | what |
|---|---|
| `group_route.py` | the routing rule (`ROUTE_SRC`, inserted verbatim into the trainer) and the seed's copy map |
| `test_group_route.py` | 7 tests: the defect at the seed, exact source top-2 at the seed, distinct slots, argmax first, gradients |
| `make_grouproute_trainer.py` | builds the patched trainer from the live one (exact-anchor edits, fails loudly on drift) |
| `grouproute.diff` | the resulting diff against `bintf2_16b.py` sha `ef0ca33a` |
| `make_route_groups.py` | writes and checks the `--route-groups` file from `seed.log` + a checkpoint |
| `route_groups_16b.json` | the 16B's map (sha `393be3fa`) |
| `route_ab_eval.py` | the offline CPU A/B (`--compare group` or `--compare top1`, `--ckpt-dir` for any checkpoint) |
| `results/` | raw A/B outputs |
| `prune_guard.py` | the 16B box auto-pruner v2 (deployed as `/workspace/agillm16b/code/prune_guard.py`) |
| `free_space.py` | the manifest-verified cleanup used to free the RPV16-era files |

On the box: `/workspace/claude_16b_eval/` (patched trainer sha `e00b59af`, logs, smoke runs, about 2 MB).
