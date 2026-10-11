#!/usr/bin/env python3
"""Free disk on the 16B box: delete only RPV16-era /workspace files whose sha256 matches ARCHIVE_MANIFEST_20261011
(the verified GDrive/KnowHow/laptop archive). Unmatched files, dot entries and everything 16B are kept.
  free_space.py            dry run (prints what would go)
  free_space.py --apply    delete matched files, then empty dirs left behind
"""
import hashlib, json, os, sys
W = '/workspace'
KEEP = {'agillm16b', 'agillm16b_run', 'agillm16b_run.sh', 'agillm16b_seed', 'agillm16b_merge_kit', 'claude_16b_eval',
        'seed_src_8b_29877', 'seed_xfer_tools', 'onstart.sh', 'onstart.sh.pre-16b-retention-20261011T184804Z',
        'ARCHIVE_MANIFEST_20261011.sha256', 'ARCHIVE_ENV_20261011.txt'}
apply = '--apply' in sys.argv
man = {}
for line in open(f'{W}/ARCHIVE_MANIFEST_20261011.sha256'):
    h, _, p = line.rstrip('\n').partition('  ')
    man[p] = h
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()
gone = kept = 0; kept_files = []; n = 0
for top in sorted(os.listdir(W)):
    if top in KEEP or top.startswith('.'):
        continue
    root = os.path.join(W, top)
    walk = [(root, [], [])] if os.path.isfile(root) or os.path.islink(root) else os.walk(root)
    for d, _, fs in walk:
        paths = [d] if d == root and not os.path.isdir(root) else [os.path.join(d, f) for f in fs]
        for p in paths:
            if os.path.islink(p) or not os.path.isfile(p):
                kept_files.append(p); continue
            rel = os.path.relpath(p, '/')
            sz = os.path.getsize(p)
            if man.get(rel) == sha(p):
                gone += sz; n += 1
                if apply:
                    os.remove(p)
            else:
                kept += sz; kept_files.append(p)
    if apply and os.path.isdir(root):
        for d, _, _ in sorted(os.walk(root), key=lambda t: -len(t[0])):
            try: os.rmdir(d)
            except OSError: pass
print(json.dumps({'apply': apply, 'files_matched': n, 'bytes_matched': gone, 'gib_matched': round(gone / 2**30, 2),
                  'kept_unmatched_bytes': kept, 'kept_unmatched': kept_files[:40], 'kept_count': len(kept_files)}))
