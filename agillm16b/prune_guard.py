#!/usr/bin/env python3
"""AGILLM-16B auto-pruner (v2), RTX 3090 Vast 53075549. Same CLI and status file as v1.

  prune_guard.py once|loop [--apply]     one pass / a pass every 10 min (without --apply: report only)
  prune_guard.py self-test               the deletion rules on fixtures in a temp dir

The trainer owns its checkpoint generations: save writes ckpt.new, removes ckpt.old, renames ckpt -> ckpt.old and
ckpt.new -> ckpt; keep_good hard-links the clean one as ckpt.good; the data stream deletes consumed pools itself.
This guard never touches ckpt, ckpt.new, ckpt.good(.new), ckpt.restore, parscale_last, data, val.bin or logs.
With --apply it removes only:

  1. backup_snap/snap_* left by a dead backup (older than 90 min, taken while holding the backup's flock), and only
     when that cannot lose progress: every file is still linked from elsewhere (a live generation holds the same
     inodes, so nothing is lost), or the live ckpt AND ckpt.old are both valid and newer than the snapshot.
  2. the uploader's Hugging Face xet chunk cache (home/.cache/huggingface/xet) above 2 GiB, while no backup runs.
  3. LAST RESORT, ckpt.old: only when free disk is below what the next save needs (the size of ckpt + 2 GiB),
     no save is in progress (no ckpt.new), ckpt is valid and newer than ckpt.old, and the off-host backup is under
     12 h old. The trainer removes ckpt.old at its next save anyway; this only does it early so the save fits.
     Nothing is freed when ckpt.good shares ckpt.old's files (the links stay), and that is reported.

v1 (monitor-only after a safety review) is kept next to this file as prune_guard.py.v1.
"""
import argparse, fcntl, json, os, shutil, tempfile, time
from pathlib import Path

BASE = Path("/workspace/agillm16b")
RUN = Path("/workspace/agillm16b_run")
SNAP = BASE / "backup_snap"
LOCK = BASE / "backup.lock"
STATUS = BASE / "prune_guard_status.json"
XET = BASE / "home/.cache/huggingface/xet"
GIB = 2 ** 30
SNAP_MIN_AGE_S = 5400
XET_MAX_BYTES = 2 * GIB
SAVE_MARGIN_BYTES = 2 * GIB
BACKUP_FRESH_H = 12


def checkpoint(path):
    f = path / "state.json"
    if not f.is_file():
        return {"exists": path.exists(), "valid": False}
    try:
        d = json.loads(f.read_text())
        return {"exists": True, "valid": True, "step": sum(d.get("steps", [])),
                "format": "low-bit" if d.get("ckpt_format") else "int8"}
    except (ValueError, OSError, TypeError):
        return {"exists": True, "valid": False}


def tree_bytes(path, only_last_link=False):
    n = 0
    for d, _, fs in os.walk(path):
        for name in fs:
            st = os.lstat(os.path.join(d, name))
            if not only_last_link or st.st_nlink == 1:
                n += st.st_size
    return n


def backup_lock(lockfile):
    """The backup producer's flock, non-blocking: an open file holding it, or None while a backup runs."""
    f = open(lockfile, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except BlockingIOError:
        f.close()
        return None


def sweep_snaps(root, run, apply, older=SNAP_MIN_AGE_S, now=None):
    now = now or time.time()
    cur, old = checkpoint(run / "ckpt"), checkpoint(run / "ckpt.old")
    out = {"candidates": [], "removed": [], "kept_unsafe": []}
    if not root.is_dir():
        return out
    for p in sorted(root.iterdir()):
        if not p.name.startswith("snap_") or p.is_symlink() or not p.is_dir() or p.parent.resolve() != root.resolve():
            continue
        if now - p.stat().st_mtime < older:
            continue
        out["candidates"].append(p.name)
        last = tree_bytes(p, only_last_link=True)
        snap = checkpoint(p)
        superseded = (cur.get("valid") and old.get("valid") and snap.get("valid")
                      and cur["step"] > snap["step"] and old["step"] >= snap["step"])
        if last and not superseded:
            out["kept_unsafe"].append({"snap": p.name, "last_copy_bytes": last, "step": snap.get("step")})
            continue
        if apply:
            shutil.rmtree(p)
            out["removed"].append({"snap": p.name, "freed_bytes": last})
    return out


def prune_xet(xet, apply, cap=XET_MAX_BYTES):
    size = tree_bytes(xet) if xet.is_dir() else 0
    out = {"bytes": size, "removed": False}
    if size > cap and apply:
        shutil.rmtree(xet, ignore_errors=True)
        out["removed"] = True
    return out


def last_resort_old(run, free, backup_age_h, apply):
    cur, old, new = checkpoint(run / "ckpt"), checkpoint(run / "ckpt.old"), run / "ckpt.new"
    need = (tree_bytes(run / "ckpt") if (run / "ckpt").is_dir() else 0) + SAVE_MARGIN_BYTES
    out = {"free_bytes": free, "next_save_needs_bytes": need, "action": "none"}
    if free >= need:
        return out
    why = [w for w, bad in (("ckpt not valid", not cur.get("valid")),
                            ("no valid ckpt.old", not old.get("valid")),
                            ("ckpt not newer than ckpt.old", cur.get("valid") and old.get("valid")
                             and cur["step"] <= old["step"]),
                            ("a save is in progress (ckpt.new)", new.exists()),
                            ("off-host backup missing or older than 12 h",
                             backup_age_h is None or backup_age_h > BACKUP_FRESH_H)) if bad]
    if why:
        out["action"] = "blocked: " + "; ".join(why)
        return out
    out["would_free_bytes"] = tree_bytes(run / "ckpt.old", only_last_link=True)
    if apply:
        shutil.rmtree(run / "ckpt.old")
        out["action"] = "removed ckpt.old"
    else:
        out["action"] = "would remove ckpt.old (report only)"
    return out


def tick(apply=False, base=BASE, run=RUN):
    snaproot, lockfile = base / "backup_snap", base / "backup.lock"
    try:
        ack = json.loads((base / "backup_last_ok.json").read_text())
        backup_age_h = round((time.time() - ack["time"]) / 3600, 2)
    except (OSError, ValueError, KeyError, TypeError):
        backup_age_h = None
    lk = backup_lock(lockfile)
    try:
        if lk is None:
            snaps = {"backup_busy": True}
            xet = {"backup_busy": True}
        else:
            snaps = {"backup_busy": False, **sweep_snaps(snaproot, run, apply)}
            xet = {"backup_busy": False, **prune_xet(base / "home/.cache/huggingface/xet", apply)}
        free = shutil.disk_usage(run).free
        last = last_resort_old(run, free, backup_age_h, apply)
    finally:
        if lk is not None:
            lk.close()
    free = shutil.disk_usage(run).free
    status = {
        "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": 2, "apply": apply,
        "disk_free_gib": round(free / GIB, 2),
        "warn_low_disk": free < 20 * GIB,
        "critical_low_disk": free < 12 * GIB,
        "off_host_backup_age_h": backup_age_h,
        "off_host_backup_missing_or_stale": backup_age_h is None or backup_age_h > BACKUP_FRESH_H,
        "ckpt": checkpoint(run / "ckpt"), "ckpt_old": checkpoint(run / "ckpt.old"),
        "ckpt_good": checkpoint(run / "ckpt.good"), "ckpt_new": checkpoint(run / "ckpt.new"),
        "snapshot_cleanup": snaps, "xet_cache": xet, "last_resort_ckpt_old": last,
        "protected": ["ckpt", "ckpt.new", "ckpt.good", "ckpt.good.new", "ckpt.restore", "parscale_last", "data",
                      "val.bin", "logs"],
    }
    if base == BASE:
        tmp = STATUS.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(status, sort_keys=True, indent=2) + "\n")
        os.replace(tmp, STATUS)
    print(json.dumps(status, sort_keys=True), flush=True)
    return status


def self_test():
    def gen(path, step, payload=b"x" * 1000, link_from=None):
        path.mkdir(parents=True)
        if link_from:
            for f in link_from.iterdir():
                os.link(f, path / f.name)
        else:
            (path / "net0.lb").write_bytes(payload)
            (path / "state.json").write_text(json.dumps({"steps": [step]}))

    with tempfile.TemporaryDirectory(prefix="agillm16b_prune_test_") as d:
        d = Path(d)
        run, snaps = d / "run", d / "snap"
        run.mkdir()
        snaps.mkdir()
        old_t = time.time() - 7200
        # live generations: ckpt (step 30), ckpt.old (step 20); ckpt.good = ckpt
        gen(run / "ckpt", 30)
        gen(run / "ckpt.old", 20)
        gen(run / "ckpt.good", 30, link_from=run / "ckpt")
        # a stale snapshot of the live ckpt (same inodes): removable, frees nothing
        gen(snaps / "snap_linked", 30, link_from=run / "ckpt")
        # a stale snapshot that is the LAST copy of step 10, superseded by valid newer ckpt and ckpt.old: removable
        gen(snaps / "snap_superseded", 10)
        # a stale last copy NEWER than ckpt.old (step 25): must be kept
        gen(snaps / "snap_lastcopy", 25)
        # a fresh snapshot: kept regardless
        gen(snaps / "snap_fresh", 5)
        for n in ("snap_linked", "snap_superseded", "snap_lastcopy"):
            os.utime(snaps / n, (old_t, old_t))
        r = sweep_snaps(snaps, run, apply=True)
        removed = sorted(x["snap"] for x in r["removed"])
        assert removed == ["snap_linked", "snap_superseded"], r
        assert (snaps / "snap_lastcopy").exists() and (snaps / "snap_fresh").exists()
        assert [x["snap"] for x in r["kept_unsafe"]] == ["snap_lastcopy"]
        assert (run / "ckpt" / "net0.lb").exists() and (run / "ckpt.good" / "net0.lb").exists()
        # a broken ckpt.old makes every last-copy snapshot unsafe
        os.utime(snaps / "snap_lastcopy", (old_t, old_t))
        (run / "ckpt.old" / "state.json").write_text("{not json")
        gen(snaps / "snap_s2", 1)
        os.utime(snaps / "snap_s2", (old_t, old_t))
        r = sweep_snaps(snaps, run, apply=True)
        assert r["removed"] == [] and (snaps / "snap_s2").exists(), r
        (run / "ckpt.old" / "state.json").write_text(json.dumps({"steps": [20]}))
        # last resort: plenty of disk -> nothing
        assert last_resort_old(run, 10 * GIB, 1.0, True)["action"] == "none"
        # low disk but stale backup -> blocked, ckpt.old kept
        r = last_resort_old(run, 0, 30.0, True)
        assert r["action"].startswith("blocked") and (run / "ckpt.old").exists(), r
        # low disk, save in progress -> blocked
        (run / "ckpt.new").mkdir()
        r = last_resort_old(run, 0, 1.0, True)
        assert "save is in progress" in r["action"] and (run / "ckpt.old").exists(), r
        (run / "ckpt.new").rmdir()
        # low disk, report only -> kept
        assert "report only" in last_resort_old(run, 0, 1.0, False)["action"] and (run / "ckpt.old").exists()
        # low disk, everything valid, fresh backup -> ckpt.old removed, ckpt and ckpt.good untouched
        r = last_resort_old(run, 0, 1.0, True)
        assert r["action"] == "removed ckpt.old" and not (run / "ckpt.old").exists(), r
        assert (run / "ckpt" / "state.json").exists() and (run / "ckpt.good" / "net0.lb").exists()
        # ckpt not newer than ckpt.old -> blocked
        gen(run / "ckpt.old", 99)
        assert "not newer" in last_resort_old(run, 0, 1.0, True)["action"] and (run / "ckpt.old").exists()
        # xet cache: under the cap kept, over the cap removed (cap lowered for the fixture)
        xet = d / "xet"
        xet.mkdir()
        (xet / "chunk").write_bytes(b"y" * 4096)
        assert not prune_xet(xet, True, cap=10 ** 6)["removed"] and xet.exists()
        assert prune_xet(xet, True, cap=100)["removed"] and not xet.exists()
        # a held backup lock blocks the snapshot and cache sweeps
        lockfile = d / "backup.lock"
        with open(lockfile, "a+") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert backup_lock(lockfile) is None
        lk = backup_lock(lockfile)
        assert lk is not None
        lk.close()
    print("SELF_TEST_PASS v2: linked/superseded snapshots removed, last-copy/fresh/broken-old kept; ckpt.old only "
          "removed when low disk + valid newer ckpt + no save + fresh backup; xet cap; backup lock respected")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("once", "loop", "self-test"))
    p.add_argument("--apply", action="store_true", help="delete what the rules allow (default: report only)")
    a = p.parse_args()
    if a.mode == "self-test":
        self_test()
    elif a.mode == "once":
        tick(a.apply)
    else:
        while True:
            try:
                tick(a.apply)
            except Exception as e:
                print("GUARD_ERROR " + type(e).__name__ + ": " + str(e), flush=True)
            time.sleep(600)
