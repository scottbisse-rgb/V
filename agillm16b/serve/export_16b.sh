#!/bin/bash
# Export one hard-linked ckpt.good snapshot of the live 16B run to the one-bit bintf2 export format (CPU only).
# Usage: export_16b.sh SNAPRUN OUT.pt   (SNAPRUN/ckpt holds hard links; they are removed when the export ends)
set -u
SNAPRUN=$1; OUT=$2
. /workspace/agillm16b/env.sh
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
t0=$(date +%s)
nice -n 19 ionice -c3 /opt/conda/bin/python /workspace/agillm16b/code/bintf2_16b.py export --run "$SNAPRUN" --out "$OUT.part"
rc=$?
rm -rf "$SNAPRUN/ckpt"
if [ $rc -eq 0 ]; then mv "$OUT.part" "$OUT" && sha256sum "$OUT" > "$OUT.sha256"; fi
echo "export rc=$rc seconds=$(( $(date +%s) - t0 )) out=$OUT"
