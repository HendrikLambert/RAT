#!/usr/bin/env bash
# Build the RAT Apptainer image — run ON A DAIC LOGIN NODE (has internet).
#
#   bash scripts/daic_build_container.sh
#
# Output .sif goes to shared storage (home is only ~8 GB; the image is several GB).
# Apptainer caches layers under APPTAINER_CACHEDIR, also redirected to shared.

set -euo pipefail

SHARED="${SHARED:-/tudelft.net/staff-umbrella/CS4725/$USER}"
SIF="${SIF:-$SHARED/rat.sif}"
DEF="$(dirname "$0")/rat.def"

# The umbrella /tudelft.net mount (CIFS) forbids chmod, which Apptainer's OCI
# cache + build sandbox require ("operation not permitted"). So build on LOCAL
# disk and copy only the finished .sif to shared storage.
LOCAL_TMP="${LOCAL_TMP:-/tmp/$USER}"
mkdir -p "$LOCAL_TMP"
export APPTAINER_CACHEDIR="$LOCAL_TMP/apptainer_cache"
export APPTAINER_TMPDIR="$LOCAL_TMP/apptainer_tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

LOCAL_SIF="$LOCAL_TMP/rat.sif"
echo "==> Building $LOCAL_SIF (local disk) from $DEF"
# --fakeroot lets %post (pip install) run unprivileged; DAIC's REIT template
# builds this way. If fakeroot is unavailable, see the fallback note below.
apptainer build --fakeroot "$LOCAL_SIF" "$DEF"

echo "==> Copying image to shared storage: $SIF"
cp -f "$LOCAL_SIF" "$SIF"
rm -rf "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$LOCAL_SIF"

echo
echo "==> Built $SIF"
apptainer exec "$SIF" python -c "import torch, hydra, datasets; \
print('torch', torch.__version__, '| datasets', datasets.__version__)"
echo
echo "Next (still on the login node) — tokenize PG19 into shared storage:"
echo "  HF_HOME=$SHARED/hf apptainer exec -B /tudelft.net/:/tudelft.net/ \\"
echo "    $SIF python tokenize/pg19.py --out_dir $SHARED/pg19 --num_proc 8"
echo
echo "If 'apptainer build --fakeroot' is denied on DAIC, fall back to:"
echo "  apptainer build $SIF docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel"
echo "  then pip-install the extras with PYTHONUSERBASE=$SHARED/pyuser (ask Claude)."
