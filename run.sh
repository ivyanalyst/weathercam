#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-weathercam:test}"
PORT="${PORT:-8014}"
OUTPUTS="${OUTPUTS:-$PWD/outputs}"
HF_DIR="${HF_DIR:-$HOME/.cache/weathercam-hf}"
GALLERY_NPZ="${GALLERY_NPZ:-}"
VERI_ZIP="${VERI_ZIP:-}"

CMD_ARGS=(--embed-worker-python /venvs/embed/bin/python --plate-worker-python /venvs/yolo/bin/python)

die() { echo "error: $*" >&2; exit 1; }

[ -d "$OUTPUTS" ] || die "missing outputs dir: $OUTPUTS"
mkdir -p "$HF_DIR"

MOUNTS=(
  -v "$OUTPUTS":/app/outputs:ro
  -v "$HF_DIR":/hf
)

if [ -n "$GALLERY_NPZ" ]; then
  [ -f "$GALLERY_NPZ" ] || die "missing file: $GALLERY_NPZ"
  MOUNTS+=(-v "$GALLERY_NPZ":/data/gallery.npz:ro)
  CMD_ARGS+=(--gallery /data/gallery.npz)
else
  echo "note: no GALLERY_NPZ set, text search may not work"
fi

if [ -n "$VERI_ZIP" ]; then
  [ -f "$VERI_ZIP" ] || die "missing file: $VERI_ZIP"
  MOUNTS+=(-v "$VERI_ZIP":/data/VeRi.zip:ro)
  CMD_ARGS+=(--image-zip /data/VeRi.zip)
else
  echo "note: no VERI_ZIP set, search results will have no thumbnails"
fi

OFFLINE=0
if [ -n "$(ls -A "$HF_DIR" 2>/dev/null)" ]; then
  OFFLINE=1
else
  echo "note: HF cache is empty, first run needs internet (model download)"
fi

echo "starting on http://127.0.0.1:${PORT}"
docker run --rm --user "$(id -u):$(id -g)" \
  -p "127.0.0.1:${PORT}:8000" \
  -e HF_HOME=/hf -e HOME=/tmp -e HF_HUB_OFFLINE="$OFFLINE" \
  "${MOUNTS[@]}" \
  "$IMAGE" "${CMD_ARGS[@]}"
