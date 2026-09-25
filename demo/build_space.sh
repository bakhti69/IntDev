#!/usr/bin/env bash
# Assemble a self-contained Hugging Face Space (Gradio SDK) from this repository.
#   bash demo/build_space.sh space/            # then push space/ to your Space's git repo
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/space}"
rm -rf "$OUT" && mkdir -p "$OUT/weights"
cp -r "$ROOT/src" "$ROOT/configs" "$OUT/"
cp "$ROOT/weights/yolo11s.pt" "$OUT/weights/"
cp "$ROOT/demo/app.py" "$OUT/app.py"
cp "$ROOT/demo/requirements.txt" "$OUT/requirements.txt"
cp "$ROOT/demo/space_README.md" "$OUT/README.md"
printf '*.pt filter=lfs diff=lfs merge=lfs -text\n*.jpg filter=lfs diff=lfs merge=lfs -text\n' > "$OUT/.gitattributes"
find "$OUT" -name __pycache__ -prune -exec rm -rf {} +
echo "Space bundle ready in $OUT"
