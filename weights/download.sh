#!/usr/bin/env bash
# Fetch the detector weights (run once, with internet, before the offline evaluation).
# yolo11s.pt is also committed to the repository, so this is only needed if it is missing.
set -euo pipefail
cd "$(dirname "$0")"
URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
SHA256="85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"
if [ ! -f yolo11s.pt ]; then
  curl -L --fail -o yolo11s.pt "$URL"
fi
echo "$SHA256  yolo11s.pt" | sha256sum -c -
