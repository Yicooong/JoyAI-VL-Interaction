#!/usr/bin/env bash
set -euo pipefail

VIDEO_PATH="${1:-${VIDEO_PATH:-./videos/example.mp4}}"
RTSP_URL="${2:-${RTSP_URL:-rtsp://127.0.0.1:8554/fire1}}"

exec ffmpeg \
  -re \
  -stream_loop -1 \
  -i "$VIDEO_PATH" \
  -vf "scale='min(1280,iw)':-2" \
  -c:v libx264 \
  -preset veryfast \
  -tune zerolatency \
  -b:v 2500k \
  -an \
  -f rtsp \
  -rtsp_transport udp \
  "$RTSP_URL"