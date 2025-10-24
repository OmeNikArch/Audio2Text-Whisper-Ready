#!/usr/bin/env bash
set -e
SRC_DIR="audio"
DST_DIR="audio_preproc"

mkdir -p "$DST_DIR"
shopt -s nullglob

for f in "$SRC_DIR"/*.mp3 "$SRC_DIR"/*.wav "$SRC_DIR"/*.m4a "$SRC_DIR"/*.mp4 "$SRC_DIR"/*.ogg; do
  [ -e "$f" ] || continue
  base="$(basename "${f%.*}")"
  echo ">> ${base}"
  ffmpeg -hide_banner -loglevel error -y -i "$f" -ac 1 -ar 16000 \
    -af "highpass=f=300,lowpass=f=3400,afftdn=nr=12:nf=-30, dynaudnorm=f=150:g=15:p=0.9, alimiter=limit=0.9" \
    "$DST_DIR/${base}.wav"
done

echo "Готово. Препроцессированные файлы в: $DST_DIR"
