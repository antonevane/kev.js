#!/usr/bin/env bash
# Build, verify and package one Kev model at an exact Hub commit:
#
#   ./build_model.sh kev-4b [jaredpalmer/kev-4b[@rev]]
#
# Steps: pin -> merge (LoRA folded in fp32) -> int8 WebGPU build(s) -> 32 MB shards -> fp32 ONNX reference ->
# PyTorch fixtures -> parity -> 300-record reference set -> package. Intermediates are deleted once used: a 9B build
# peaks near 85 GB. Every artefact records the pinned run, and package.py refuses fixtures from another commit.
#
# Environment (all optional):
#   KEV_BUILD_DIR     where the intermediates go (default: build, next to this script), e.g. a roomier external disk
#   KEV_WEB_VARIANTS  the WebGPU variants to build, space-separated from: q8f32 q8 (default: per model, below)
#   KEV_RUN_SOURCE    a local, checksum-verified copy of the pinned run's files (kev_web_export/pin.py source())
set -euo pipefail
cd "$(dirname "$0")"
name=$1
run=$(uv run python -m kev_web_export.pin "${2:-jaredpalmer/$name}" 2>/dev/null | tail -1)
build_dir=${KEV_BUILD_DIR:-build}
out=$build_dir/$name
log() { echo "[$name] $*"; }
log "pinned $run"

case $name in
  kev-0.8b) web=(q8f32 q8) fixtures=40 keep_fp32=1 ;;   # fp32 stays in the bundle: the Node tests' exact variant
  *)        web=(q8f32)    fixtures=20 keep_fp32=0 ;;
esac
src() { case $1 in q8f32) echo q8f32-webgpu ;; q8) echo q8f16-webgpu ;; esac; }   # bash 3.2: no associative arrays
if [ -n "${KEV_WEB_VARIANTS:-}" ]; then
  read -r -a web <<< "$KEV_WEB_VARIANTS"
  for v in "${web[@]}"; do
    [ -n "$(src "$v")" ] || { echo "[$name] KEV_WEB_VARIANTS: unknown variant '$v' (known: q8f32 q8)" >&2; exit 2; }
  done
fi

log "merge"
rm -rf "$out"
uv run python -m kev_web_export.merge --run "$run" --out "$out"

for v in "${web[@]}"; do
  log "build $(src "$v")"
  ./build.sh "$out" "$(src "$v")"
  log "shard $v (rope -> SERVE_MAX_BRANCH)"
  uv run python -m kev_web_export.postprocess --src "$out/onnx-$(src "$v")" --out "$out/web-$v"
  rm -rf "$out/onnx-$(src "$v")"
done
log "build fp32 reference"
./build.sh "$out" fp32-cpu
rm -rf "$out/merged"

log "fixtures (PyTorch)"
uv run python -m kev_web_export.fixtures --run "$run" --n "$fixtures" --out "../fixtures/$name.json" | tail -1
log "parity"
uv run python -m kev_web_export.parity --model "$out/onnx-fp32-cpu/model.onnx" --head "$out/head.safetensors" --fixtures "../fixtures/$name.json" | tail -1
for v in "${web[@]}"; do
  uv run python -m kev_web_export.parity --model "$out/web-$v/model.onnx" --head "$out/head.safetensors" --fixtures "../fixtures/$name.json" | tail -1
done
log "300-record reference"
uv run python -m kev_web_export.evalset --model "$out/onnx-fp32-cpu/model.onnx" --head "$out/head.safetensors" \
  --base "$(python3 -c "import json;print(json.load(open('$out/kev.json'))['base'])")" \
  --base_revision "$(python3 -c "import json;print(json.load(open('$out/kev.json'))['base_revision'])")" \
  --out "../fixtures/$name-transfer-v4-dev300.json" | tail -1

log "package"
vargs=(); for v in "${web[@]}"; do vargs+=(--variant "$v=web-$v"); done
[ "$keep_fp32" = 1 ] && vargs+=(--variant fp32=onnx-fp32-cpu)
uv run python -m kev_web_export.package --build "$out" --out "../public/models/$name" "${vargs[@]}" --fixtures "../fixtures/$name.json" | tail -2
[ "$keep_fp32" = 1 ] || rm -rf "$out/onnx-fp32-cpu"
log "done"
