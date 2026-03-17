export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

set -euo pipefail
ROOT_DIR_STORAGE=""; OBELICS_DIR=""

usage(){ echo "Usage: $0 [--root DIR] [--obelics_dir DIR]"; exit 1; }
while [ $# -gt 0 ]; do case "$1" in --root) ROOT_DIR_STORAGE="$2"; shift 2;; --obelics_dir) OBELICS_DIR="$2"; shift 2;; *) usage;; esac; done
prefix(){ printf '%s' "${ROOT_DIR_STORAGE:+${ROOT_DIR_STORAGE%/}/}$1"; }

set -x; PY="$(prefix 'segcompass/bin/python')"; [ -x "$PY" ] || PY=python

"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  --module sae_harness.cache_hiddens \
  --data_dirs "$OBELICS_DIR" \
  --image_dirs "none" \
  --support_bf16 "true" \
  --sae_layer_k 16 \
  --batch_size 16 \
  --llm_version "llava-1.5" \
  --llm_hf "$(prefix 'pretrained_models/llava-1.5-13b-hf')" \
  --hidden_save_dir "$(prefix 'data/sae_hiddens/sae_llava-1.5-13b_L16')"