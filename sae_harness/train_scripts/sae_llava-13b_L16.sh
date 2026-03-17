export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

set -euo pipefail
ROOT_DIR_STORAGE=""
RUN_NAME="$(basename "$0" .sh)"
RUN_FLAG="default"

usage(){ echo "Usage: $0 [--root DIR] [--flag NAME]"; exit 1; }
while [ $# -gt 0 ]; do case "$1" in --root) ROOT_DIR_STORAGE="$2"; shift 2;; --flag) RUN_FLAG="$2"; shift 2;; *) usage;; esac; done
prefix(){ printf '%s' "${ROOT_DIR_STORAGE:+${ROOT_DIR_STORAGE%/}/}$1"; }

set -x; PY="$(prefix 'segcompass/bin/python')"; [ -x "$PY" ] || PY=python

"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  --module sae_harness.sae_trainer \
  --config sae_harness/train_scripts/initial.yaml \
  sae_model.d_in=5120 \
  data.cached_dir="$(prefix 'data/sae_hiddens/sae_llava-1.5-13b_L16')" \
  data.batch_size=64 \
  ckpt.save_dir="$(prefix "sae_checkpoints/${RUN_NAME}/${RUN_FLAG}")" \
  ckpt.load_path=null \
  train.max_epochs=10 \
