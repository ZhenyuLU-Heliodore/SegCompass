######## Devices ########
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

######## bind attention backend with bf16 support ########
SUPP_BF16=true    # <<<<<< A100为true

if [ "$SUPP_BF16" = "true" ]; then export VLLM_ATTENTION_BACKEND=FLASH_ATTN; DTYPE=bfloat16;
else export VLLM_ATTENTION_BACKEND=SDPA; export VLLM_USE_TRITON=0; export XFORMERS_FORCE_DISABLE_TRITON=1; DTYPE=float16; fi

######## function tools ########
set -euo pipefail
ROOT_DIR=""; RUN_FLAG="default"; RUN_NAME="$(basename "$0" .sh)"
usage(){ echo "Usage: $0 [-r|--root DIR] [-f|--run_flag RUN_FLAG]"; exit 1; }
while [ $# -gt 0 ]; do case "$1" in
  -r|--root|-f|--run_flag)
    [ $# -ge 2 ] || usage; key="$1"; val="$2"; shift 2;
    case "$key" in -r|--root) ROOT_DIR="$val";; -f|--run_flag) RUN_FLAG="$val";; esac ;;
  *) usage ;; esac; done
prefix(){ printf '%s' "${ROOT_DIR:+${ROOT_DIR%/}/}$1"; }

######## Disable NCCL InfiniBand/RDMA and wandb ########
export NCCL_IB_DISABLE=1; export WANDB_MODE=offline; export WANDB_DIR="$(prefix 'wandb')"

######## Grouped CLI args ########
ARGS=(
  ### paths ###
  data.train_files="$(prefix 'data/grefcoco/grefcoco_train')"
  data.sam_embed_dir="$(prefix 'data/refcoco_series_sam_embed')"
  worker.actor.model.model_path="$(prefix 'pretrained_models/SegCompass-llava-13b-init')"
  worker.actor.model.init_sae_ckpt="$(prefix 'sae_checkpoints/sae_llava-13b_L16/default/ep_4.pt')"

  trainer.save_checkpoint_path="$(prefix "checkpoints/${RUN_NAME}/${RUN_FLAG}")"
  worker.actor.model.init_sam_ckpt="$(prefix 'pretrained_models/sam_vit_h_4b8939.pth')"
  trainer.load_checkpoint_path=null
  config="scripts/initial.yaml"

  ### multi-slots and SAE ###
  worker.sae.d_in=5120
  worker.llm_version="llava-1.5"
  worker.actor.model.k_slots=4
  worker.sae.hook_layer=16

  ### batch size ###   (global_batch_size * n) / nnodes 要被 micro_batch_size_per_device 整除
  worker.rollout.n=8
  data.rollout_batch_size=16
  worker.actor.global_batch_size=16
  worker.actor.micro_batch_size_per_device_for_update=2
  worker.actor.micro_batch_size_per_device_for_experience=8

  ### rollout ###
  worker.rollout.tensor_parallel_size=4
  worker.rollout.max_num_seqs=64
  worker.rollout.gpu_memory_utilization=0.6
  worker.rollout.dtype=${DTYPE}
  worker.rollout.enable_chunked_prefill=${SUPP_BF16}

  ### trainer ###
  trainer.n_gpus_per_node=8
  trainer.total_episodes=4
  trainer.save_freq=400
  worker.supp_bf16=${SUPP_BF16}

  ### loss coef###
  worker.actor.optim.base_lr=1.6e-6
  worker.actor.seg_loss_coef=0.3
  worker.actor.conf_loss_coef=0.2
  worker.actor.kl_loss_coef=0.2
  worker.actor.adjust_loss_step=1500
  worker.actor.entropy_coef=0.0
  worker.actor.dice_loss_coef_new=2.0
  worker.actor.focal_loss_coef_new=5.0
)

set -x; PY="$(prefix 'segcompass/bin/python')"; [ -x "$PY" ] || PY=python
PYTHONUNBUFFERED=1 "$PY" -u -m verl.trainer.main "${ARGS[@]}" 2>&1 | tee -a "$(prefix 'print_log.txt')"