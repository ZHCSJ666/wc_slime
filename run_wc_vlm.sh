#!/bin/bash

export PATH=$CONDA_PREFIX/bin:$PATH

# ==============================
# Basic config
# ==============================
TRAIN_BACKEND=${SLIME_SCRIPT_TRAIN_BACKEND:-"megatron"}
MODEL_NAME=${SLIME_SCRIPT_MODEL_NAME:-"Qwen3-VL-8B-Instruct"}
MODEL_HF_REPO=${SLIME_SCRIPT_MODEL_HF_REPO:-"Qwen/Qwen3-VL-8B-Instruct"}

# 项目根目录（仓库根目录）
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# 依赖路径
MEGATRON_ROOT=${SLIME_SCRIPT_MEGATRON_ROOT:-"/root/Megatron-LM"}

# Hugging Face dataset
HF_DATASET_NAME=${SLIME_SCRIPT_HF_DATASET_NAME:-"ZHCSJ/wc-en-open-slime-4k"}

# 模型、数据、输出目录：都保留 team/longqin
MODEL_ROOT=${SLIME_SCRIPT_MODEL_ROOT:-"/data/oss_bucket_0/users/xintong/team/longqin/models"}
DATASET_ROOT=${SLIME_SCRIPT_DATASET_ROOT:-"/data/oss_bucket_0/users/xintong/team/longqin/datasets"}
OUTPUT_DIR=${SLIME_SCRIPT_OUTPUT_DIR:-"/data/oss_bucket_0/users/xintong/team/longqin/outputs/test_slime"}

DATASET_LOCAL_NAME=${SLIME_SCRIPT_DATASET_LOCAL_NAME:-"wc-en-open-slime-4k"}
OUT_NAME=${SLIME_SCRIPT_OUTPUT_NAME:-"wc_vlm_${MODEL_NAME}"}

# GPU 数
NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-8}

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${DATASET_ROOT}"
mkdir -p "${MODEL_ROOT}"

# ==============================
# Validate required base paths
# ==============================
for _path in \
   "${PROJECT_ROOT}" \
   "${MEGATRON_ROOT}" \
   "${MODEL_ROOT}" \
   "${DATASET_ROOT}" \
   "/data/oss_bucket_0/users/xintong/team/longqin"
do
   if [ ! -d "$_path" ]; then
      echo "Error: 文件地址不存在: $_path"
      exit 1
   fi
done

# ==============================
# Log
# ==============================
if [ "${SLIME_SCRIPT_LOG:-1}" = "1" ]; then
   LOG_DIR=${OUTPUT_DIR}/${OUT_NAME}/logs
   mkdir -p "${LOG_DIR}"
   LOG_FILE=${SLIME_SCRIPT_LOG_FILE:-"${LOG_DIR}/wc_vlm_${MODEL_NAME}.log"}
   exec > >(tee -a "${LOG_FILE}") 2>&1
fi

echo "========== ENV CHECK =========="
conda env list || true
micromamba env list || true
pip list || true
echo "configuration checked"

# ==============================
# Validate MODEL_NAME
# ==============================
VALID_MODELS="
  Qwen2.5-VL-3B-Instruct
  Qwen2.5-VL-7B-Instruct
  Qwen2.5-VL-32B-Instruct
  Qwen2.5-VL-72B-Instruct
  Qwen3-VL-2B-Instruct
  Qwen3-VL-4B-Instruct
  Qwen3-VL-8B-Instruct
  Qwen3-VL-30B-A3B-Instruct
  Qwen3-VL-235B-A22B-Instruct
  Qwen3-VL-2B-Thinking
  Qwen3-VL-4B-Thinking
  Qwen3-VL-8B-Thinking
  Qwen3-VL-30B-A3B-Thinking
  Qwen3-VL-30B-A3B-Thinking-FP8
  Qwen3-VL-235B-A22B-Thinking
"
if ! echo "$VALID_MODELS" | grep -qw "$MODEL_NAME"; then
   echo "Error: MODEL_NAME must be one of: $VALID_MODELS"
   exit 1
fi

MODEL_NAME_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')
MODEL_PATH="${MODEL_ROOT}/${MODEL_NAME}"

# ==============================
# Auto download model if needed
# ==============================
if [ ! -d "${MODEL_PATH}" ]; then
   echo "Local model not found: ${MODEL_PATH}"
   echo "Downloading model from Hugging Face: ${MODEL_HF_REPO}"
   mkdir -p "${MODEL_PATH}"

   if command -v hf >/dev/null 2>&1; then
      hf download "${MODEL_HF_REPO}" --local-dir "${MODEL_PATH}"
   elif command -v huggingface-cli >/dev/null 2>&1; then
      huggingface-cli download "${MODEL_HF_REPO}" --local-dir "${MODEL_PATH}"
   else
      echo "Error: neither 'hf' nor 'huggingface-cli' is installed."
      exit 1
   fi
fi

if [ ! -d "${MODEL_PATH}" ]; then
   echo "Error: model download failed: ${MODEL_PATH}"
   exit 1
fi

echo "Model ready at ${MODEL_PATH}"

# ==============================
# Dataset prepare
# ==============================
TRAIN_FILE="${DATASET_ROOT}/${DATASET_LOCAL_NAME}/train.parquet"

if [ ! -f "${TRAIN_FILE}" ]; then
   echo "Local parquet dataset not found. Preparing dataset from Hugging Face..."
   mkdir -p "${DATASET_ROOT}/${DATASET_LOCAL_NAME}"

   python3 "${PROJECT_ROOT}/scripts/prepare_wc_dataset.py" \
      --hf-dataset "${HF_DATASET_NAME}" \
      --output-dir "${DATASET_ROOT}/${DATASET_LOCAL_NAME}"
fi

if [ ! -f "${TRAIN_FILE}" ]; then
   echo "Error: dataset prepare failed, train parquet not found: ${TRAIN_FILE}"
   exit 1
fi

echo "Dataset ready at ${TRAIN_FILE}"

# ==============================
# External Ray flag
# ==============================
if [ -z "$SLIME_SCRIPT_EXTERNAL_RAY" ] || [ "$SLIME_SCRIPT_EXTERNAL_RAY" = "0" ]; then
   USE_EXTERNAL_RAY=0
else
   USE_EXTERNAL_RAY=1
fi

# ==============================
# Cleanup
# ==============================
pkill -9 sglang || true
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   ray stop --force || true
   pkill -9 ray || true
fi
pkill -9 slime || true
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   pkill -9 ray || true
fi
pkill -9 slime || true
pkill -9 redis || true

set -ex

export PYTHONBUFFERED=16
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ==============================
# Detect NVLink
# ==============================
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ==============================
# Common args
# ==============================
CKPT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
   --rotary-base 5000000
   --save "${OUTPUT_DIR}/${OUT_NAME}"
   --save-interval 10
   --save-debug-rollout-data "${OUTPUT_DIR}/${OUT_NAME}/debug_rollout/rollout_{rollout_id}.pt"
)

# 4k 样本正好一轮：
# rollout-batch-size=32
# num-rollout=125
# 125 * 32 = 4000 prompts
ROLLOUT_ARGS=(
   --prompt-data "${TRAIN_FILE}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle
   --custom-rm-path reward.wc_reward.reward_func
   --num-rollout 125
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --num-steps-per-rollout 1
   --global-batch-size 256
   --rollout-max-response-len 64
   --rollout-temperature 0.8
)

# required for vlm datasets
MULTIMODAL_KEYS='{"image": "images"}'

# 不做 eval
EVAL_ARGS=()

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   --sglang-ep-size 1
   --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64
)

MISC_ARGS=(
   --colocate
)

# ==============================
# Megatron backend args
# 8B 是 dense model，不是 MoE
# ==============================
BACKEND_ARGS=(
   --train-backend megatron
   --load "${MODEL_PATH}"
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --megatron-to-hf-mode bridge
)

# ==============================
# Model args
# 对 Qwen3-VL-8B-Instruct，用 qwen3-8B.sh
# ==============================
MODEL_ARGS_ROTARY_BASE=5000000 source "${PROJECT_ROOT}/scripts/models/qwen3-8B.sh"

# ==============================
# Start Ray if not using external Ray
# ==============================
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
   export no_proxy="127.0.0.1,${MASTER_ADDR}"
   ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

# ==============================
# Build runtime env
# ==============================
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_ROOT}:${PROJECT_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${PROJECT_ROOT}/train.py" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${BACKEND_ARGS[@]} \
   ${MISC_ARGS[@]}