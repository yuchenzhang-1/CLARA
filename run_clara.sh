#!/usr/bin/env bash
set -euo pipefail


export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-...}
N_GPUS=${N_GPUS:-...}
SCRIPT=main.py



DATASET_ROOT=${DATASET_ROOT:-...}
DATASET_NAME=${DATASET_NAME:-...}
SPLIT_JSON=${SPLIT_JSON:-...}


BATCH_SIZE=${BATCH_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-0}
EPOCHS=${EPOCHS:-50}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
EARLY_PATIENCE=${EARLY_PATIENCE:-10}
EARLY_THRESH=${EARLY_THRESH:-0.0}
BEST_METRIC=${BEST_METRIC:-acc}
USE_TRANSFORMER=${USE_TRANSFORMER:-true}
USE_CL=${USE_CL:-true}

FOLD=${FOLD:-0}
BUDGET=${BUDGET:-40}
RAT_SOURCE=${RAT_SOURCE:-Qwen}
TEXT_EMB_MODEL=${TEXT_EMB_MODEL:-bert}

LR=${LR:-3e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}

PROJ_OUT_DIM=${PROJ_OUT_DIM:-256}
BACKEND=${BACKEND:-moe}
MOE_NUM_EXPERTS=${MOE_NUM_EXPERTS:-8}
MOE_TOP_K=${MOE_TOP_K:-3}
MOE_RA_MODE=${MOE_RA_MODE:-scalar}


GVT_NUM_LAYERS=${GVT_NUM_LAYERS:-4}
GVT_NUM_HEADS=${GVT_NUM_HEADS:-8}
GVT_FFN_MULT=${GVT_FFN_MULT:-4}
GVT_RATIONALE_MODE=${GVT_RATIONALE_MODE:-both}


SEG_MODE=${SEG_MODE:-independent}
SEG_NUM_PAIRS=${SEG_NUM_PAIRS:-1}
SEG_L_RATIO=${SEG_L_RATIO:-0.2}
SEG_G_RATIO=${SEG_G_RATIO:-0.8}
CONTRAST_TAU=${CONTRAST_TAU:-0.1}
CL_WEIGHT=${CL_WEIGHT:-0.3}



DATE=$(date +%y%m%d)
TIME=$(date +%H%M%S)



OUTDIR="...."


mkdir -p "${OUTDIR}"
MOE_LOG_DIR="${OUTDIR}/logs_moe"
GVT_LOG_DIR="${OUTDIR}/logs_gvt"
mkdir -p "${MOE_LOG_DIR}" "${GVT_LOG_DIR}"
LOG_FILE="${OUTDIR}/training.log"


COMMON_ARGS=(
  --do_train
  --do_eval
  --do_predict
  --bf16
  --use_transformer "${USE_TRANSFORMER}"
  --use_contrastive "${USE_CL}"
  --dataset_root "${DATASET_ROOT}"
  --dataset_name "${DATASET_NAME}"
  --split_json "${SPLIT_JSON}"
  --fold "${FOLD}"
  --budget "${BUDGET}"
  --rationale_source "${RAT_SOURCE}"
  --map_location "${MAP_LOCATION}"
  --text_emb_model "${TEXT_EMB_MODEL}"
  --output_dir "${OUTDIR}"

  --learning_rate "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --num_train_epochs "${EPOCHS}"
  --warmup_ratio "${WARMUP_RATIO}"
  
  --per_device_train_batch_size "${BATCH_SIZE}"
  --per_device_eval_batch_size "${BATCH_SIZE}"
  --clip_microbatch_size "${MICRO_BATCH_SIZE}"
  --early_stopping_patience "${EARLY_PATIENCE}"
  --early_stopping_threshold "${EARLY_THRESH}"
  --proj_out_dim "${PROJ_OUT_DIM}"
  --clip_encoder_backend "${BACKEND}"
  
  --moe_num_experts "${MOE_NUM_EXPERTS}"
  --moe_top_k "${MOE_TOP_K}"

  --gvt_num_layers "${GVT_NUM_LAYERS}"
  --gvt_num_heads "${GVT_NUM_HEADS}"
  --gvt_ffn_mult "${GVT_FFN_MULT}"
  --gvt_rationale_mode "${GVT_RATIONALE_MODE}"
  --gvt_use_cls true
  --gvt_use_mean_pool false

  --seg_mode "${SEG_MODE}"
  --seg_num_pairs_per_video "${SEG_NUM_PAIRS}"
  --seg_l_ratio "${SEG_L_RATIO}"
  --seg_g_ratio "${SEG_G_RATIO}"
  --contrast_tau "${CONTRAST_TAU}"
  --contrastive_weight "${CL_WEIGHT}"

  --dataloader_pin_memory true
  --dataloader_num_workers 8
  --eval_strategy epoch
  --save_strategy epoch
  --save_total_limit 1
  --logging_strategy steps
  --logging_steps 50
  --load_best_model_at_end true
  --metric_for_best_model "${BEST_METRIC}"
  --greater_is_better true
  --save_last false
  --moe_log_dir "${MOE_LOG_DIR}"
  --gvt_log_dir "${GVT_LOG_DIR}"
  --remove_unused_columns false
)




if [ "${N_GPUS}" -gt 1 ]; then
  torchrun --standalone --nnodes=1 --nproc_per_node="${N_GPUS}" "${SCRIPT}" "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1
else
  python "${SCRIPT}" "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1
fi