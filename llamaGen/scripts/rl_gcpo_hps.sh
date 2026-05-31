#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

RANDOM_PORT=$(shuf -i 10000-65535 -n 1)
echo "Randomly selected port: $RANDOM_PORT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT_DIR"
echo "$ROOT_DIR"

MASTER_PORT=${RANDOM_PORT}

# ===== paths =====
model_name_or_path="LlamaGen-T2I Path"
data_path="$ROOT_DIR/data/train_data.json"
vq_model_ckpt="vq_ds16_t2i.pt Path"
hps_model_path="HPS_v2.1_compressed.pt Path"
text_tokenizer="flan-t5-xl Path"

# 要指定的最终保存目录
save_dir="Your Save Path"

# ===== speed-oriented settings for 4x A100 80GB =====
per_device_bs=32
dataloader_workers=8

/data/conda_envs/gcpo_llamagen/bin/python -m accelerate.commands.launch \
    --config_file simpar/configs/accelerate_configs/zero2.yaml \
    --main_process_port ${MASTER_PORT} \
    --num_machines 1 \
    --num_processes 4 \
    --machine_rank 0 \
    simpar/train/llamaGen_trainer_gcpo_hps.py \
        --config simpar/configs/config_grpo_hps.yaml \
        --model_name_or_path "$model_name_or_path" \
        --data_path "$data_path" \
        --vq_model_ckpt "$vq_model_ckpt" \
        --hps_model_path "$hps_model_path" \
        --text_tokenizer "$text_tokenizer" \
        --dataset_name hps-data \
        --output_dir "$save_dir" \
        --image_size 256 \
        --downscale_factor 16 \
        --max_steps 900 \
        --save_strategy steps \
        --save_steps 100 \
        --save_total_limit 1 \
        --overwrite_output_dir true \
        --logging_steps 10 \
        --per_device_train_batch_size ${per_device_bs} \
        --gradient_accumulation_steps 1 \
        --dataloader_num_workers ${dataloader_workers} \
        --bf16 true
