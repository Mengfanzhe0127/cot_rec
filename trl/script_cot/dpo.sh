#!/bin/bash

GPU_ID=$1
GPU_COUNT=$(echo "$GPU_ID" | awk -F',' '{print NF}')

master_port=$2

model_name_or_path="/media/public/models/huggingface/Qwen/Qwen2.5-7B-Instruct"
dataset_name="/home/wangxiaolei/mengfanzhe/trl-15/trl-test/dataset/dpo_0503"
dataset_train_split="train"


batch_size=2
learning_rate=5e-7
min_lr_rate=0.1
warmup_ratio=0.1
weight_decay=0.01
max_prompt_length=2048
beta=0.1
max_completion_length=2048
max_length=4096

dataset_size=-1
gradient_accumulation_steps=8
wandb_run_name="dpo_filter_user_match"

run_dir_suffix="qwen2.5-7b-instruct-dpo_0503"
timestamp=$(date +"%Y%m%d-%H%M%S")
log_dir=log/dpo_user_match/${run_dir_suffix}_${timestamp}
mkdir -p ${log_dir}

run_name=bs-${batch_size}+lr-${learning_rate}+gas-${gradient_accumulation_steps}+warmup-${warmup_ratio}+weight_decay-${weight_decay}+prompt_length-${max_prompt_length}+completion_length-${max_completion_length}+beta-${beta}+dpo_0503

output_dir=outputs/dpo_user_match/${run_dir_suffix}_${timestamp}/${run_name}
echo "Output directory: ${output_dir}"

# 检查deepspeed配置文件是否存在
if [ ! -f "/home/wangxiaolei/mengfanzhe/trl-15/trl-test/script/ds_z3_bf16.json" ]; then
    echo "Error: DeepSpeed config file not found at /home/wangxiaolei/mengfanzhe/trl-15/trl-test/script/ds_z3_bf16.json"
    exit 1
fi

# 检查trl脚本是否存在
if [ ! -f "/home/wangxiaolei/mengfanzhe/trl-15/trl-test/src/train/dpo.py" ]; then
    echo "Error: SFT script not found at /home/wangxiaolei/mengfanzhe/trl-15/trl-test/src/train/dpo.py"
    exit 1
fi
mkdir -p ${output_dir}

echo "Starting distributed training with ${GPU_COUNT} GPUs..."

CUDA_VISIBLE_DEVICES=${GPU_ID} torchrun --nproc_per_node="${GPU_COUNT}" --master-port="${master_port}" \
  /home/wangxiaolei/mengfanzhe/trl-15/trl-test/src/train/dpo.py \
  --seed 42 \
  --model_name_or_path ${model_name_or_path} \
  --bf16 \
  --torch_dtype bfloat16 \
  --attn_implementation flash_attention_2 \
  --dataset_num_proc 32 \
  --dataset_name ${dataset_name} \
  --dataset_size ${dataset_size} \
  --max_prompt_length ${max_prompt_length} \
  --max_completion_length ${max_completion_length} \
  --max_length ${max_length} \
  --save_strategy epoch \
  --save_only_model True \
  --output_dir ${output_dir} \
  --logging_dir ${log_dir} \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${batch_size} \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --learning_rate ${learning_rate} \
  --beta ${beta} \
  --weight_decay ${weight_decay} \
  --warmup_ratio ${warmup_ratio} \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs "{\"min_lr_rate\": ${min_lr_rate}}" \
  --max_grad_norm 1.0 \
  --logging_strategy steps \
  --logging_steps 1 \
  --gradient_checkpointing True \
  --deepspeed /home/wangxiaolei/mengfanzhe/trl-15/trl-test/script/ds_z3_bf16.json \
  --report_to wandb \
  --disable_dropout True \
  --wandb_run_name ${wandb_run_name} \
  2>&1 | tee ${log_dir}/${run_name}.log