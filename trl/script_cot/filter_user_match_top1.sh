#!/bin/bash

export TOKENIZERS_PARALLELISM=false

GPU_ID=$1
GPU_COUNT=$(echo "$GPU_ID" | awk -F',' '{print NF}')

master_port=$2

model_name_or_path="/media/public/models/huggingface/Qwen/Qwen2.5-7B-Instruct"
model_name='Qwen/Qwen2.5-7B-Instruct'
dataset_name="/home/wangxiaolei/mengfanzhe/trl-15/trl-test/dataset/filter_user_match_top3_0.3"
response_template='<|im_start|>assistant'

# 训练相关参数
batch_size=2
learning_rate=1e-6
min_lr_rate=0.1
warmup_ratio=0.1
weight_decay=0.01
# 数据集相关参数
dataset_size=-1
max_seq_length=3072 # 最大长度11651，中位数token长度: 2859，平均3252

wandb_run_name="sft_filter_user_match"

gradient_accumulation_steps=1

# 设置输出目录
run_dir_suffix="qwen2.5-7b-instruct-filter_user_match_top3_0.3"
timestamp=$(date +"%Y%m%d-%H%M%S")
log_dir=log/filter_user_match/${run_dir_suffix}_${timestamp}
mkdir -p ${log_dir}

run_name=bs-${batch_size}+gas-${gradient_accumulation_steps}+lr-${learning_rate}+warmup-${warmup_steps}+weight_decay-${weight_decay}+len-${max_seq_length}+top3_0.3
output_dir=outputs/filter_user_match/${run_dir_suffix}_${timestamp}/${run_name}
echo "Output directory: ${output_dir}"

# 检查deepspeed配置文件是否存在
if [ ! -f "/home/wangxiaolei/mengfanzhe/trl-15/trl-test/script/ds_z3_bf16.json" ]; then
    echo "Error: DeepSpeed config file not found at /home/wangxiaolei/mengfanzhe/trl-15/trl-test/script/ds_z3_bf16.json"
    exit 1
fi

# 检查trl脚本是否存在
if [ ! -f "/home/wangxiaolei/mengfanzhe/trl-15/trl-test/src/train/sft_completion.py" ]; then
    echo "Error: SFT script not found at /home/wangxiaolei/mengfanzhe/trl-15/trl-test/src/train/sft_completion.py"
    exit 1
fi

# 创建输出目录
mkdir -p ${output_dir}

# 启动分布式训练
echo "Starting distributed training with ${GPU_COUNT} GPUs..."

CUDA_VISIBLE_DEVICES=${GPU_ID} torchrun --nproc_per_node="${GPU_COUNT}" --master-port="${master_port}" \
  /home/wangxiaolei/mengfanzhe/trl-15/trl-test/src/train/sft_completion.py \
  --seed 42 \
  --model_name_or_path ${model_name_or_path} \
  --bf16 \
  --torch_dtype bfloat16 \
  --attn_implementation flash_attention_2 \
  --dataset_num_proc 4 \
  --response_template ${response_template} \
  --dataset_name ${dataset_name} \
  --dataset_size ${dataset_size} \
  --max_seq_length ${max_seq_length} \
  --save_strategy epoch \
  --output_dir ${output_dir} \
  --logging_dir ${log_dir} \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${batch_size} \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --learning_rate ${learning_rate} \
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
  --dataset_format jsonl \
  --wandb_run_name ${wandb_run_name} \
  2>&1 | tee ${log_dir}/${run_name}.log

#   --test
# --log_level warning \
  # --test \