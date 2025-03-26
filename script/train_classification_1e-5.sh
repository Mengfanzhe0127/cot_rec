#! /bin/bash

export TOKENIZERS_PARALLELISM=false

GPU_ID=$1
GPU_COUNT=$(echo "$GPU_ID" | awk -F',' '{print NF}')

master_port=$2

model_name_or_path="/mnt/wangxiaolei/model/Qwen/gte-Qwen2-7B-instruct"
train_file="/home/wangxiaolei/mengfanzhe/cot_rec/dataset/qwen/new_train_qwen.jsonl"
valid_file="/home/wangxiaolei/mengfanzhe/cot_rec/dataset/qwen/new_valid_qwen.jsonl"
test_file="/home/wangxiaolei/mengfanzhe/cot_rec/dataset/qwen/new_test_qwen.jsonl"
movie_name_path="/home/wangxiaolei/mengfanzhe/cot_rec/dataset/filtered_movies.csv"

max_seq_length=1536 # COT文本最大长度1500
train_batch_size=64
eval_batch_size=64
num_epochs=5
learning_rate=1e-5
min_lr_rate=0.1
warmup_ratio=0.1
weight_decay=0.01

run_dir_suffix="gte-Qwen2-7B-instruct"
timestamp=$(date +"%Y%m%d-%H%M%S")
run_name=bs-${train_batch_size}+lr-${learning_rate}+warmup-${warmup_ratio}+weight_decay-${weight_decay}
log_dir=log/${run_dir_suffix}/${run_name}_${timestamp}
mkdir -p ${log_dir}

output_dir=outputs/${run_dir_suffix}/${run_name}_${timestamp}
echo "Output directory: ${output_dir}"

if [ ! -f "/home/wangxiaolei/mengfanzhe/cot_rec/utils/ds_z3_bf16.json" ]; then
    echo "Error: DeepSpeed config file not found"
    exit 1
fi

if [ ! -f "/home/wangxiaolei/mengfanzhe/cot_rec/run_recommendation.py" ]; then
    echo "Error: main script not found"
    exit 1
fi

mkdir -p ${output_dir}

echo "Starting distributed training with ${GPU_COUNT} GPUs..."

CUDA_VISIBLE_DEVICES=${GPU_ID} torchrun --nproc_per_node="${GPU_COUNT}" --master-port="${master_port}" \
  /home/wangxiaolei/mengfanzhe/cot_rec/run_recommendation.py \
  --seed 42 \
  --model_name_or_path ${model_name_or_path} \
  --movie_name_path ${movie_name_path} \
  --train_file ${train_file} \
  --validation_file ${valid_file} \
  --test_file ${test_file} \
  --output_dir ${output_dir} \
  --bf16 \
  --torch_dtype bfloat16 \
  --attn_implementation flash_attention_2 \
  --max_seq_length ${max_seq_length} \
  --do_train \
  --do_eval \
  --do_predict \
  --save_only_model \
  --save_strategy epoch \
  --logging_dir ${log_dir} \
  --num_train_epochs ${num_epochs} \
  --per_device_train_batch_size ${train_batch_size} \
  --per_device_eval_batch_size ${eval_batch_size} \
  --gradient_accumulation_steps 1 \
  --learning_rate ${learning_rate} \
  --weight_decay ${weight_decay} \
  --warmup_ratio ${warmup_ratio} \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs "{\"min_lr_rate\": ${min_lr_rate}}" \
  --max_grad_norm 1.0 \
  --logging_strategy steps \
  --logging_steps 1 \
  --evaluation_strategy steps \
  --eval_steps 41 \
  --gradient_checkpointing True \
  --deepspeed /home/wangxiaolei/mengfanzhe/cot_rec/utils/ds_z3_bf16.json \
  --report_to wandb \
  2>&1 | tee ${log_dir}/${run_name}.log

#   --save_steps \
#   --save_total_limit 1 \
#   --load_best_model_at_end \
#   --metric_for_best_model "mrr@10" \
#   --greater_is_better True \
#   --overwrite_output_dir True \