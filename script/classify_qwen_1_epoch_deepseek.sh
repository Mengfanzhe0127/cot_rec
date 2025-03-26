#! /bin/bash

export TOKENIZERS_PARALLELISM=false

GPU_ID=$1
GPU_COUNT=$(echo "$GPU_ID" | awk -F',' '{print NF}')

master_port=$2

model_name_or_path="/mnt/wangxiaolei/model/Qwen/gte-Qwen2-7B-instruct"
train_file="/home/wangxiaolei/mengfanzhe/cot_rec/dataset/deepseek_0324/train.jsonl"
valid_file="/home/wangxiaolei/mengfanzhe/cot_rec/dataset/deepseek_0324/valid.jsonl"
test_file="/home/wangxiaolei/mengfanzhe/cot_rec/dataset/deepseek_0324/test.jsonl"
movie_name_path="/home/wangxiaolei/mengfanzhe/cot_rec/dataset/deepseek_0324/filtered_movies_deepseek_test.csv"

max_seq_length=512 # COT文本最大长度406
train_batch_size=4
eval_batch_size=4
num_epochs=1
learning_rate=5e-7
gradient_accumulation_steps=1
weight_decay=0.01

run_dir_suffix="result-deepseek-new-1epoch"
timestamp=$(date +"%Y%m%d-%H%M%S")
run_name=new_data+lr-${learning_rate}+gas-${gradient_accumulation_steps}+bs-${train_batch_size}
log_dir=log/qwen/${run_dir_suffix}/${run_name}_${timestamp}
mkdir -p ${log_dir}

output_dir=outputs/qwen/${run_dir_suffix}/${run_name}_${timestamp}
echo "Output directory: ${output_dir}"

if [ ! -f "/home/wangxiaolei/mengfanzhe/cot_rec/utils/ds_z3_bf16.json" ]; then
    echo "Error: DeepSpeed config file not found"
    exit 1
fi

if [ ! -f "/home/wangxiaolei/mengfanzhe/cot_rec/run_recommendation_1_epoch_deepseek.py" ]; then
    echo "Error: main script not found"
    exit 1
fi

mkdir -p ${output_dir}

echo "Starting distributed training with ${GPU_COUNT} GPUs..."

CUDA_VISIBLE_DEVICES=${GPU_ID} torchrun --nproc_per_node="${GPU_COUNT}" --master-port="${master_port}" \
  /home/wangxiaolei/mengfanzhe/cot_rec/run_recommendation_1_epoch_deepseek.py \
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
  --save_strategy epoch \
  --save_only_model \
  --logging_dir ${log_dir} \
  --num_train_epochs ${num_epochs} \
  --per_device_train_batch_size ${train_batch_size} \
  --per_device_eval_batch_size ${eval_batch_size} \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --learning_rate ${learning_rate} \
  --weight_decay ${weight_decay} \
  --lr_scheduler_type constant \
  --max_grad_norm 1.0 \
  --logging_strategy steps \
  --logging_steps 1 \
  --evaluation_strategy epoch \
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

# --save_only_model \

# --save_strategy steps \
#   --save_steps 204 \
#   --load_best_model_at_end \
#   --save_total_limit 1 \
#   --metric_for_best_model "recall@10" \
#   --greater_is_better True \
#   --overwrite_output_dir True \