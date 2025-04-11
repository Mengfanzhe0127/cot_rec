#! /bin/bash

# export TOKENIZERS_PARALLELISM=false

GPU_ID=$1
GPU_COUNT=$(echo "$GPU_ID" | awk -F',' '{print NF}')

master_port=$2

model_name_or_path="/mnt/wangxiaolei/model/Qwen/gte-Qwen2-7B-instruct"
train_file="/home/wangxiaolei/mengfanzhe/cot_rec/dataset/filter_user_match/sft_top1_0.2/train_sft_sample-top1-0.2.jsonl"
valid_file="/home/wangxiaolei/mengfanzhe/cot_rec/dataset/filter_user_match/sft_top1_0.2/valid_sft_sample-top1-0.2.jsonl"
test_file="/home/wangxiaolei/mengfanzhe/cot_rec/dataset/filter_user_match/sft_top1_0.2/test_sft_sample-top1-0.2.jsonl"
movie_info_path="/home/wangxiaolei/mengfanzhe/cot_rec/dataset/filter_user_match/sft_top1_0.2/matched_movies_origin.json"
item_max_length=128 # 电影详细信息文本绝大部分100以内
item_batch_size=32
num_negative_samples=32

max_seq_length=1024 # COT文本最大长度956(带模板)
train_batch_size=2
eval_batch_size=2
num_epochs=1
learning_rate=5e-6
min_lr_rate=0.1
warmup_ratio=0.1
weight_decay=0.01
similarity_temperature=0.07
gradient_accumulation_steps=4

wandb_run_name="match_filter_user_sft"

run_dir_suffix="match_filter_user_sft"
timestamp=$(date +"%Y%m%d-%H%M%S")
run_name=match_filter_user_sft+top1-0.2+epoch-${num_epochs}+bs-${train_batch_size}+lr-${learning_rate}+gradient_accumulation_steps-${gradient_accumulation_steps}+num_negative_samples-${num_negative_samples}
log_dir=log/match_filter_user_sft/${run_dir_suffix}/${run_name}_${timestamp}
mkdir -p ${log_dir}

output_dir=outputs/match_filter_user_sft/${run_dir_suffix}/${run_name}_${timestamp}
echo "Output directory: ${output_dir}"

if [ ! -f "/home/wangxiaolei/mengfanzhe/cot_rec/utils/ds_z3_bf16.json" ]; then
    echo "Error: DeepSpeed config file not found"
    exit 1
fi

if [ ! -f "/home/wangxiaolei/mengfanzhe/cot_rec/run_match_like_dislike.py" ]; then
    echo "Error: main script not found"
    exit 1
fi

if [ ! -f "${movie_info_path}" ]; then
    echo "Error: movie info file not found"
    exit 1
fi

mkdir -p ${output_dir}

echo "Starting distributed training with ${GPU_COUNT} GPUs..."

CUDA_VISIBLE_DEVICES=${GPU_ID} torchrun --nproc_per_node="${GPU_COUNT}" --master-port="${master_port}" \
  /home/wangxiaolei/mengfanzhe/cot_rec/run_match_like_dislike.py \
  --seed 42 \
  --model_name_or_path ${model_name_or_path} \
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
  --eval_strategy epoch \
  --gradient_checkpointing True \
  --deepspeed /home/wangxiaolei/mengfanzhe/cot_rec/utils/ds_z3_bf16.json \
  --report_to wandb \
  --similarity_temperature ${similarity_temperature} \
  --movie_info_path ${movie_info_path} \
  --item_max_length ${item_max_length} \
  --num_negative_samples ${num_negative_samples} \
  --wandb_run_name ${wandb_run_name} \
  2>&1 | tee ${log_dir}/${run_name}.log
#   --save_steps \
#   --save_total_limit 1 \
#   --load_best_model_at_end \
#   --metric_for_best_model "mrr@10" \
#   --greater_is_better True \
#   --overwrite_output_dir True \

# --save_only_model \


  # --save_steps 49 \
  # --load_best_model_at_end \
  # --save_total_limit 1 \
  # --metric_for_best_model "recall@10" \
  # --greater_is_better True \
  # --overwrite_output_dir True \