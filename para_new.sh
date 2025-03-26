#!/bin/bash

# 参数组合
declare -a param_combinations=(
    "1 1e-5"
    "2 1e-5"
    "16 5e-5"
    "32 1e-5"
    "32 5e-5"
    "64 1e-5"
    "64 5e-5"
)

# 默认 GPU ID 和 master_port
GPU_ID="5,6"
MASTER_PORT="8005"

# 遍历参数组合
for combination in "${param_combinations[@]}"; do
    # 解析参数组合
    IFS=' ' read -r gradient_accumulation_steps learning_rate <<< "$combination"
    
    # 构造运行命令
    run_command="bash script/train_classification_qwen_1_epoch_new.sh $GPU_ID $MASTER_PORT"
    
    # 替换参数值
    modified_script=$(mktemp)
    while IFS= read -r line; do
        if [[ "$line" == *"gradient_accumulation_steps="* ]]; then
            line="gradient_accumulation_steps=$gradient_accumulation_steps"
        elif [[ "$line" == *"learning_rate="* ]]; then
            line="learning_rate=$learning_rate"
        fi
        echo "$line" >> "$modified_script"
    done < "script/train_classification_qwen_1_epoch_new.sh"
    
    # 运行修改后的脚本
    bash "$modified_script" "$GPU_ID" "$MASTER_PORT"
    
    # 删除临时文件
    rm "$modified_script"
done