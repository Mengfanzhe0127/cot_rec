#!/bin/bash

# 参数组合
declare -a param_combinations=(
    "128"
    "64"
)

# 默认 GPU ID 和 master_port
GPU_ID="0,9"
MASTER_PORT="8010"

# 遍历参数组合
for num_negative_samples in "${param_combinations[@]}"; do
    # 构造运行命令
    run_command="bash script/match/match_sample_1.5B_constant.sh $GPU_ID $MASTER_PORT"
    
    # 使用sed命令替换参数值
    sed -i "s/num_negative_samples=[0-9]*/num_negative_samples=${num_negative_samples}/" "script/match/match_sample_1.5B_constant.sh"
    
    # 运行修改后的脚本
    echo "Running with num_negative_samples=${num_negative_samples}..."
    bash script/match/match_sample_1.5B_constant.sh "$GPU_ID" "$MASTER_PORT"
    
    # 可选：将输出重定向到日志文件
    # bash script/match/match_sample_1.5B_constant.sh "$GPU_ID" "$MASTER_PORT" >> "log_${num_negative_samples}.txt"
done