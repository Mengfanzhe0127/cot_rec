GPU_ID=$1
GPU_COUNT=$(echo "$GPU_ID" | awk -F',' '{print NF}')

model=/mnt/wangxiaolei/model/Qwen/Qwen3-8B

CUDA_VISIBLE_DEVICES=${GPU_ID} vllm serve ${model} \
  --enable-reasoning \
  --reasoning-parser deepseek_r1 \
  --port "$2" \
  --tensor-parallel-size "${GPU_COUNT}" \
  --swap-space 0 --gpu-memory-utilization 0.95 \
  --disable-log-requests \
  --disable-frontend-multiprocessing --seed 42