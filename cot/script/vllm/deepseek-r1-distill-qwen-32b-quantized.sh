GPU_ID=$1
GPU_COUNT=$(echo "$GPU_ID" | awk -F',' '{print NF}')

model=/mnt/wangxiaolei/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8

CUDA_VISIBLE_DEVICES=${GPU_ID} vllm serve ${model} \
  --enable-reasoning \
  --reasoning-parser deepseek_r1 \
  --port "$2" \
  --tensor-parallel-size "${GPU_COUNT}" \
  --disable-log-requests \
  --swap-space 0 --gpu-memory-utilization 0.95 \
  --disable-frontend-multiprocessing --seed 42