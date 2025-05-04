GPU_ID=$1
GPU_COUNT=$(echo "$GPU_ID" | awk -F',' '{print NF}')

model=/media/public/models/huggingface/Qwen/Qwen2.5-7B-Instruct

CUDA_VISIBLE_DEVICES=${GPU_ID} vllm serve ${model} \
  --port "$2" \
  --tensor-parallel-size "${GPU_COUNT}" \
  --swap-space 0 --gpu-memory-utilization 0.95 \
  --disable-log-requests \
  --disable-frontend-multiprocessing --seed 42
#   --disable-custom-all-reduce --enforce-eager \
#   --trust-remote-code --enable-prefix-caching --seed 42 \
#   --uvicorn-log-level warning --disable-log-requests
