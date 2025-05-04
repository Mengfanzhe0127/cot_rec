GPU_ID=$1
GPU_COUNT=$(echo "$GPU_ID" | awk -F',' '{print NF}')

model=/mnt/wangxiaolei/model/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8

CUDA_VISIBLE_DEVICES=${GPU_ID} vllm serve ${model} \
  --port "$2" \
  --tensor-parallel-size "${GPU_COUNT}" \
  --swap-space 0 --gpu-memory-utilization 0.95 \
  --disable-log-requests \
  --disable-frontend-multiprocessing --seed 42