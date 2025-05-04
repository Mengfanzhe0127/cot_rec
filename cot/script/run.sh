export CUDA_VISIBLE_DEVICES=4

# Start vllm serve and redirect output to a log file
vllm serve /media/public/models/huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct > vllm.log 2>&1 &

# Wait for the log file to be created and contain the PID information
sleep 10

# Extract the PID from the log file
VLLM_PID=$(grep "Started engine process with PID" vllm.log | awk '{print $NF}')
echo "VLLM PID: $VLLM_PID"

sleep 200
python memoCRS_main.py

# Kill the specific vllm serve process
kill $VLLM_PID
echo "VLLM process killed"