export OPENAI_API_BASE="https://open.xiaojingai.com/v1/"
export LITELLM_LOG="DEBUG"
# main
# datasets=("crslab_tgredial" "crslab_redial")

# datasets=("crslab_redial")
# methods=("none" "general" "memory-general")

# for d in  "${datasets[@]}"; do
#     for m in "${methods[@]}"; do
#         echo "Running experiment with dataset: $d, method: $m"
#         python memoCRS_main.py --dataset $d --method $m
#     done
# done

# python memoCRS_main.py --dataset "crslab_tgredial" --method none

# python async_memoCRS_main.py --dataset "crslab_redial" --method none --valid_size 200 --description "gpt4" --model "gpt-4-1106-preview"


# python async_memoCRS_main.py --dataset "crslab_redial" --method general --model "llama" --model_list_path "config/llama_model_list.yaml" --raw_uccr
python async_memoCRS_main.py --dataset "crslab_redial" --method general --model "llama" --model_list_path "config/llama_model_list.yaml" --candidate_num 10
python async_memoCRS_main.py --dataset "crslab_redial" --method general --model "llama" --model_list_path "config/llama_model_list.yaml" --candidate_num 20
python async_memoCRS_main.py --dataset "crslab_redial" --method general --model "llama" --model_list_path "config/llama_model_list.yaml" --candidate_num 40
python async_memoCRS_main.py --dataset "crslab_redial" --method general --model "llama" --model_list_path "config/llama_model_list.yaml" --candidate_num 50
# python async_memoCRS_main.py --dataset "crslab_redial" --method general --valid_size 200 --model "gpt-4-1106-preview"
# python async_memoCRS_main.py --dataset "crslab_redial" --method general  --model "None"
# python async_memoCRS_main.py --dataset "crslab_redial" --method label --description "prompt2.1"