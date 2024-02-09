# for "deepseek" "mistral" 
# for eval_name in "gpt-4" "gpt-3.5-turbo" "gemini" "deepseek" "mistral" "deepseek_moe" "gpt-neox" "gpt-j" "mistral_moe" "alpaca" "vicuna" "llama2" 
# do  
#     for lang in "zh-en" "en-de" "yor-en"
#     do
#         CUDA_VISIBLE_DEVICES=4 python3 code/load_models.py -model_name llama2 -task_type "${lang}" -batch_size 1 -eval_name "${eval_name}"
#         echo "llama2 finishes ${eval_name} at ${lang}"
#     done 
# done 

# for name in "deepseek" # "mistral" 
# do
#     for lang in "zh-en" # "en-de" "yor-en"
#     do
#         CUDA_VISIBLE_DEVICES=7 python3 code/load_models.py -model_name "${name}" -task_type "${lang}" -batch_size 1 
#         echo "vicuna finishes ${lang}"
#     done 
# done

# lang="zh-en" # "yor-en" "zh-en"
# for eval_name in "gpt-4" "gpt-3.5-turbo" "gemini" # "deepseek" "mistral" "deepseek_moe" "gpt-neox" "gpt-j" "mistral_moe" "alpaca" "vicuna" "llama2" 
# do  
#     CUDA_VISIBLE_DEVICES=6 python3 code/eval_bleurt.py -lang_dir "${lang}" -model_type "${eval_name}"
#     echo "bleurt finishes ${eval_name} at ${lang}"
# done 