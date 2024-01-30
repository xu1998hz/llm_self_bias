
# for eval_name in "vicuna" "llama2" "deepseek" "deepseek_moe" "gpt-neox" "gpt-j" "mistral" "mistral_moe" "alpaca"
# do  
#     for lang in "zh-en" "en-de" "yor-en"
#     do
#         CUDA_VISIBLE_DEVICES=7 python3 code/load_models.py -model_name vicuna -task_type "${lang}" -batch_size 1 -eval_name "${eval_name}"
#         echo "vicuna finishes ${eval_name}"
#     done 
# done 

for name in "mistral" "deepseek"
do
    for lang in "zh-en" "en-de" "yor-en"
    do
        CUDA_VISIBLE_DEVICES=7 python3 code/load_models.py -model_name "${name}" -task_type "${lang}" -batch_size 1 
        echo "vicuna finishes ${lang}"
    done 
done