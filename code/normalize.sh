# first normalize all bleurt scores

# for lang in "en-de" # "zh-en" # "en-de" "yor-en"
# do
# for eval_name in "gpt-4" "gpt-3.5-turbo" "gemini" "deepseek" "mistral" "deepseek_moe" "gpt-neox" "gpt-j" "mistral_moe" "alpaca" "vicuna" "llama2" 
# do 
#     python3 code/quantile_mapping.py -human_file /mnt/data3/wendaxu/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/human-scores/zh-en.mqm.seg.score \
#     -obs_score_file  /mnt/data3/wendaxu/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/metric-scores/zh-en/BLEURT-20-refA.seg.score \
#     -new_score_file "model_outputs/${eval_name}/yor-en_base_outputs_${eval_name}_bleurt.txt" \
#     -save_name "model_outputs/${eval_name}/yor-en_base_outputs_${eval_name}_bleurt_nor.txt"
# done
# done

# lang="zh-en"
# for model_name in "deepseek" "mistral" "deepseek_moe" "gpt-neox" "gpt-j" "mistral_moe" "alpaca" "vicuna" "llama2" 
# do
#     for eval_name in "gpt-4" "gpt-3.5-turbo" "gemini" "deepseek" "mistral" "deepseek_moe" "gpt-neox" "gpt-j" "mistral_moe" "alpaca" "vicuna" "llama2" 
#     do 
#         python3 code/quantile_mapping.py -human_file model_outputs/wmt_sys/zh-en_scores_wmt_sys.txt \
#         -obs_score_file  "model_outputs/${model_name}/zh-en_${model_name}_eval_wmt_sys.txt" \
#         -new_score_file "model_outputs/${model_name}/${lang}_${model_name}_eval_${eval_name}.txt" \
#         -save_name "model_outputs/${model_name}/${lang}_${model_name}_eval_${eval_name}_nor.txt"
#     done
# done

lang="yor-en"
for model_name in "gemini" # "mistral" "deepseek" "deepseek_moe" "gpt-neox" "gpt-j" "mistral_moe" "alpaca" "vicuna" "llama2" 
do
    for eval_name in "gpt-4" "gpt-3.5-turbo" "gemini" "deepseek" "mistral" "deepseek_moe" "gpt-neox" "gpt-j" "mistral_moe" "alpaca" "vicuna" "llama2" 
    do  
        echo "${model_name} judges ${eval_name} at ${lang}"
        python3 code/compute_bias.py -lang "${lang}" -model_name "${model_name}" -eval_name "${eval_name}"   
    done
done