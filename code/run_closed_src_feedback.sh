
echo "gpt-3.5-turbo"
for model_name in "deepseek" "mistral" "deepseek_moe" "gpt-neox" "gpt-j" "mistral_moe" "alpaca" "vicuna" "llama2"
do
    for lang in "zh-en" "en-de" "yor-en"
    do
        OPENAI_API_KEY="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" nohup python3 code/generate_feedback.py -task_type mt -lang_dir "${lang}" -suffix "${model_name}" \
        -api_source openai -base_name "model_outputs/${model_name}/${lang}_base_outputs_${model_name}.txt" \
        -model_type gpt-3.5-turbo &
        echo "${model_name} at ${lang}"
        wait
    done
done