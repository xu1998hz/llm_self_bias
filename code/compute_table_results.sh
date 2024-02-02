lang="yor-en"
eval_name="gpt-3.5-turbo"
model_name="gpt-3.5-turbo"

for iteration in "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10"
do 
    # first compute all bleurt scores on translaiton outputs
    CUDA_VISIBLE_DEVICES=0 python3 code/eval_bleurt.py -lang_dir yor-en \
    -file_name "model_outputs/${model_name}/self_refine/${lang}/${model_name}-outputs/${lang}_self_refinement_100_${model_name}_new_${iteration}_rerun.txt" \
    -save_name "model_outputs/${model_name}/self_refine/${lang}/bleurt-raw/${lang}_self_refinement_100_${eval_name}_new_${iteration}_rerun_bleurt.txt"

    # first normalize all bleurt scores
    python3 code/quantile_mapping.py -human_file ../.mt-metrics-eval/mt-metrics-eval-v2/wmt22/human-scores/zh-en.mqm.seg.score \
    -obs_score_file ../.mt-metrics-eval/mt-metrics-eval-v2/wmt22/metric-scores/zh-en/BLEURT-20-refA.seg.score \
    -new_score_file "model_outputs/${model_name}/self_refine/${lang}/bleurt-raw/${lang}_self_refinement_100_${eval_name}_new_${iteration}_rerun_bleurt.txt" \
    -save_name "model_outputs/${model_name}/self_refine/${lang}/bleurt-nor/${eval_name}_${iteration}_bleurt_nor.txt"
done

echo "\nAll files are normalized!\n"
# compute bias score for all iterations

for iteration in "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10"
do  
    echo "${model_name} judges ${eval_name} at ${lang}"
    python3 code/compute_bias.py -lang "${lang}" -bleurt_nor_file "model_outputs/${eval_name}/self_refine/${lang}/bleurt-nor/${eval_name}_${iteration}_bleurt_nor.txt" \
    -llm_score_file "model_outputs/${model_name}/self_refine/${lang}/${model_name}-scores/${lang}_eval_100_one-shot_${eval_name}_new_${iteration}_rerun.txt"   
done