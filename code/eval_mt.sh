for file_name in "model_outputs/madlad400-10b-mt/yor-en_100" "model_outputs/madlad400-10b-mt/yor-en_100_gpt-4_paraphrase" "model_outputs/madlad400-10b-mt/yor-en_100_gpt-3.5-turbo_paraphrase" "model_outputs/madlad400-10b-mt/yor-en_100_gemini_paraphrase" 
do
    for iteration in "0" # "1" "2" "3" "4" "5" # "6" "7" "8" "9" "10" 
    do 
        # first compute all bleurt scores on translaiton outputs
        CUDA_VISIBLE_DEVICES="${device_id}" python3 code/eval_bleurt.py  \
        -file_name "${file_name}.txt" -save_name "${file_name}_bleurt_raw.txt" # "model_outputs-inst/${model_name}/self_refine/${lang}/${model_name}-outputs/${lang}_refinement_100_${model_name}_new_${iteration}_rerun.txt"  "model_outputs-inst/${model_name}/self_refine/${lang}/bleurt-raw/${lang}_refinement_100_${eval_name}_new_${iteration}_rerun_bleurt.txt"

        # first normalize all bleurt scores
        python3 code/quantile_mapping.py -human_file ../.mt-metrics-eval/mt-metrics-eval-v2/wmt22/human-scores/zh-en.mqm.seg.score \
        -obs_score_file ../.mt-metrics-eval/mt-metrics-eval-v2/wmt22/metric-scores/zh-en/BLEURT-20-refA.seg.score \
        -new_score_file "${file_name}_bleurt_raw.txt" -save_name "${file_name}_bleurt_nor.txt" # "model_outputs-inst/${model_name}/self_refine/${lang}/bleurt-raw/${lang}_refinement_100_${eval_name}_new_${iteration}_rerun_bleurt.txt" # "model_outputs-inst/${model_name}/self_refine/${lang}/bleurt-nor/${eval_name}_${iteration}_bleurt_nor.txt"
    done
done

python3 code/compute_bias.py -bleurt_nor_file "model_outputs/madlad400-10b-mt/yor-en_100_bleurt_nor.txt" \
    -llm_score_file "model_outputs/madlad400-10b-mt/yor-en_100_gemini_feedback.txt" 

python3 code/compute_bias.py -bleurt_nor_file "model_outputs/madlad400-10b-mt/yor-en_100_gemini_paraphrase_bleurt_nor.txt" \
    -llm_score_file "model_outputs/madlad400-10b-mt/yor-en_100_gemini_paraphrase_feedback.txt" 