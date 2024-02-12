for model_mt in "madlad400-10b-mt" "nllb-moe-54b"
do
    load_name="model_outputs/${model_mt}/yor-en_100_${model_name}_paraphrase.txt"
    save_name="model_outputs/${model_mt}/yor-en_100_${model_name}_paraphrase_feedback.txt"

    # we generate the corresponding feedback for initial translation
    # "model_outputs-inst/${model_name}/self_refine/${lang}/${model_name}-outputs/${lang}_refinement_100_${model_name}_new_0_rerun.txt" 
    # "model_outputs-inst/${model_name}/self_refine/${lang}/${model_name}-scores/${lang}_eval_100_one-shot_${model_name}_new_0_rerun.txt"  
    OPENAI_API_KEY="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" CUDA_VISIBLE_DEVICES="${device_id}" python3 code/generate_feedback.py -task_type "${task_type}" -lang_dir "${lang}" \
        -api_source "${api_source}" -base_name "${load_name}" -model_type "${model_name}" -savename "${save_name}" -batch_size "${batch_size}" -instructscore_enable "${inst_enable}"

    echo "Initial feedback is done"
done