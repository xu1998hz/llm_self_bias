model_name="gpt-3.5-turbo" # "gemini" # 
api_source="openai" # "google" # 
lang="yor-en"

# we start with initial translation 
OPENAI_API_KEY="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" python3 code/generate_base_output.py -task_type mt -lang_dir yor-en -api_source "${api_source}" -model_type "${model_name}" \
    -save_name "model_outputs/${model_name}/self_refine/${lang}/${model_name}-outputs/${lang}_self_refinement_100_${model_name}_new_0_rerun.txt"

echo "Initial generation is done"

# we generate the corresponding feedback for initial translation
OPENAI_API_KEY="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" python3 code/generate_feedback.py -task_type mt -lang_dir "${lang}" -savename "model_outputs/${model_name}/self_refine/" \
    -api_source "${api_source}" -base_name "model_outputs/${model_name}/self_refine/${lang}/${model_name}-outputs/${lang}_self_refinement_100_${model_name}_new_0_rerun.txt" \
    -model_type "${model_name}" -savename "model_outputs/${model_name}/self_refine/${lang}/${model_name}-scores/${lang}_eval_100_one-shot_${model_name}_new_0_rerun.txt"

echo "Initial feedback is done"

# iterative self-refine: refine the translation based on current feedback and translation, generate corresponding feedback
OPENAI_API_KEY="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" python3 code/refinement.py -lang_dir yor-en -task_type mt \
    -start_index 0 -iteration 10 -api_source "${api_source}" -model_type "${model_name}"
