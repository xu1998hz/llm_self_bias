# sh code/iterative_refine.sh -m gemini -a transformers -l yor-en -d 0 -b 1 -t mt  -s 10 -i 0 -e True
while getopts "m:a:l:d:t:b:s:i:e:" flag
do
    case "${flag}" in
        m) model_name=${OPTARG};;
        a) api_source=${OPTARG};;
        l) lang=${OPTARG};;
        d) device_id=${OPTARG};;
        t) task_type=${OPTARG};;
        b) batch_size=${OPTARG};;
        s) step=${OPTARG};;
        i) init_step=${OPTARG};;
        e) inst_enable=${OPTARG};;
    esac
done

# mkdir model_outputs
# mkdir model_outputs/${model_name}
# mkdir model_outputs/${model_name}/self_refine/
# mkdir model_outputs/${model_name}/self_refine/${lang}
# mkdir model_outputs/${model_name}/self_refine/${lang}/${model_name}-outputs/
# mkdir model_outputs/${model_name}/self_refine/${lang}/${model_name}-scores/
# mkdir model_outputs/${model_name}/self_refine/${lang}/bleurt-raw/
# mkdir model_outputs/${model_name}/self_refine/${lang}/bleurt-nor/

# mkdir model_outputs
# mkdir model_outputs/${model_name}
# mkdir model_outputs/${model_name}/self_refine/
# mkdir model_outputs/${model_name}/self_refine/${task_type}
# mkdir model_outputs/${model_name}/self_refine/${task_type}/${model_name}-outputs/
# mkdir model_outputs/${model_name}/self_refine/${task_type}/${model_name}-scores/
# mkdir model_outputs/${model_name}/self_refine/${task_type}/bleurt-raw/
# mkdir model_outputs/${model_name}/self_refine/${task_type}/bleurt-nor/

# # we start with initial translation 
# OPENAI_API_KEY="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" CUDA_VISIBLE_DEVICES="${device_id}" python3 code/generate_base_output.py -task_type "${task_type}" -lang_dir "${lang}" -api_source "${api_source}" -model_type "${model_name}" \
#     -save_name "model_outputs/${model_name}/self_refine/${task_type}/${model_name}-outputs/${task_type}_refinement_100_${model_name}_new_0_rerun.txt" -batch_size "${batch_size}"

# echo "Initial generation is done"

# we generate the corresponding feedback for initial translation
# OPENAI_API_KEY="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" CUDA_VISIBLE_DEVICES="${device_id}" python3 code/generate_feedback.py -task_type "${task_type}" -lang_dir "${lang}" \
#     -api_source "${api_source}" -base_name "model_outputs/${model_name}/self_refine/${task_type}/${model_name}-outputs/${task_type}_refinement_100_${model_name}_new_0_rerun.txt" \
#     -model_type "${model_name}" -savename "model_outputs/${model_name}/self_refine/${task_type}/${model_name}-scores/${task_type}_eval_100_one-shot_${model_name}_new_0_rerun.txt"  -batch_size "${batch_size}" -instructscore_enable "${inst_enable}"

# echo "Initial feedback is done"

# iterative self-refine: refine the translation based on current feedback and translation, generate corresponding feedback
OPENAI_API_KEY="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" CUDA_VISIBLE_DEVICES="${device_id}" python3 code/refinement.py -lang_dir "${lang}" -task_type "${task_type}" \
    -start_index "${init_step}" -iteration "${step}" -api_source "${api_source}" -model_type "${model_name}" -instructscore_enable "${inst_enable}"