# sh code/iterative_refine.sh -m mistral-inst2 -mqm -a transformers -l yor-en -d 6
while getopts "m:a:l:d:" flag
do
    case "${flag}" in
        m) model_name=${OPTARG};;
        a) api_source=${OPTARG};;
        l) lang=${OPTARG};;
        d) device_id=${OPTARG};;
    esac
done

mkdir model_outputs/${model_name}
mkdir model_outputs/${model_name}/self_refine/
mkdir model_outputs/${model_name}/self_refine/${lang}
mkdir model_outputs/${model_name}/self_refine/${lang}/${model_name}-outputs/
mkdir model_outputs/${model_name}/self_refine/${lang}/${model_name}-scores/
mkdir model_outputs/${model_name}/self_refine/${lang}/bleurt-raw/
mkdir model_outputs/${model_name}/self_refine/${lang}/bleurt-nor/

# we start with initial translation 
OPENAI_API_KEY="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" CUDA_VISIBLE_DEVICES="${device_id}" python3 code/generate_base_output.py -task_type mt -lang_dir "${lang}" -api_source "${api_source}" -model_type "${model_name}" \
    -save_name "model_outputs/${model_name}/self_refine/${lang}/${model_name}-outputs/${lang}_refinement_100_${model_name}_new_0_rerun.txt"

echo "Initial generation is done"

# we generate the corresponding feedback for initial translation
OPENAI_API_KEY="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" CUDA_VISIBLE_DEVICES="${device_id}" python3 code/generate_feedback.py -task_type mt -lang_dir "${lang}" \
-api_source "${api_source}" -base_name "model_outputs/${model_name}/self_refine/${lang}/${model_name}-outputs/${lang}_refinement_100_${model_name}_new_0_rerun.txt" \
-model_type "${model_name}" -savename "model_outputs/${model_name}/self_refine/${lang}/${model_name}-scores/${lang}_eval_100_one-shot_${model_name}_new_0_rerun.txt"

echo "Initial feedback is done"

# iterative self-refine: refine the translation based on current feedback and translation, generate corresponding feedback
OPENAI_API_KEY="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" CUDA_VISIBLE_DEVICES="${device_id}" python3 code/refinement.py -lang_dir "${lang}" -task_type mt \
    -start_index 0 -iteration 10 -api_source "${api_source}" -model_type "${model_name}"
